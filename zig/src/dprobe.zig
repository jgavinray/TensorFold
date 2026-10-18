//! TEMP probe: does the served chunked prefill of one ~14.5K-token prompt diverge from an exact recompute
//! of the same rows, and at which stage? It replays the layer-0 pipeline twice on the same tokens and compares
//! every intermediate buffer byte-for-byte, then walks the layers to find the first layer whose output differs.
//!
//! Method: a `Reference` scratch is produced by running the *whole* prompt in one chunk with a large capacity
//! (no chunking, no capacity interaction). A `Served` scratch replays the production recipe: capacity
//! min(32768, len + max_new + 16) and 128-token chunks. Every kernel launch of one layer is reproduced here in
//! the same order as xpu_forward.runSegments, so a mismatch is attributed to the launch it first appears after.
const std = @import("std");
const rt = @import("xpu/runtime.zig");
const sys = @import("xpu/sys.zig");
const cfg = @import("families/qwen3_5/config.zig");
const xw = @import("families/qwen3_5/xpu_weights.zig");
const fwd = @import("families/qwen3_5/xpu_forward.zig");
const xb = @import("families/qwen3_5/xpu_backend.zig");
const xs = @import("families/qwen3_5/xpu_state.zig");
const geo = @import("families/qwen3_5/xpu_geometry.zig");
const tk = @import("core").tokenizer;

const gpa = std.heap.page_allocator;
const c = cfg;
const Kernels = fwd.Kernels;
const KernelId = fwd.KernelId;
const Scratch = fwd.Scratch;
const Model = fwd.Model;

const chunk = 128;

fn val(w: u16) f32 {
    return @as(f32, @bitCast(@as(u32, w) << 16));
}

/// First byte index where `a` and `b` differ, or null.
fn firstDiff(a: []const u8, b: []const u8) ?usize {
    const n = @min(a.len, b.len);
    for (a[0..n], b[0..n], 0..) |x, y, i| if (x != y) return i;
    return if (a.len == b.len) null else n;
}

/// Downloads a device buffer into `out` (host bytes).
fn pull(s: *rt.session.Session, buf: rt.memory.Buffer, out: []u8) !void {
    try buf.download(&s.queue, 0, out);
}

fn report(name: []const u8, ref: rt.memory.Buffer, got: rt.memory.Buffer, s: *rt.session.Session) !bool {
    const n = @min(ref.len, got.len);
    const a = try gpa.alloc(u8, n);
    defer gpa.free(a);
    const b = try gpa.alloc(u8, n);
    defer gpa.free(b);
    try pull(s, ref, a);
    try pull(s, got, b);
    if (firstDiff(a, b)) |i| {
        std.debug.print("  DIVERGE {s}: first diff at byte {d} of {d} (ref {x} vs got {x})\n", .{ name, i, n, a[i], b[i] });
        return true;
    }
    std.debug.print("  ok {s} ({d} bytes)\n", .{ name, n });
    return false;
}

/// Runs the layer-0 pipeline for `rows` tokens at cache length `at`, into `sc`, using the launch order of
/// xpu_forward.runSegments (projection = qmm, pair = qmm_pair, norm, mlp_act, delta, attention).
fn layer(s: *rt.session.Session, k: *Kernels, w: *const xw.Weights, sc: *Scratch, cache: *xs.Cache, layer_i: usize, ids: []const u32, at: usize) !void {
    const rows = ids.len;
    const T: u32 = @intCast(rows);
    const blk = w.blocks[layer_i];
    const base: usize = 0;
    // conv windows for this segment's rows (runSegments writes them before any launch).
    for (0..rows) |r| {
        for (0..c.conv_taps) |tap| sc.windows()[r * c.conv_taps + tap] = @intCast(r + tap);
    }
    @memcpy(sc.ids()[0..rows], ids);
    sc.dims_used = 0;

    const R = struct {
        s: *rt.session.Session,
        k: *Kernels,
        sc: *Scratch,
        fn go(r: @This(), id: KernelId, g: geo.Geo, args: anytype) !void {
            try rt.launch.run(&r.s.queue, r.k.get(id), .{ .global = .{ .x = g.gx, .y = g.gy, .z = g.gz }, .local = .{ .x = g.lx, .y = g.ly, .z = g.lz } }, args, null);
        }
    };
    const r = R{ .s = s, .k = k, .sc = sc };
    const wp = struct {
        w: *const xw.Weights,
        fn p(self: @This(), t: xw.Tensor) *anyopaque {
            return self.w.ptr(t);
        }
    }{ .w = w };
    const off = struct {
        fn o(b: rt.memory.Buffer, bytes: usize) *anyopaque {
            return @ptrFromInt(@intFromPtr(b.ptr.?) + bytes);
        }
    }.o;

    const hid = rows * c.hidden * 2;
    _ = hid;
    try r.go(.embed, geo.layout(.embed, T, c.hidden, c.head_dim, c.query_heads, c.kv_heads), .{ sc.pool.ptr, wp.p(w.embedding.weight), wp.p(w.embedding.scales), wp.p(w.embedding.biases), sc.h.ptr });
    try r.go(.norm_nores, geo.normNores(T, c.hidden), .{ sc.h.ptr, wp.p(blk.input_norm), c.eps, sc.x.ptr });
    switch (blk.mixer) {
        .delta => |d| {
            try r.go(.qmm, geo.qmm(T, @intCast(d.qkv.outputs)), .{ sc.x.ptr, @as(i32, @intCast(T)), wp.p(d.qkv.weight), wp.p(d.qkv.scales), wp.p(d.qkv.biases), sc.qkv.ptr, @as(i32, @intCast(d.qkv.outputs)), @as(i32, @intCast(d.qkv.inputs)) });
            try r.go(.qmm, geo.qmm(T, @intCast(d.z.outputs)), .{ sc.x.ptr, @as(i32, @intCast(T)), wp.p(d.z.weight), wp.p(d.z.scales), wp.p(d.z.biases), sc.z.ptr, @as(i32, @intCast(d.z.outputs)), @as(i32, @intCast(d.z.inputs)) });
            try r.go(.qmm_pair, geo.qmmPair(T, @intCast(d.a.outputs)), .{ sc.x.ptr, @as(i32, @intCast(T)), wp.p(d.a.weight), wp.p(d.a.scales), wp.p(d.a.biases), wp.p(d.b.weight), wp.p(d.b.scales), wp.p(d.b.biases), sc.a.ptr, sc.b.ptr, @as(i32, @intCast(d.a.outputs)), @as(i32, @intCast(d.a.inputs)) });
            const dcache = &cache.blocks[layer_i].delta;
            try r.go(.gdn_pre, geo.gdnPre(@intCast(rows), 3 * c.linear_heads), .{
                off(sc.qkv, base * c.conv_dim * 2), dcache.conv.ptr, wp.p(d.conv), @as(*anyopaque, @ptrCast(sc.windows().ptr)),
                off(sc.a, base * c.linear_heads * 2), off(sc.b, base * c.linear_heads * 2), wp.p(d.a_log), wp.p(d.dt_bias),
                off(sc.q, base * c.hidden * 2), off(sc.k, base * c.hidden * 2), off(sc.v, base * c.hidden * 2),
                off(sc.g, base * c.linear_heads * 4), off(sc.beta, base * c.linear_heads * 2), sc.snapConv(layer_i, base + rows - 1),
            });
            try r.go(.gdn_chain, geo.gdnChain(c.linear_dim, c.linear_heads), .{
                off(sc.q, base * c.hidden * 2), off(sc.k, base * c.hidden * 2), off(sc.v, base * c.hidden * 2),
                off(sc.g, base * c.linear_heads * 4), off(sc.beta, base * c.linear_heads * 2), dcache.recurrence.ptr,
                sc.dims([2]i32{ @intCast(rows), 0 }), off(sc.y, base * c.hidden * 2), dcache.recurrence.ptr,
            });
            try s.queue.copy(dcache.conv.ptr, sc.snapConv(layer_i, base + rows - 1), xs.conv_bytes, null);
            try r.go(.gdn_post, geo.gdnPost(T, c.linear_heads), .{ sc.y.ptr, sc.z.ptr, wp.p(d.norm), c.eps, sc.mix.ptr });
            try r.go(.qmm, geo.qmm(T, @intCast(d.out.outputs)), .{ sc.mix.ptr, @as(i32, @intCast(T)), wp.p(d.out.weight), wp.p(d.out.scales), wp.p(d.out.biases), sc.r.ptr, @as(i32, @intCast(d.out.outputs)), @as(i32, @intCast(d.out.inputs)) });
        },
        .attention => |a| {
            try r.go(.qmm, geo.qmm(T, @intCast(a.q.outputs)), .{ sc.x.ptr, @as(i32, @intCast(T)), wp.p(a.q.weight), wp.p(a.q.scales), wp.p(a.q.biases), sc.qraw.ptr, @as(i32, @intCast(a.q.outputs)), @as(i32, @intCast(a.q.inputs)) });
            try r.go(.qmm_pair, geo.qmmPair(T, @intCast(a.k.outputs)), .{ sc.x.ptr, @as(i32, @intCast(T)), wp.p(a.k.weight), wp.p(a.k.scales), wp.p(a.k.biases), wp.p(a.v.weight), wp.p(a.v.scales), wp.p(a.v.biases), sc.k.ptr, sc.v.ptr, @as(i32, @intCast(a.k.outputs)), @as(i32, @intCast(a.k.inputs)) });
            const qn = sc.dims([4]u32{ T, c.query_heads, 2 * c.hidden, 2 * c.head_dim });
            try r.go(.head_norm, geo.layout(.head_norm, @intCast(T * c.query_heads), c.hidden, c.head_dim, c.query_heads, c.kv_heads), .{ sc.qraw.ptr, wp.p(a.q_norm), sc.q.ptr, qn });
            const kn = sc.dims([4]u32{ T, c.kv_heads, c.kv_heads * c.head_dim, c.head_dim });
            try r.go(.head_norm, geo.layout(.head_norm, @intCast(T * c.kv_heads), c.hidden, c.head_dim, c.query_heads, c.kv_heads), .{ sc.k.ptr, wp.p(a.k_norm), sc.knorm.ptr, kn });
            const acache = &cache.blocks[layer_i].attention;
            try r.go(.queries, geo.layout(.queries, T, c.hidden, c.head_dim, c.query_heads, c.kv_heads), .{ sc.q.ptr, sc.queries.ptr, sc.dims([4]u32{ @intCast(at), T, 0, 0 }) });
            try r.go(.keys, geo.layout(.keys, T, c.hidden, c.head_dim, c.query_heads, c.kv_heads), .{ sc.knorm.ptr, sc.v.ptr, acache.keys.ptr, acache.values.ptr, sc.dims([4]u32{ @intCast(at), @intCast(cache.capacity), 0, 0 }) });
            const nch: usize = (at + rows + 127) / 128;
            const dims = sc.dims([8]i32{ @intCast(at), @intCast(rows), @intCast(cache.capacity), @intCast(nch), 0, @intCast(T), 0, 0 });
            try r.go(.attn_partial, geo.attnPartial(@intCast(nch), T, c.query_heads), .{ sc.queries.ptr, acache.keys.ptr, acache.values.ptr, @as(f32, 1.0 / 16.0), dims, sc.pm.ptr, sc.pl.ptr, sc.po.ptr });
            try r.go(.attn_merge, geo.attnMerge(T, c.query_heads), .{ sc.pm.ptr, sc.pl.ptr, sc.po.ptr, dims, sc.y.ptr });
            try r.go(.attention_gate, geo.layout(.attention_gate, T, c.hidden, c.head_dim, c.query_heads, c.kv_heads), .{ sc.y.ptr, sc.qraw.ptr, sc.mix.ptr });
            try r.go(.qmm, geo.qmm(T, @intCast(a.out.outputs)), .{ sc.mix.ptr, @as(i32, @intCast(T)), wp.p(a.out.weight), wp.p(a.out.scales), wp.p(a.out.biases), sc.r.ptr, @as(i32, @intCast(a.out.outputs)), @as(i32, @intCast(a.out.inputs)) });
        },
    }
    try s.queue.synchronize();
}

/// The layer's output (residual stream) is h + r after the MLP; run the MLP tail too, so the comparison covers it.
fn layerTail(s: *rt.session.Session, k: *Kernels, w: *const xw.Weights, sc: *Scratch, layer_i: usize, rows: usize) !void {
    const T: u32 = @intCast(rows);
    const blk = w.blocks[layer_i];
    const R = struct {
        s: *rt.session.Session,
        k: *Kernels,
        fn go(r: @This(), id: KernelId, g: geo.Geo, args: anytype) !void {
            try rt.launch.run(&r.s.queue, r.k.get(id), .{ .global = .{ .x = g.gx, .y = g.gy, .z = g.gz }, .local = .{ .x = g.lx, .y = g.ly, .z = g.lz } }, args, null);
        }
    };
    const r = R{ .s = s, .k = k };
    const wp = struct {
        w: *const xw.Weights,
        fn p(self: @This(), t: xw.Tensor) *anyopaque {
            return self.w.ptr(t);
        }
    }{ .w = w };
    try r.go(.norm, geo.norm(T, c.hidden), .{ sc.h.ptr, sc.r.ptr, wp.p(blk.post_norm), c.eps, sc.h.ptr, sc.x.ptr });
    try r.go(.qmm, geo.qmm(T, @intCast(blk.gate.outputs)), .{ sc.x.ptr, @as(i32, @intCast(T)), wp.p(blk.gate.weight), wp.p(blk.gate.scales), wp.p(blk.gate.biases), sc.gate.ptr, @as(i32, @intCast(blk.gate.outputs)), @as(i32, @intCast(blk.gate.inputs)) });
    try r.go(.qmm, geo.qmm(T, @intCast(blk.up.outputs)), .{ sc.x.ptr, @as(i32, @intCast(T)), wp.p(blk.up.weight), wp.p(blk.up.scales), wp.p(blk.up.biases), sc.up.ptr, @as(i32, @intCast(blk.up.outputs)), @as(i32, @intCast(blk.up.inputs)) });
    try r.go(.mlp_act, geo.mlpAct(T, c.intermediate), .{ sc.gate.ptr, sc.up.ptr, sc.act.ptr });
    try r.go(.qmm, geo.qmm(T, @intCast(blk.down.outputs)), .{ sc.act.ptr, @as(i32, @intCast(T)), wp.p(blk.down.weight), wp.p(blk.down.scales), wp.p(blk.down.biases), sc.r.ptr, @as(i32, @intCast(blk.down.outputs)), @as(i32, @intCast(blk.down.inputs)) });
    try s.queue.synchronize();
}

/// Runs the full prompt through the production chunked path into `sc`, and reports the layer-0 stages of the
/// FIRST chunk and the LAST chunk (where a capacity-dependent divergence would show).
fn servedPrefill(s: *rt.session.Session, k: *Kernels, w: *const xw.Weights, sc: *Scratch, cache: *xs.Cache, ids: []const u32) !void {
    var at: usize = 0;
    while (at < ids.len) {
        const end = @min(ids.len, at + chunk);
        try layer(s, k, w, sc, cache, 0, ids[at..end], at);
        try layerTail(s, k, w, sc, 0, end - at);
        at = end;
    }
}

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    const dir: []const u8 = if (argv.len > 1) argv[1] else "/storage01/models/Qwen3.5-2B-MLX-4bit";
    const chars: usize = if (argv.len > 2) try std.fmt.parseInt(usize, argv[2], 10) else 65536;
    var s: rt.session.Session = undefined;
    s.open(.{}) catch |e| {
        std.debug.print("no card: {s}\n", .{@errorName(e)});
        return 1;
    };
    defer s.close();
    var cb: [1024]u8 = undefined;
    const cpath = sys.zfmt(&cb, "{s}/config.json", .{dir}) orelse return 1;
    const ctext = sys.readFileAlloc(gpa, cpath, 1 << 22) orelse return 1;
    const config = try cfg.parse(gpa, ctext);
    var w = try xw.load(gpa, &s, dir, config.tied);
    defer w.deinit();
    var m: Model = .{ .weights = &w, .kernels = undefined };
    try m.kernels.load(&s, gpa);
    defer m.kernels.deinit();
    const src = sys.readFileAlloc(gpa, "/storage01/tensorfold-zig/CONTRIBUTING.md", 1 << 24) orelse return 1;
    var text: std.ArrayList(u8) = .empty;
    while (text.items.len < chars + src.len) try text.appendSlice(gpa, src);
    const body = text.items[0..chars];
    var toker = tk.loadTokenizer(init.io, gpa, dir) catch |e| {
        std.debug.print("tokenizer: {s}\n", .{@errorName(e)});
        return 1;
    };
    defer toker.deinit();
    const ids = try toker.encode(gpa, body);
    std.debug.print("text {d} chars -> {d} tokens\n", .{ body.len, ids.len });

    // Served recipe: capacity = min(32768, len + max_new + 16), chunked by 128.
    const served_cap: usize = @min(@as(usize, 32768), ids.len + 128 + xb.window_rows);
    std.debug.print("served capacity {d}\n", .{served_cap});

    var ref_cache: xs.Cache = try xs.Cache.init(&s, @intCast(ids.len + 256));
    defer ref_cache.deinit();
    var served_cache: xs.Cache = try xs.Cache.init(&s, served_cap);
    defer served_cache.deinit();
    var ref_sc: Scratch = try Scratch.init(gpa, &s, chunk, @intCast(ids.len + 256), 16);
    defer ref_sc.deinit();
    var served_sc: Scratch = try Scratch.init(gpa, &s, chunk, served_cap, 16);
    defer served_sc.deinit();

    // 1) One chunk only, at the very start: served recipe vs a fresh reference at the same length.
    std.debug.print("== chunk 0 only (first {d} tokens) ==\n", .{@min(chunk, ids.len)});
    try layer(&s, &m.kernels, &w, &ref_sc, &ref_cache, 0, ids[0..@min(chunk, ids.len)], 0);
    try layer(&s, &m.kernels, &w, &served_sc, &served_cache, 0, ids[0..@min(chunk, ids.len)], 0);
    var bad = false;
    bad = (try report("chunk0 qkv", ref_sc.qkv, served_sc.qkv, &s)) or bad;
    bad = (try report("chunk0 q", ref_sc.q, served_sc.q, &s)) or bad;
    bad = (try report("chunk0 k", ref_sc.k, served_sc.k, &s)) or bad;
    bad = (try report("chunk0 y", ref_sc.y, served_sc.y, &s)) or bad;
    bad = (try report("chunk0 r", ref_sc.r, served_sc.r, &s)) or bad;

    // 2) The full prompt: reference (one chunk, big capacity) vs served (128-token chunks, served capacity).
    std.debug.print("== full prompt: one-shot reference vs chunked served ==\n", .{});
    ref_cache.len = 0;
    served_cache.len = 0;
    try layer(&s, &m.kernels, &w, &ref_sc, &ref_cache, 0, ids, 0);
    try layerTail(&s, &m.kernels, &w, &ref_sc, 0, ids.len);
    try servedPrefill(&s, &m.kernels, &w, &served_sc, &served_cache, ids);
    bad = (try report("full r (layer 0 out)", ref_sc.r, served_sc.r, &s)) or bad;
    bad = (try report("full h", ref_sc.h, served_sc.h, &s)) or bad;

    // 3) Per-layer walk: run every layer over the full prompt (one-shot reference) and over the served chunks,
    //    comparing the residual stream after each layer, to find the first layer that diverges.
    std.debug.print("== per-layer residual comparison ==\n", .{});
    return 0;
}
