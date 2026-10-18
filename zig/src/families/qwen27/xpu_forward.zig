//! Qwen 3.8 27B forward pass on an Intel B70 (Level Zero + OpenCL kernels), mirroring forward.zig's op order and rounding
//! points (bf16 activations between every op, fp32 recurrence) and the verified Qwen3.5-2B XPU forward
//! (families/qwen3_5/xpu_forward.zig). One source serves both models: the shared kernels under
//! zig/kernels/xpu/shared/<key>/ carry the 2B shape values as preprocessor defaults and the 27B build passes the
//! model-shape overrides as -D flags (build option `qwen-shape-defines`, see zig/build/xpu.zig). The launch geometry
//! (families/qwen3_5/xpu_geometry.zig) takes the runtime shape values, so one geometry table serves both.
//! Kernels are embedded in the executable at build time (@embedFile of the ocloc outputs), so the binary works from any
//! working directory; TF_XPU_QWEN35_KERNELS names a directory of <module key>.bin files that replaces them.
//! One segment per call (prefill rows or a decode row appended to the cache); no snapshot recording.
const std = @import("std");
pub const rt = @import("../../xpu/runtime.zig");
pub const sys = @import("../../xpu/sys.zig");
pub const cfg = @import("config.zig");
pub const xw = @import("xpu_weights.zig");
pub const xs = @import("xpu_state.zig");
const kbins = @import("xpu_qwen35_kernels");
const cfgmod = cfg;
const c = struct {
    pub const hidden = cfgmod.xpu_hidden;
    pub const vocab = cfgmod.xpu_vocab;
    pub const layers = cfgmod.xpu_layers;
    pub const intermediate = cfgmod.xpu_intermediate;
    pub const heads = cfgmod.xpu_heads;
    pub const kv_heads = cfgmod.xpu_kv_heads;
    pub const head_dim = cfgmod.xpu_head_dim;
    pub const k_heads = cfgmod.xpu_k_heads;
    pub const v_heads = cfgmod.xpu_v_heads;
    pub const dk = cfgmod.xpu_dk;
    pub const dv = cfgmod.xpu_dv;
    pub const conv_kernel = cfgmod.xpu_conv_kernel;
    pub const eps = cfgmod.xpu_eps;
    /// q_proj's output width: 2 * heads * head_dim (the qraw row stride and the attention_gate q stride).
    pub fn qDim() usize {
        return 2 * heads * head_dim;
    }
    /// k_proj's / v_proj's output width (the KV cache row width).
    pub fn kvDim() usize {
        return kv_heads * head_dim;
    }
};
const geo = @import("../qwen3_5/xpu_geometry.zig");

const Buffer = rt.memory.Buffer;
const Kernel = rt.module.Kernel;
const Dim = rt.launch.Dim3;
pub const Head = enum { last, all };
pub const Error = rt.Error || std.mem.Allocator.Error || error{ QwenRowsExceeded, QwenContextFull, InvalidQwenToken, KernelBinaryMissing, ReplayNeedsOneRow };
extern "c" fn getenv(name: [*:0]const u8) ?[*:0]const u8;

/// True when `name` is set and not "0" (used by `Kernels.load` for the TF_XPU_QWEN35_KERNELS override).
pub fn getenvBool(name: [*:0]const u8) bool {
    const v = getenv(name) orelse return false;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// Module keys (file names under shared/<key>/) and the kernels each provides.
const Module = enum { layout, norm, norm_nores, qmm, mlp_act, gdn_pre, gdn_chain, gdn_post, attn_partial, attn_merge };
pub const KernelId = enum {
    embed, head_norm, queries, keys, attention_gate, norm, norm_nores, qmm, mlp_act, gdn_pre, gdn_chain, gdn_post, attn_partial, attn_merge,

    fn module(self: KernelId) Module {
        return switch (self) {
            .embed, .head_norm, .queries, .keys, .attention_gate => .layout,
            inline else => |t| @field(Module, @tagName(t)),
        };
    }
    fn function(self: KernelId) [:0]const u8 {
        return "qwen35_" ++ switch (self) {
            inline else => |t| @tagName(t),
        };
    }
};
const module_count = std.enums.values(Module).len;
const kernel_count = std.enums.values(KernelId).len;

/// The compiled module's bytes, embedded in the executable by the build graph (`zig/build/xpu.zig`).
fn builtinBytes(comptime m: Module) []const u8 {
    return &@field(kbins, @tagName(m));
}

pub const Kernels = struct {
    modules: [module_count]rt.module.Module,
    loaded: usize = 0,
    list: [kernel_count]Kernel,
    made: usize = 0,

    pub fn load(self: *Kernels, s: *rt.session.Session, gpa: std.mem.Allocator) !void {
        self.loaded = 0;
        self.made = 0;
        errdefer self.deinit();
        const dir: ?[:0]const u8 = if (getenv("TF_XPU_QWEN35_KERNELS")) |p| std.mem.span(p) else null;
        inline for (std.enums.values(Module), 0..) |mod, i| {
            const fname = @tagName(mod);
            var buf: [1024]u8 = undefined;
            // TF_XPU_QWEN35_KERNELS (development) reads <dir>/<key>.bin; otherwise the module embedded at build time.
            var file: ?[]u8 = null;
            const bin: []const u8 = if (dir) |d| blk: {
                const path = sys.zfmt(&buf, "{s}/{s}.bin", .{ d, fname }) orelse return error.KernelBinaryMissing;
                file = sys.readFileAlloc(gpa, path, 64 << 20) orelse {
                    std.debug.print("cannot read kernel module {s}\n", .{path});
                    return error.KernelBinaryMissing;
                };
                break :blk file.?;
            } else builtinBytes(mod);
            defer if (file) |b| gpa.free(b);
            var log: rt.module.BuildLog = .{};
            self.modules[i] = rt.module.Module.load(&s.ctx, bin, &log) catch |e| {
                std.debug.print("module {s} load failed: {s}\n", .{ fname, log.text() });
                return e;
            };
            self.loaded += 1;
        }
        inline for (std.enums.values(KernelId), 0..) |id, i| {
            self.list[i] = try self.modules[@intFromEnum(id.module())].kernel(comptime id.function());
            self.made += 1;
        }
    }

    pub fn deinit(self: *Kernels) void {
        for (self.list[0..self.made]) |*k| k.deinit();
        for (self.modules[0..self.loaded]) |*m| m.unload();
        self.made = 0;
        self.loaded = 0;
    }

    fn get(self: *Kernels, id: KernelId) *Kernel {
        return &self.list[@intFromEnum(id)];
    }
    /// One line per module (kept for parity with the profiler output).
    pub fn describe(self: *const Kernels) void {
        std.debug.print("xpu qwen27 kernels: shared ({d} modules)", .{self.loaded});
        inline for (std.enums.values(Module)) |m| std.debug.print(" {s}", .{@tagName(m)});
        std.debug.print("\n", .{});
    }
};

/// Intermediate buffers (device), plus a small host-visible pool for ids, conv windows and kernel `dims` arrays
/// (every launch gets its own slot, so the whole forward can be queued before the host waits).
pub const Scratch = struct {
    rows: usize,
    capacity: usize,
    logit_rows: usize,
    cfg: cfgmod.Config,
    pool: Buffer,
    h: Buffer,
    x: Buffer,
    r: Buffer,
    z: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    y: Buffer,
    mix: Buffer,
    queries: Buffer,
    qkv: Buffer,
    gate: Buffer,
    up: Buffer,
    act: Buffer,
    qraw: Buffer,
    knorm: Buffer,
    a: Buffer,
    b: Buffer,
    beta: Buffer,
    g: Buffer,
    pm: Buffer,
    pl: Buffer,
    po: Buffer,
    logits: Buffer,
    conv_snap: Buffer,
    /// Per-layer window snapshots for shared rounds, indexed by the absolute scratch row of the segment's last row.
    snaps_delta: [cfgmod.max_layers]?Buffer = @splat(null),
    snaps_conv: [cfgmod.max_layers]?Buffer = @splat(null),
    made: usize = 0,
    /// Snapshot buffers made so far, for `deinit`.
    made_snaps: usize = 0,
    dims_used: usize = 0,
    /// While a decode step is being recorded, every `dims` slot index lands here so the replay can rewrite the
    /// per-token scalars inside the captured arrays. Null outside recording.
    record_slots: ?*std.ArrayList(usize) = null,
    /// Allocator for `record_slots` appends; the caller's, since the scratch does not own one otherwise.
    gpa: std.mem.Allocator = undefined,

    const names = .{ "pool", "h", "x", "r", "z", "q", "k", "v", "y", "mix", "queries", "qkv", "gate", "up", "act", "qraw", "knorm", "a", "b", "beta", "g", "pm", "pl", "po", "logits", "conv_snap" };
    pub const dims_slots = 1024;
    pub const slot_bytes = 32;

    pub fn init(s: *rt.session.Session, config: cfgmod.Config, rows: usize, capacity: usize, logit_rows: usize) rt.Error!Scratch {
        return initGpa(std.heap.page_allocator, s, config, rows, capacity, logit_rows);
    }

    pub fn initGpa(gpa: std.mem.Allocator, s: *rt.session.Session, config: cfgmod.Config, rows: usize, capacity: usize, logit_rows: usize) rt.Error!Scratch {
        var out: Scratch = undefined;
        out.rows = rows;
        out.capacity = capacity;
        out.logit_rows = logit_rows;
        out.cfg = config;
        out.gpa = gpa;
        out.record_slots = null;
        out.made = 0;
        out.made_snaps = 0;
        out.dims_used = 0;
        out.snaps_delta = @splat(null);
        out.snaps_conv = @splat(null);
        errdefer out.deinit();
        const chunks = (capacity + 127) / 128;
        const hid = rows * config.hidden * 2;
        // Every activation row buffer is sized to the widest writer, not to `hidden`: at 2B the GDN value width
        // (v_heads*dv) and the query width (heads*head_dim) both happen to equal `hidden`, so `rows*hidden` was
        // always enough; at 27B they are 6144 vs hidden 5120, and sizing by `hidden` under-allocated `q`, `v`,
        // `y`, `mix` and `queries` by 20% - a silent device-buffer overrun (and a page-fault risk). The Metal
        // frame (gpu_frame.zig) sizes by the same maxima; these match them.
        const mix_w = @max(config.hidden, config.gdnValueDim()); // gdn_post writes rows*v_heads*dv, attention_gate rows*hidden
        const q_w = @max(config.hidden, config.heads * config.head_dim); // head_norm writes rows*heads*head_dim
        const v_w = @max(config.hidden, config.gdnValueDim()); // gdn_pre writes rows*v_heads*dv
        const rows_mix = rows * mix_w * 2;
        const rows_q = rows * q_w * 2;
        const rows_v = rows * v_w * 2;
        const sizes = [_]usize{
            rows * 4 + rows * config.conv_kernel * 4 + dims_slots * slot_bytes, hid, hid, hid, rows_v, rows_q, hid, rows_v, rows_mix, rows_mix, rows_q,
            rows * config.intermediate * 2, rows * config.intermediate * 2, rows * config.intermediate * 2, rows * config.intermediate * 2,
            rows * config.qDim() * 2, rows * config.kvDim() * 2,
            rows * config.v_heads * 2, rows * config.v_heads * 2, rows * config.v_heads * 2, rows * config.v_heads * 4,
            rows * config.heads * chunks * 4, rows * config.heads * chunks * 4, rows * config.heads * chunks * config.head_dim * 4,
            logit_rows * config.vocab * 2, rows * xs.convBytes(config),
        };
        inline for (names, 0..) |n, i| {
            const kind: rt.memory.Kind = if (i == 0) .host else .device;
            @field(out, n) = try Buffer.alloc(&s.ctx, kind, sizes[i]);
            out.made = i + 1;
        }
        // One recurrence and one conv snapshot buffer per linear layer, `rows` rows wide (one per window row).
        for (&out.snaps_delta, 0..) |*b, i| if (config.kind(i) == .linear) {
            b.* = try Buffer.alloc(&s.ctx, .device, rows * xs.deltaBytes(config));
            out.made_snaps = i + 1;
        };
        for (&out.snaps_conv, 0..) |*b, i| if (config.kind(i) == .linear) {
            b.* = try Buffer.alloc(&s.ctx, .device, rows * xs.convBytes(config));
        };
        return out;
    }

    pub fn deinit(self: *Scratch) void {
        for (self.snaps_delta[0..self.made_snaps]) |*b| if (b.*) |buf| {
            var x = buf;
            x.free();
            b.* = null;
        };
        for (&self.snaps_conv) |*b| if (b.*) |buf| {
            var x = buf;
            x.free();
            b.* = null;
        };
        self.made_snaps = 0;
        inline for (names, 0..) |n, i| if (i < self.made) @field(self, n).free();
        self.made = 0;
    }

    fn ids(self: *Scratch) []u32 {
        return @as([*]u32, @ptrCast(@alignCast(self.pool.ptr.?)))[0..self.rows];
    }
    fn windows(self: *Scratch) []i32 {
        return @as([*]i32, @ptrCast(@alignCast(@as([*]u8, @ptrCast(self.pool.ptr.?)) + self.rows * 4)))[0 .. self.rows * c.conv_kernel];
    }
    fn dimsBase(self: *Scratch) [*]u8 {
        return @as([*]u8, @ptrCast(self.pool.ptr.?)) + self.rows * 4 + self.rows * c.conv_kernel * 4;
    }
    /// Copies `vals` into a fresh slot of the pool and returns its device-visible address. With a recorder active the
    /// slot's index is appended to `record_slots` so the replay can rewrite the per-token scalars inside it.
    fn dims(self: *Scratch, vals: anytype) *anyopaque {
        const bytes = std.mem.asBytes(&vals);
        std.debug.assert(bytes.len <= slot_bytes and self.dims_used < dims_slots);
        const p = self.dimsBase() + self.dims_used * slot_bytes;
        @memcpy(p[0..bytes.len], bytes);
        if (self.record_slots) |list| list.append(self.gpa, self.dims_used) catch {};
        self.dims_used += 1;
        return @ptrCast(p);
    }

    /// The recurrence snapshot of `layer` at absolute scratch row `row` (`deltaBytes` long); `keep` reads it back.
    pub fn snapDelta(self: *Scratch, layer: usize, row: usize) *anyopaque {
        const b = self.snaps_delta[layer] orelse unreachable;
        std.debug.assert(row < self.rows);
        return @ptrFromInt(@intFromPtr(b.ptr.?) + row * xs.deltaBytes(self.cfg));
    }
    /// The conv-tail snapshot of `layer` at absolute scratch row `row` (`convBytes` long); `keep` reads it back.
    pub fn snapConv(self: *Scratch, layer: usize, row: usize) *anyopaque {
        const b = self.snaps_conv[layer] orelse unreachable;
        std.debug.assert(row < self.rows);
        return @ptrFromInt(@intFromPtr(b.ptr.?) + row * xs.convBytes(self.cfg));
    }
};

pub const Model = struct {
    weights: *const xw.Weights,
    kernels: Kernels,
};

const Run = struct {
    s: *rt.session.Session,
    k: *Kernels,
    w: *const xw.Weights,
    sc: *Scratch,
    /// When set, launches are recorded into this regular list instead of executed on the immediate queue.
    rec: ?*rt.cmdlist.Recorder = null,

    fn go(self: Run, id: KernelId, g: geo.Geo, args: anytype) rt.Error!void {
        const global: Dim = .{ .x = g.gx, .y = g.gy, .z = g.gz };
        const local: Dim = .{ .x = g.lx, .y = g.ly, .z = g.lz };
        const launch_cfg = rt.launch.Config{ .global = global, .local = local };
        if (self.rec) |r| {
            try r.run(self.k.get(id), launch_cfg, args);
            return;
        }
        try rt.launch.run(&self.s.queue, self.k.get(id), launch_cfg, args, null);
    }
    fn wp(self: Run, t: xw.Tensor) *anyopaque {
        return self.w.ptr(t);
    }
    fn projection(self: Run, l: xw.Linear, x: *anyopaque, y: *anyopaque, rows: usize) rt.Error!void {
        try self.go(.qmm, geo.qmm(@intCast(rows), @intCast(l.outputs)), .{
            x, @as(i32, @intCast(rows)), self.wp(l.weight), self.wp(l.scales), self.wp(l.biases), y, @as(i32, @intCast(l.outputs)), @as(i32, @intCast(l.inputs)),
        });
    }
    fn norm(self: Run, h: *anyopaque, residual: ?*anyopaque, weight: xw.Tensor, x: *anyopaque, rows: usize) rt.Error!void {
        if (residual) |r| {
            try self.go(.norm, geo.norm(@intCast(rows), c.hidden), .{ h, r, self.wp(weight), c.eps, h, x });
        } else {
            try self.go(.norm_nores, geo.normNores(@intCast(rows), c.hidden), .{ h, self.wp(weight), c.eps, x });
        }
    }
};

fn off(b: Buffer, bytes: usize) *anyopaque {
    return @ptrFromInt(@intFromPtr(b.ptr.?) + bytes);
}

pub const Options = struct {
    /// Stop after this layer's MLP (no final norm / head): the layer output is h + r in the scratch.
    stop_after: ?usize = null,
    head: Head = .last,
    /// When set, the forward *records* into this regular command list instead of executing: no work reaches the card
    /// and nothing is synchronized. The caller closes the list and replays it (see `xpu_replay.zig`).
    rec: ?*rt.cmdlist.Recorder = null,
    /// Receives the pool slot index of every `dims` array the recorded step allocates, so the replay can find the
    /// per-token scalars (position, cache length) inside them. Only used when `rec` is set.
    slots: ?*std.ArrayList(usize) = null,
};

/// A stream's rows in one shared round: its cache and how many rows it contributes, in segment order.
pub const Segment = struct { cache: *xs.Cache, rows: usize };

/// A `dims` pool slot a recorded decode step reads, and which of its elements moves per token.
pub const DimsSlot = struct {
    /// Byte offset of the slot from the pool's start.
    offset: usize,
    /// Index of the element that holds the RoPE position, or null.
    position_at: ?usize = null,
    /// Index of the element that holds the cache length, or null.
    cache_len_at: ?usize = null,
};

/// Streams one shared round serves at most (one segment a stream); the snapshot scratch has this many slots.
pub const max_segments = xs.max_segments;

/// Forward `tokens` as one segment appended to `cache` (no snapshot recording). Blocks until the card has finished.
/// On return `cache.len` has advanced and `cache.logits` holds the last row's bf16 logits (unless `stop_after` is
/// set); with head = .all `scratch.logits` holds every row.
pub fn run(s: *rt.session.Session, m: *Model, sc: *Scratch, cache: *xs.Cache, tokens: []const u32, o: Options) Error!void {
    return runSegments(s, m, sc, &.{.{ .cache = cache, .rows = tokens.len }}, tokens, false, o);
}

/// One decode token recorded into `rec` as a regular command list: the same launches as `run` with a single token and
/// `head = .last`, but no work reaches the card and nothing is synchronized. `slots` receives one entry per `dims`
/// array the step allocates, with the element index of every scalar that changes per token.
pub fn recordDecode(
    s: *rt.session.Session,
    rec: *rt.cmdlist.Recorder,
    m: *Model,
    sc: *Scratch,
    cache: *xs.Cache,
    tokens: []const u32,
    position: u32,
    slots: *std.ArrayList(DimsSlot),
) Error!void {
    if (tokens.len != 1) return error.ReplayNeedsOneRow;
    var capture: std.ArrayList(usize) = .empty;
    defer capture.deinit(sc.gpa);
    sc.record_slots = &capture;
    defer sc.record_slots = null;
    try runSegments(s, m, sc, &.{.{ .cache = cache, .rows = 1 }}, tokens, false, .{ .head = .last, .rec = rec, .slots = &capture });
    const pool: [*]u8 = @ptrCast(sc.pool.ptr.?);
    const base = @intFromPtr(sc.dimsBase()) - @intFromPtr(sc.pool.ptr.?);
    for (capture.items) |slot_i| {
        const d: [*]const i32 = @ptrCast(@alignCast(pool + base + slot_i * Scratch.slot_bytes));
        var slot: DimsSlot = .{ .offset = base + slot_i * Scratch.slot_bytes };
        for (0..8) |j| {
            const v: u32 = @intCast(@max(d[j], 0));
            if (v == position and slot.position_at == null) slot.position_at = j;
            if (v == cache.len and slot.cache_len_at == null) slot.cache_len_at = j;
        }
        try slots.append(sc.gpa, slot);
    }
}

/// Forward `tokens` split across `segments`, each its own cache and row range, in one launch set (the shared-round
/// forward). With `record`, every row of a linear layer's recurrence and each segment's conv tail are saved so a
/// later `keep` can roll back; `segments.len == 1 and !record` is the plain single-stream path.
pub fn runSegments(s: *rt.session.Session, m: *Model, sc: *Scratch, segments: []const Segment, tokens: []const u32, record: bool, o: Options) Error!void {
    var total: usize = 0;
    for (segments) |seg| {
        if (seg.rows == 0 or seg.cache.len + seg.rows > seg.cache.capacity) return error.QwenContextFull;
        if (seg.cache.capacity > sc.capacity) return error.QwenContextFull;
        total += seg.rows;
    }
    if (total == 0 or total != tokens.len or total > sc.rows or (o.head == .all and total > sc.logit_rows)) return error.QwenRowsExceeded;
    if (record and (segments.len > max_segments or total > xs.max_segments * xs.window_rows)) return error.QwenRowsExceeded;
    if (!record and segments.len != 1) return error.QwenRowsExceeded;
    for (tokens) |id| if (id >= c.vocab) return error.InvalidQwenToken;
    @memcpy(sc.ids()[0..total], tokens);
    // Each segment's conv windows are relative to its own rows, and its scratch row is the running `base`.
    var base: usize = 0;
    for (segments) |seg| {
        for (0..seg.rows) |r| for (0..c.conv_kernel) |tap| {
            sc.windows()[(base + r) * c.conv_kernel + tap] = @intCast(r + tap);
        };
        base += seg.rows;
    }
    sc.dims_used = 0;
    sc.record_slots = o.slots;
    defer sc.record_slots = null;
    const ru = Run{ .s = s, .k = &m.kernels, .w = m.weights, .sc = sc, .rec = o.rec };
    const w = m.weights;
    const T: u32 = @intCast(total);
    try ru.go(.embed, geo.layout(.embed, T, c.hidden, c.head_dim, c.heads, c.kv_heads), .{ sc.pool.ptr, w.ptr(w.embed.weight), w.ptr(w.embed.scales), w.ptr(w.embed.biases), sc.h.ptr });
    for (w.layers, 0..) |layer, i| {
        try ru.norm(sc.h.ptr.?, if (i == 0) null else sc.r.ptr, layer.input_norm, sc.x.ptr.?, total);
        switch (layer.mixer) {
            .linear => |d| try delta(ru, d, i, segments, record, total),
            .attention => |a| try attention(ru, a, i, segments, total),
        }
        try ru.norm(sc.h.ptr.?, sc.r.ptr, layer.post_norm, sc.x.ptr.?, total);
        try ru.projection(layer.mlp.gate, sc.x.ptr.?, sc.gate.ptr.?, total);
        try ru.projection(layer.mlp.up, sc.x.ptr.?, sc.up.ptr.?, total);
        try ru.go(.mlp_act, geo.mlpAct(T, c.intermediate), .{ sc.gate.ptr, sc.up.ptr, sc.act.ptr });
        try ru.projection(layer.mlp.down, sc.act.ptr.?, sc.r.ptr.?, total);
        if (o.stop_after) |n| if (i == n) {
            if (o.rec == null) try s.queue.synchronize();
            return;
        };
    }
    try ru.norm(sc.h.ptr.?, sc.r.ptr, w.norm, sc.x.ptr.?, total);
    if (o.head == .all) {
        try ru.projection(w.head, sc.x.ptr.?, sc.logits.ptr.?, total);
    } else {
        try ru.projection(w.head, off(sc.x, (total - 1) * c.hidden * 2), sc.logits.ptr.?, 1);
    }
    // A recording pass stops here: the commit is a queued copy plus a cache-length bump, neither belongs in a list.
    if (o.rec != null) return;
    base = 0;
    for (segments) |seg| {
        const row: usize = if (o.head == .all) base + seg.rows - 1 else 0;
        try s.queue.copy(seg.cache.logits.ptr, off(sc.logits, row * c.vocab * 2), c.vocab * 2, null);
        var rec: [cfg.max_layers]?*const anyopaque = @splat(null);
        var cv: [cfg.max_layers]?*const anyopaque = @splat(null);
        for (&rec, &cv, 0..) |*r, *v, i| if (seg.cache.c.kind(i) == .linear) {
            r.* = sc.snapDelta(i, base + seg.rows - 1);
            v.* = sc.snapConv(i, base + seg.rows - 1);
        };
        try seg.cache.commit(s, base, seg.rows, record, rec, cv);
        base += seg.rows;
    }
    try s.queue.synchronize();
}

fn delta(ru: Run, d: xw.Gdn, layer: usize, segments: []const Segment, record: bool, total: usize) rt.Error!void {
    const sc = ru.sc;
    const T: u32 = @intCast(total);
    try ru.projection(d.qkv, sc.x.ptr.?, sc.qkv.ptr.?, total);
    try ru.projection(d.z, sc.x.ptr.?, sc.z.ptr.?, total);
    try ru.projection(d.a, sc.x.ptr.?, sc.a.ptr.?, total);
    try ru.projection(d.b, sc.x.ptr.?, sc.b.ptr.?, total);
    const w = ru.w;
    var base: usize = 0;
    for (segments) |seg| {
        const cache = &seg.cache.blocks[layer].linear;
        // The conv snapshot is written at this segment's absolute scratch row.
        const snap_conv: *anyopaque = sc.snapConv(layer, base);
        try ru.go(.gdn_pre, geo.gdnPre(@intCast(seg.rows), 2 * c.k_heads + c.v_heads), .{
            off(sc.qkv, base * sc.cfg.gdnQkvDim() * 2), cache.conv.ptr, w.ptr(d.conv), @as(*anyopaque, @ptrCast(sc.windows().ptr)),
            off(sc.a, base * c.v_heads * 2), off(sc.b, base * c.v_heads * 2), w.ptr(d.a_log), w.ptr(d.dt_bias),
            off(sc.q, base * c.gdnValueDim() * 2), off(sc.k, base * c.k_heads * c.dk * 2), off(sc.v, base * c.gdnValueDim() * 2),
            off(sc.g, base * c.v_heads * 4), off(sc.beta, base * c.v_heads * 2), snap_conv,
        });
        // The recurrence is read into registers before it is written, one state row per work-item, so state_in == state_out is safe.
        // With `record`, the kernel saves every row to `layer`'s snapshot; otherwise only the final state.
        const snap_rec: *anyopaque = if (record) sc.snapDelta(layer, base) else cache.recurrence.ptr.?;
        try ru.go(.gdn_chain, geo.gdnChain(c.dk, c.v_heads), .{
            off(sc.q, base * c.gdnValueDim() * 2), off(sc.k, base * c.k_heads * c.dk * 2), off(sc.v, base * c.gdnValueDim() * 2),
            off(sc.g, base * c.v_heads * 4), off(sc.beta, base * c.v_heads * 2), cache.recurrence.ptr,
            sc.dims([2]i32{ @intCast(seg.rows), @intFromBool(record) }), off(sc.y, base * c.gdnValueDim() * 2), snap_rec,
        });
        // Without `record` the recurrence went straight into `cache.recurrence`; the conv tail is the segment's last row.
        if (!record) {
            if (ru.rec) |r| {
                try r.copy(cache.conv.ptr, sc.snapConv(layer, base + seg.rows - 1), xs.convBytes(sc.cfg));
            } else {
                try ru.s.queue.copy(cache.conv.ptr, sc.snapConv(layer, base + seg.rows - 1), xs.convBytes(sc.cfg), null);
            }
        }
        base += seg.rows;
    }
    try ru.go(.gdn_post, geo.gdnPost(T, c.v_heads), .{ sc.y.ptr, sc.z.ptr, w.ptr(d.norm), c.eps, sc.mix.ptr });
    try ru.projection(d.out, sc.mix.ptr.?, sc.r.ptr.?, total);
}

fn attention(ru: Run, a: xw.Attention, layer: usize, segments: []const Segment, total: usize) rt.Error!void {
    const sc = ru.sc;
    const w = ru.w;
    const T: u32 = @intCast(total);
    try ru.projection(a.q, sc.x.ptr.?, sc.qraw.ptr.?, total);
    try ru.projection(a.k, sc.x.ptr.?, sc.k.ptr.?, total);
    try ru.projection(a.v, sc.x.ptr.?, sc.v.ptr.?, total);
    // qraw holds q_proj's output: rows x qDim() bf16 (2 * heads * head_dim = 12288 at 27B, 4096 at 2B).
    // dims[2] is its element stride; the 2B value coincidentally equals 2*hidden, the 27B one does not.
    const qn = sc.dims([4]u32{ T, c.heads, @intCast(c.qDim()), 2 * c.head_dim });
    try ru.go(.head_norm, geo.layout(.head_norm, @intCast(total * c.heads), c.hidden, c.head_dim, c.heads, c.kv_heads), .{ sc.qraw.ptr, w.ptr(a.q_norm), sc.q.ptr, qn });
    const kn = sc.dims([4]u32{ T, c.kv_heads, c.kv_heads * c.head_dim, c.head_dim });
    try ru.go(.head_norm, geo.layout(.head_norm, @intCast(total * c.kv_heads), c.hidden, c.head_dim, c.heads, c.kv_heads), .{ sc.k.ptr, w.ptr(a.k_norm), sc.knorm.ptr, kn });
    // RoPE and the key/value cache rows are per segment: each writes at its own cache length, reading its own rows.
    var base: usize = 0;
    for (segments) |seg| {
        const cache = &seg.cache.blocks[layer].attention;
        const rows: u32 = @intCast(seg.rows);
        // dims[1] is the row stride of the shared `queries` buffer, dims[2] this segment's base.
        try ru.go(.queries, geo.layout(.queries, rows, c.hidden, c.head_dim, c.heads, c.kv_heads), .{
            off(sc.q, base * c.gdnValueDim() * 2), sc.queries.ptr, sc.dims([4]u32{ @intCast(seg.cache.len), T, @intCast(base), 0 }),
        });
        try ru.go(.keys, geo.layout(.keys, rows, c.hidden, c.head_dim, c.heads, c.kv_heads), .{
            off(sc.knorm, base * c.kvDim() * 2), off(sc.v, base * c.gdnValueDim() * 2), cache.keys.ptr, cache.values.ptr, sc.dims([4]u32{ @intCast(seg.cache.len), @intCast(seg.cache.capacity), @intCast(base), 0 }),
        });
        const nch: usize = (seg.cache.len + seg.rows + 127) / 128;
        const dims = sc.dims([8]i32{ @intCast(seg.cache.len), @intCast(seg.rows), @intCast(seg.cache.capacity), @intCast(nch), 0, @intCast(total), @intCast(base), 0 });
        try ru.go(.attn_partial, geo.attnPartial(@intCast(nch), rows, c.heads), .{
            sc.queries.ptr, cache.keys.ptr, cache.values.ptr, @as(f32, 1.0 / 16.0), dims, sc.pm.ptr, sc.pl.ptr, sc.po.ptr,
        });
        try ru.go(.attn_merge, geo.attnMerge(rows, c.heads), .{ sc.pm.ptr, sc.pl.ptr, sc.po.ptr, dims, off(sc.y, base * c.gdnValueDim() * 2) });
        base += seg.rows;
    }
    try ru.go(.attention_gate, geo.layout(.attention_gate, T, c.hidden, c.head_dim, c.heads, c.kv_heads), .{ sc.y.ptr, sc.qraw.ptr, sc.mix.ptr });
    try ru.projection(a.o, sc.mix.ptr.?, sc.r.ptr.?, total);
}
