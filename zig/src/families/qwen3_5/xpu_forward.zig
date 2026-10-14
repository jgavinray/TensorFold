//! Qwen3.5-2B forward pass on an Intel B70 (Level Zero + OpenCL kernels), mirroring forward.zig's op order and rounding
//! points (bf16 activations between every op, fp32 recurrence). Kernels are loaded by key from ocloc-compiled modules
//! (build option `xpu_qwen35_kernels`); a faster port replaces a reference kernel by swapping the module file, either in
//! zig/kernels/xpu/qwen3_5/ or at run time through the directory named by TF_XPU_QWEN35_KERNELS (<module key>.bin).
//! One segment per call (prefill rows or a decode row appended to the cache); no snapshot recording (record = false).
const std = @import("std");
const rt = @import("../../xpu/runtime.zig");
const sys = @import("../../xpu/sys.zig");
const kbins = @import("xpu_qwen35_kernels");
const xw = @import("xpu_weights.zig");
const xs = @import("xpu_state.zig");
const c = @import("config.zig");

const Buffer = rt.memory.Buffer;
const Kernel = rt.module.Kernel;
const Dim = rt.launch.Dim3;
pub const Head = enum { last, all };
pub const Error = rt.Error || error{ QwenRowsExceeded, QwenContextFull, InvalidQwenToken, KernelBinaryMissing };
extern "c" fn getenv(name: [*:0]const u8) ?[*:0]const u8;

/// Module keys (file names under reference/<key>/) and the kernels each provides.
const Module = enum { layout, norm, norm_nores, qmm, mlp_act, gdn_pre, gdn_chain, gdn_post, attn_partial, attn_merge };
const KernelId = enum {
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

fn builtinPath(m: Module) []const u8 {
    return switch (m) {
        inline else => |t| @field(kbins, @tagName(t)),
    };
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
            const path = if (dir) |d| sys.zfmt(&buf, "{s}/{s}.bin", .{ d, fname }) else sys.zfmt(&buf, "{s}", .{builtinPath(mod)});
            const bin = sys.readFileAlloc(gpa, path orelse return error.KernelBinaryMissing, 64 << 20) orelse {
                std.debug.print("cannot read kernel module {s}\n", .{path orelse "?"});
                return error.KernelBinaryMissing;
            };
            defer gpa.free(bin);
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
};

/// Intermediate buffers (device), plus a small host-visible pool for ids, conv windows and kernel `dims` arrays
/// (every launch gets its own slot, so the whole forward can be queued before the host waits).
pub const Scratch = struct {
    rows: usize,
    capacity: usize,
    logit_rows: usize,
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
    made: usize = 0,
    dims_used: usize = 0,

    const names = .{ "pool", "h", "x", "r", "z", "q", "k", "v", "y", "mix", "queries", "qkv", "gate", "up", "act", "qraw", "knorm", "a", "b", "beta", "g", "pm", "pl", "po", "logits", "conv_snap" };
    pub const dims_slots = 1024;
    pub const slot_bytes = 32;

    pub fn init(s: *rt.session.Session, rows: usize, capacity: usize, logit_rows: usize) rt.Error!Scratch {
        var out: Scratch = undefined;
        out.rows = rows;
        out.capacity = capacity;
        out.logit_rows = logit_rows;
        out.made = 0;
        out.dims_used = 0;
        errdefer out.deinit();
        const chunks = (capacity + 127) / 128;
        const hid = rows * c.hidden * 2;
        const sizes = [_]usize{
            rows * 4 + rows * c.conv_taps * 4 + dims_slots * slot_bytes, hid, hid, hid, hid, hid, hid, hid, hid, hid, hid,
            rows * c.intermediate * 2, rows * c.intermediate * 2, rows * c.intermediate * 2, rows * c.intermediate * 2,
            rows * 2 * c.hidden * 2, rows * c.kv_heads * c.head_dim * 2,
            rows * c.linear_heads * 2, rows * c.linear_heads * 2, rows * c.linear_heads * 2, rows * c.linear_heads * 4,
            rows * c.query_heads * chunks * 4, rows * c.query_heads * chunks * 4, rows * c.query_heads * chunks * c.head_dim * 4,
            logit_rows * c.vocab * 2, rows * xs.conv_bytes,
        };
        inline for (names, 0..) |n, i| {
            const kind: rt.memory.Kind = if (i == 0) .host else .device;
            @field(out, n) = try Buffer.alloc(&s.ctx, kind, sizes[i]);
            out.made = i + 1;
        }
        return out;
    }

    pub fn deinit(self: *Scratch) void {
        inline for (names, 0..) |n, i| if (i < self.made) @field(self, n).free();
        self.made = 0;
    }

    fn ids(self: *Scratch) []u32 {
        return @as([*]u32, @ptrCast(@alignCast(self.pool.ptr.?)))[0..self.rows];
    }
    fn windows(self: *Scratch) []i32 {
        return @as([*]i32, @ptrCast(@alignCast(@as([*]u8, @ptrCast(self.pool.ptr.?)) + self.rows * 4)))[0 .. self.rows * c.conv_taps];
    }
    fn dimsBase(self: *Scratch) [*]u8 {
        return @as([*]u8, @ptrCast(self.pool.ptr.?)) + self.rows * 4 + self.rows * c.conv_taps * 4;
    }
    /// Copies `vals` into a fresh slot of the pool and returns its device-visible address.
    fn dims(self: *Scratch, vals: anytype) *anyopaque {
        const bytes = std.mem.asBytes(&vals);
        std.debug.assert(bytes.len <= slot_bytes and self.dims_used < dims_slots);
        const p = self.dimsBase() + self.dims_used * slot_bytes;
        @memcpy(p[0..bytes.len], bytes);
        self.dims_used += 1;
        return @ptrCast(p);
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

    fn go(self: Run, id: KernelId, global: Dim, local: Dim, args: anytype) rt.Error!void {
        try rt.launch.run(&self.s.queue, self.k.get(id), .{ .global = global, .local = local }, args, null);
    }
    fn wp(self: Run, t: xw.Tensor) *anyopaque {
        return self.w.ptr(t);
    }
    fn projection(self: Run, l: xw.Linear, x: *anyopaque, y: *anyopaque, rows: usize) rt.Error!void {
        const gx: u32 = @intCast((l.outputs + 63) / 64 * 64);
        try self.go(.qmm, .{ .x = gx, .y = @intCast(rows) }, .{ .x = 64 }, .{
            x, @as(i32, @intCast(rows)), self.wp(l.weight), self.wp(l.scales), self.wp(l.biases), y, @as(i32, @intCast(l.outputs)), @as(i32, @intCast(l.inputs)),
        });
    }
    fn norm(self: Run, h: *anyopaque, residual: ?*anyopaque, weight: xw.Tensor, x: *anyopaque, rows: usize) rt.Error!void {
        const g: Dim = .{ .x = 128, .y = @intCast(rows) };
        if (residual) |r| {
            try self.go(.norm, g, .{ .x = 128 }, .{ h, r, self.wp(weight), c.eps, h, x });
        } else {
            try self.go(.norm_nores, g, .{ .x = 128 }, .{ h, self.wp(weight), c.eps, x });
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
};

/// Forward `tokens` as one segment appended to `cache`. Blocks until the card has finished. On return `cache.len` has
/// advanced and `cache.logits` holds the last row's bf16 logits (unless `stop_after` is set); with head = .all
/// `scratch.logits` holds every row.
pub fn run(s: *rt.session.Session, m: *Model, sc: *Scratch, cache: *xs.Cache, tokens: []const u32, o: Options) Error!void {
    const total = tokens.len;
    if (total == 0 or total > sc.rows or (o.head == .all and total > sc.logit_rows)) return error.QwenRowsExceeded;
    if (cache.len + total > cache.capacity or cache.capacity > sc.capacity) return error.QwenContextFull;
    for (tokens) |id| if (id >= c.vocab) return error.InvalidQwenToken;
    @memcpy(sc.ids()[0..total], tokens);
    for (0..total) |r| for (0..c.conv_taps) |tap| {
        sc.windows()[r * c.conv_taps + tap] = @intCast(r + tap);
    };
    sc.dims_used = 0;
    const ru = Run{ .s = s, .k = &m.kernels, .w = m.weights, .sc = sc };
    const w = m.weights;
    const T: u32 = @intCast(total);
    try ru.go(.embed, .{ .x = 1024, .y = T }, .{ .x = 64 }, .{ sc.pool.ptr, w.ptr(w.embedding.weight), w.ptr(w.embedding.scales), w.ptr(w.embedding.biases), sc.h.ptr });
    for (w.blocks, 0..) |block, i| {
        try ru.norm(sc.h.ptr.?, if (i == 0) null else sc.r.ptr, block.input_norm, sc.x.ptr.?, total);
        switch (block.mixer) {
            .delta => |d| try delta(ru, d, &cache.blocks[i].delta, total),
            .attention => |a| try attention(ru, a, &cache.blocks[i].attention, cache.len, cache.capacity, total),
        }
        try ru.norm(sc.h.ptr.?, sc.r.ptr, block.post_norm, sc.x.ptr.?, total);
        try ru.projection(block.gate, sc.x.ptr.?, sc.gate.ptr.?, total);
        try ru.projection(block.up, sc.x.ptr.?, sc.up.ptr.?, total);
        try ru.go(.mlp_act, .{ .x = c.intermediate, .y = T }, .{ .x = 256 }, .{ sc.gate.ptr, sc.up.ptr, sc.act.ptr });
        try ru.projection(block.down, sc.act.ptr.?, sc.r.ptr.?, total);
        if (o.stop_after) |n| if (i == n) {
            try s.queue.synchronize();
            return;
        };
    }
    try ru.norm(sc.h.ptr.?, sc.r.ptr, w.norm, sc.x.ptr.?, total);
    if (o.head == .all) {
        try ru.projection(w.head(), sc.x.ptr.?, sc.logits.ptr.?, total);
    } else {
        try ru.projection(w.head(), off(sc.x, (total - 1) * c.hidden * 2), sc.logits.ptr.?, 1);
    }
    const row: usize = if (o.head == .all) total - 1 else 0;
    try s.queue.copy(cache.logits.ptr, off(sc.logits, row * c.vocab * 2), c.vocab * 2, null);
    try s.queue.synchronize();
    cache.len += total;
}

fn delta(ru: Run, d: xw.Delta, cache: *xs.Delta, total: usize) rt.Error!void {
    const sc = ru.sc;
    const T: u32 = @intCast(total);
    try ru.projection(d.qkv, sc.x.ptr.?, sc.qkv.ptr.?, total);
    try ru.projection(d.z, sc.x.ptr.?, sc.z.ptr.?, total);
    try ru.projection(d.a, sc.x.ptr.?, sc.a.ptr.?, total);
    try ru.projection(d.b, sc.x.ptr.?, sc.b.ptr.?, total);
    const w = ru.w;
    try ru.go(.gdn_pre, .{ .x = 128, .y = 3 * c.linear_heads, .z = T }, .{ .x = 128 }, .{
        sc.qkv.ptr, cache.conv.ptr, w.ptr(d.conv), @as(*anyopaque, @ptrCast(sc.windows().ptr)), sc.a.ptr, sc.b.ptr, w.ptr(d.a_log), w.ptr(d.dt_bias),
        sc.q.ptr, sc.k.ptr, sc.v.ptr, sc.g.ptr, sc.beta.ptr, sc.conv_snap.ptr,
    });
    // The recurrence is read into registers before it is written, one state row per work-item, so state_in == state_out is safe.
    try ru.go(.gdn_chain, .{ .x = c.linear_dim, .y = c.linear_heads }, .{ .x = 64 }, .{
        sc.q.ptr, sc.k.ptr, sc.v.ptr, sc.g.ptr, sc.beta.ptr, cache.recurrence.ptr, sc.dims([2]i32{ @intCast(total), 0 }), sc.y.ptr, cache.recurrence.ptr,
    });
    try ru.s.queue.copy(cache.conv.ptr, off(sc.conv_snap, (total - 1) * xs.conv_bytes), xs.conv_bytes, null);
    try ru.go(.gdn_post, .{ .x = 128, .y = c.linear_heads, .z = T }, .{ .x = 128 }, .{ sc.y.ptr, sc.z.ptr, w.ptr(d.norm), c.eps, sc.mix.ptr });
    try ru.projection(d.out, sc.mix.ptr.?, sc.r.ptr.?, total);
}

fn attention(ru: Run, a: xw.Attention, cache: *xs.Attention, len: usize, capacity: usize, total: usize) rt.Error!void {
    const sc = ru.sc;
    const w = ru.w;
    const T: u32 = @intCast(total);
    try ru.projection(a.q, sc.x.ptr.?, sc.qraw.ptr.?, total);
    try ru.projection(a.k, sc.x.ptr.?, sc.k.ptr.?, total);
    try ru.projection(a.v, sc.x.ptr.?, sc.v.ptr.?, total);
    const qn = sc.dims([4]u32{ T, c.query_heads, 2 * c.hidden, 2 * c.head_dim });
    try ru.go(.head_norm, .{ .x = @intCast(64 * total * c.query_heads) }, .{ .x = 64 }, .{ sc.qraw.ptr, w.ptr(a.q_norm), sc.q.ptr, qn });
    const kn = sc.dims([4]u32{ T, c.kv_heads, c.kv_heads * c.head_dim, c.head_dim });
    try ru.go(.head_norm, .{ .x = @intCast(64 * total * c.kv_heads) }, .{ .x = 64 }, .{ sc.k.ptr, w.ptr(a.k_norm), sc.knorm.ptr, kn });
    try ru.go(.queries, .{ .x = c.head_dim, .y = c.query_heads, .z = T }, .{ .x = 256 }, .{ sc.q.ptr, sc.queries.ptr, sc.dims([4]u32{ @intCast(len), T, 0, 0 }) });
    try ru.go(.keys, .{ .x = c.head_dim, .y = c.kv_heads, .z = T }, .{ .x = 256 }, .{ sc.knorm.ptr, sc.v.ptr, cache.keys.ptr, cache.values.ptr, sc.dims([4]u32{ @intCast(len), @intCast(capacity), 0, 0 }) });
    const nch: usize = (len + total + 127) / 128;
    const dims = sc.dims([8]i32{ @intCast(len), @intCast(total), @intCast(capacity), @intCast(nch), 0, @intCast(total), 0, 0 });
    try ru.go(.attn_partial, .{ .x = 256, .y = @intCast(nch), .z = @intCast(c.kv_heads * total) }, .{ .x = 256 }, .{
        sc.queries.ptr, cache.keys.ptr, cache.values.ptr, @as(f32, 1.0 / 16.0), dims, sc.pm.ptr, sc.pl.ptr, sc.po.ptr,
    });
    try ru.go(.attn_merge, .{ .x = 256, .y = c.query_heads, .z = T }, .{ .x = 256 }, .{ sc.pm.ptr, sc.pl.ptr, sc.po.ptr, dims, sc.y.ptr });
    try ru.go(.attention_gate, .{ .x = c.hidden, .y = T }, .{ .x = 256 }, .{ sc.y.ptr, sc.qraw.ptr, sc.mix.ptr });
    try ru.projection(a.out, sc.mix.ptr.?, sc.r.ptr.?, total);
}
