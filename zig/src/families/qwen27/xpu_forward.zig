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
/// TEMP: launch tracing, enabled with TF_XPU_QWEN27_TRACE=1.
var trace_on: bool = false;
pub fn setTrace(v: bool) void {
    trace_on = v;
}
pub const Head = enum { last, all };
pub const Error = rt.Error || error{ QwenRowsExceeded, QwenContextFull, InvalidQwenToken, KernelBinaryMissing };
extern "c" fn getenv(name: [*:0]const u8) ?[*:0]const u8;

/// TEMP: true when `name` is set and not "0".
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
    made: usize = 0,
    dims_used: usize = 0,

    const names = .{ "pool", "h", "x", "r", "z", "q", "k", "v", "y", "mix", "queries", "qkv", "gate", "up", "act", "qraw", "knorm", "a", "b", "beta", "g", "pm", "pl", "po", "logits", "conv_snap" };
    pub const dims_slots = 1024;
    pub const slot_bytes = 32;

    pub fn init(s: *rt.session.Session, config: cfgmod.Config, rows: usize, capacity: usize, logit_rows: usize) rt.Error!Scratch {
        var out: Scratch = undefined;
        out.rows = rows;
        out.capacity = capacity;
        out.logit_rows = logit_rows;
        out.cfg = config;
        out.made = 0;
        out.dims_used = 0;
        errdefer out.deinit();
        const chunks = (capacity + 127) / 128;
        const hid = rows * config.hidden * 2;
        const sizes = [_]usize{
            rows * 4 + rows * config.conv_kernel * 4 + dims_slots * slot_bytes, hid, hid, hid, hid, hid, hid, hid, hid, hid, hid,
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
        return @as([*]i32, @ptrCast(@alignCast(@as([*]u8, @ptrCast(self.pool.ptr.?)) + self.rows * 4)))[0 .. self.rows * c.conv_kernel];
    }
    fn dimsBase(self: *Scratch) [*]u8 {
        return @as([*]u8, @ptrCast(self.pool.ptr.?)) + self.rows * 4 + self.rows * c.conv_kernel * 4;
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

    fn go(self: Run, id: KernelId, g: geo.Geo, args: anytype) rt.Error!void {
        const global: Dim = .{ .x = g.gx, .y = g.gy, .z = g.gz };
        const local: Dim = .{ .x = g.lx, .y = g.ly, .z = g.lz };
        if (trace_on) std.debug.print("TRACE {s} g=({d},{d},{d}) l=({d},{d},{d})\n", .{ @tagName(id), g.gx, g.gy, g.gz, g.lx, g.ly, g.lz });
        try rt.launch.run(&self.s.queue, self.k.get(id), .{ .global = global, .local = local }, args, null);
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
    for (0..total) |r| for (0..c.conv_kernel) |tap| {
        sc.windows()[r * c.conv_kernel + tap] = @intCast(r + tap);
    };
    sc.dims_used = 0;
    const ru = Run{ .s = s, .k = &m.kernels, .w = m.weights, .sc = sc };
    const w = m.weights;
    const T: u32 = @intCast(total);
    try ru.go(.embed, geo.layout(.embed, T, c.hidden, c.head_dim, c.heads, c.kv_heads), .{ sc.pool.ptr, w.ptr(w.embed.weight), w.ptr(w.embed.scales), w.ptr(w.embed.biases), sc.h.ptr });
    for (w.layers, 0..) |layer, i| {
        try ru.norm(sc.h.ptr.?, if (i == 0) null else sc.r.ptr, layer.input_norm, sc.x.ptr.?, total);
        switch (layer.mixer) {
            .linear => |d| try delta(ru, d, &cache.blocks[i].linear, total),
            .attention => |a| try attention(ru, a, &cache.blocks[i].attention, cache.len, cache.capacity, total),
        }
        try ru.norm(sc.h.ptr.?, sc.r.ptr, layer.post_norm, sc.x.ptr.?, total);
        try ru.projection(layer.mlp.gate, sc.x.ptr.?, sc.gate.ptr.?, total);
        try ru.projection(layer.mlp.up, sc.x.ptr.?, sc.up.ptr.?, total);
        try ru.go(.mlp_act, geo.mlpAct(T, c.intermediate), .{ sc.gate.ptr, sc.up.ptr, sc.act.ptr });
        try ru.projection(layer.mlp.down, sc.act.ptr.?, sc.r.ptr.?, total);
        if (o.stop_after) |n| if (i == n) {
            try s.queue.synchronize();
            return;
        };
    }
    try ru.norm(sc.h.ptr.?, sc.r.ptr, w.norm, sc.x.ptr.?, total);
    if (o.head == .all) {
        try ru.projection(w.head, sc.x.ptr.?, sc.logits.ptr.?, total);
    } else {
        try ru.projection(w.head, off(sc.x, (total - 1) * c.hidden * 2), sc.logits.ptr.?, 1);
    }
    const row: usize = if (o.head == .all) total - 1 else 0;
    try s.queue.copy(cache.logits.ptr, off(sc.logits, row * c.vocab * 2), c.vocab * 2, null);
    try s.queue.synchronize();
    cache.len += total;
}

fn delta(ru: Run, d: xw.Gdn, cache: *xs.Gdn, total: usize) rt.Error!void {
    const sc = ru.sc;
    const T: u32 = @intCast(total);
    try ru.projection(d.qkv, sc.x.ptr.?, sc.qkv.ptr.?, total);
    try ru.projection(d.z, sc.x.ptr.?, sc.z.ptr.?, total);
    try ru.projection(d.a, sc.x.ptr.?, sc.a.ptr.?, total);
    try ru.projection(d.b, sc.x.ptr.?, sc.b.ptr.?, total);
    const w = ru.w;
    try ru.go(.gdn_pre, geo.gdnPre(T, 2 * c.k_heads + c.v_heads), .{
        sc.qkv.ptr, cache.conv.ptr, w.ptr(d.conv), @as(*anyopaque, @ptrCast(sc.windows().ptr)), sc.a.ptr, sc.b.ptr, w.ptr(d.a_log), w.ptr(d.dt_bias),
        sc.q.ptr, sc.k.ptr, sc.v.ptr, sc.g.ptr, sc.beta.ptr, sc.conv_snap.ptr,
    });
    // The recurrence is read into registers before it is written, one state row per work-item, so state_in == state_out is safe.
    try ru.go(.gdn_chain, geo.gdnChain(c.dk, c.v_heads), .{
        sc.q.ptr, sc.k.ptr, sc.v.ptr, sc.g.ptr, sc.beta.ptr, cache.recurrence.ptr, sc.dims([2]i32{ @intCast(total), 0 }), sc.y.ptr, cache.recurrence.ptr,
    });
    try ru.s.queue.copy(cache.conv.ptr, off(sc.conv_snap, (total - 1) * xs.convBytes(sc.cfg)), xs.convBytes(sc.cfg), null);
    try ru.go(.gdn_post, geo.gdnPost(T, c.v_heads), .{ sc.y.ptr, sc.z.ptr, w.ptr(d.norm), c.eps, sc.mix.ptr });
    try ru.projection(d.out, sc.mix.ptr.?, sc.r.ptr.?, total);
}

fn attention(ru: Run, a: xw.Attention, cache: *xs.Attention, len: usize, capacity: usize, total: usize) rt.Error!void {
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
    try ru.go(.queries, geo.layout(.queries, T, c.hidden, c.head_dim, c.heads, c.kv_heads), .{ sc.q.ptr, sc.queries.ptr, sc.dims([4]u32{ @intCast(len), T, 0, 0 }) });
    try ru.go(.keys, geo.layout(.keys, T, c.hidden, c.head_dim, c.heads, c.kv_heads), .{ sc.knorm.ptr, sc.v.ptr, cache.keys.ptr, cache.values.ptr, sc.dims([4]u32{ @intCast(len), @intCast(capacity), 0, 0 }) });
    const nch: usize = (len + total + 127) / 128;
    const dims = sc.dims([8]i32{ @intCast(len), @intCast(total), @intCast(capacity), @intCast(nch), 0, @intCast(total), 0, 0 });
    try ru.go(.attn_partial, geo.attnPartial(@intCast(nch), T, c.heads), .{
        sc.queries.ptr, cache.keys.ptr, cache.values.ptr, @as(f32, 1.0 / 16.0), dims, sc.pm.ptr, sc.pl.ptr, sc.po.ptr,
    });
    try ru.go(.attn_merge, geo.attnMerge(T, c.heads), .{ sc.pm.ptr, sc.pl.ptr, sc.po.ptr, dims, sc.y.ptr });
    try ru.go(.attention_gate, geo.layout(.attention_gate, T, c.hidden, c.head_dim, c.heads, c.kv_heads), .{ sc.y.ptr, sc.qraw.ptr, sc.mix.ptr });
    try ru.projection(a.o, sc.mix.ptr.?, sc.r.ptr.?, total);
}
