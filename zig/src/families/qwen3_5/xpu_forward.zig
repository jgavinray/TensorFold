//! Qwen3.5-2B forward pass on an Intel B70 (Level Zero + OpenCL kernels), mirroring forward.zig's op order and rounding
//! points (bf16 activations between every op, fp32 recurrence). Kernels are loaded by key from ocloc-compiled modules
//! embedded in the executable at build time (build option `xpu_qwen35_kernels`, @embedFile of the ocloc outputs), so the
//! binary works from any working directory. Each module has a reference build and, where one was accepted, a port
//! (<key>_port); a kernel's launch geometry comes with its variant (xpu_geometry.zig). Default: ports where they exist,
//! reference otherwise; TF_XPU_QWEN35_VARIANT=reference forces all-reference. TF_XPU_QWEN35_KERNELS names a directory of
//! <module key>.bin files that replaces the embedded modules (development override; the variant still selects the geometry).
//! One segment per call (prefill rows or a decode row appended to the cache); no snapshot recording (record = false).
const std = @import("std");
const rt = @import("../../xpu/runtime.zig");
const sys = @import("../../xpu/sys.zig");
const kbins = @import("xpu_qwen35_kernels");
const xw = @import("xpu_weights.zig");
const xs = @import("xpu_state.zig");
const c = @import("config.zig");
const geo = @import("xpu_geometry.zig");

const Buffer = rt.memory.Buffer;
const Kernel = rt.module.Kernel;
const Dim = rt.launch.Dim3;
pub const Head = enum { last, all };
pub const Error = rt.Error || error{ QwenRowsExceeded, QwenContextFull, InvalidQwenToken, KernelBinaryMissing };
extern "c" fn getenv(name: [*:0]const u8) ?[*:0]const u8;

/// Module keys (file names under reference/<key>/) and the kernels each provides.
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

pub const Variant = geo.Variant;

/// The module's port exists in this build (zig/build/xpu.zig `port_keys`).
fn hasPort(comptime m: Module) bool {
    return @hasDecl(kbins, @tagName(m) ++ "_port");
}

/// The compiled module's bytes, embedded in the executable by the build graph (`zig/build/xpu.zig`).
fn builtinBytes(m: Module, v: Variant) []const u8 {
    return switch (m) {
        inline else => |t| if (comptime hasPort(t)) (if (v == .port) &@field(kbins, @tagName(t) ++ "_port") else &@field(kbins, @tagName(t))) else &@field(kbins, @tagName(t)),
    };
}

/// Variant a module runs as: the port where one is built, unless TF_XPU_QWEN35_VARIANT=reference.
fn chooseVariant(m: Module, force_reference: bool) Variant {
    if (force_reference) return .reference;
    return switch (m) {
        inline else => |t| if (comptime hasPort(t)) .port else .reference,
    };
}

pub const Kernels = struct {
    modules: [module_count]rt.module.Module,
    variant: [module_count]Variant = @splat(.reference),
    loaded: usize = 0,
    list: [kernel_count]Kernel,
    made: usize = 0,

    pub fn load(self: *Kernels, s: *rt.session.Session, gpa: std.mem.Allocator) !void {
        self.loaded = 0;
        self.made = 0;
        errdefer self.deinit();
        const force_ref = if (getenv("TF_XPU_QWEN35_VARIANT")) |v| std.mem.eql(u8, std.mem.span(v), "reference") else false;
        const dir: ?[:0]const u8 = if (getenv("TF_XPU_QWEN35_KERNELS")) |p| std.mem.span(p) else null;
        inline for (std.enums.values(Module), 0..) |mod, i| {
            const fname = @tagName(mod);
            self.variant[i] = chooseVariant(mod, force_ref);
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
            } else builtinBytes(mod, self.variant[i]);
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
    pub fn variantOf(self: *const Kernels, id: KernelId) Variant {
        return self.variant[@intFromEnum(id.module())];
    }
    /// One line per module: which variant is running.
    pub fn describe(self: *const Kernels) void {
        std.debug.print("xpu qwen3.5 kernel variants:", .{});
        inline for (std.enums.values(Module), 0..) |m, i| std.debug.print(" {s}={s}", .{ @tagName(m), @tagName(self.variant[i]) });
        std.debug.print("\n", .{});
    }
};

/// Intermediate buffers (device), plus a small host-visible pool for ids, conv windows and kernel `dims` arrays
/// (every launch gets its own slot, so the whole forward can be queued before the host waits).
/// Per-kernel device time of one forward call. Every launch gets its own timestamp event; `collect` reads them after the
/// call has finished (the queue must be synchronized) and resets them. Copies (conv-state snapshot, logits) are not timed.
pub const Profile = struct {
    pool: rt.event.Pool,
    events: []rt.event.Event,
    ids: []KernelId,
    used: usize = 0,
    ns: [kernel_count]f64 = @splat(0),
    count: [kernel_count]usize = @splat(0),

    pub fn init(s: *rt.session.Session, gpa: std.mem.Allocator, capacity: u32) !Profile {
        var pool = try rt.event.Pool.init(&s.ctx, capacity);
        errdefer pool.deinit();
        const events = try gpa.alloc(rt.event.Event, capacity);
        errdefer gpa.free(events);
        const ids = try gpa.alloc(KernelId, capacity);
        for (events, 0..) |*e, i| e.* = try pool.event(@intCast(i));
        return .{ .pool = pool, .events = events, .ids = ids };
    }
    pub fn deinit(self: *Profile, gpa: std.mem.Allocator) void {
        for (self.events) |*e| e.deinit();
        gpa.free(self.events);
        gpa.free(self.ids);
        self.pool.deinit();
    }
    fn next(self: *Profile, id: KernelId) ?*const rt.event.Event {
        if (self.used >= self.events.len) return null;
        self.ids[self.used] = id;
        self.used += 1;
        return &self.events[self.used - 1];
    }
    /// Adds the launches since the last call to `ns`/`count` and rearms their events.
    pub fn collect(self: *Profile) rt.Error!void {
        for (self.events[0..self.used], self.ids[0..self.used]) |*e, id| {
            self.ns[@intFromEnum(id)] += try e.kernelNs();
            self.count[@intFromEnum(id)] += 1;
            try e.reset();
        }
        self.used = 0;
    }
    pub fn clear(self: *Profile) void {
        self.ns = @splat(0);
        self.count = @splat(0);
    }
};
pub const kernel_ids = std.enums.values(KernelId);

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
    /// Optional per-kernel device timing (Level Zero event timestamps); see `Profile`.
    prof: ?Profile = null,
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
        out.prof = null;
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

    fn go(self: Run, id: KernelId, g: geo.Geo, args: anytype) rt.Error!void {
        const global: Dim = .{ .x = g.gx, .y = g.gy, .z = g.gz };
        const local: Dim = .{ .x = g.lx, .y = g.ly, .z = g.lz };
        if (self.sc.prof) |*p| {
            const ev = p.next(id) orelse return error.Invalid;
            try rt.launch.run(&self.s.queue, self.k.get(id), .{ .global = global, .local = local }, args, ev);
            return;
        }
        try rt.launch.run(&self.s.queue, self.k.get(id), .{ .global = global, .local = local }, args, null);
    }
    fn v(self: Run, id: KernelId) Variant {
        return self.k.variantOf(id);
    }
    fn wp(self: Run, t: xw.Tensor) *anyopaque {
        return self.w.ptr(t);
    }
    fn projection(self: Run, l: xw.Linear, x: *anyopaque, y: *anyopaque, rows: usize) rt.Error!void {
        try self.go(.qmm, geo.qmm(self.v(.qmm), @intCast(rows), @intCast(l.outputs)), .{
            x, @as(i32, @intCast(rows)), self.wp(l.weight), self.wp(l.scales), self.wp(l.biases), y, @as(i32, @intCast(l.outputs)), @as(i32, @intCast(l.inputs)),
        });
    }
    fn norm(self: Run, h: *anyopaque, residual: ?*anyopaque, weight: xw.Tensor, x: *anyopaque, rows: usize) rt.Error!void {
        if (residual) |r| {
            try self.go(.norm, geo.norm(self.v(.norm), @intCast(rows)), .{ h, r, self.wp(weight), c.eps, h, x });
        } else {
            try self.go(.norm_nores, geo.normNores(self.v(.norm_nores), @intCast(rows)), .{ h, self.wp(weight), c.eps, x });
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
    try ru.go(.embed, geo.layout(ru.v(.embed), .embed, T), .{ sc.pool.ptr, w.ptr(w.embedding.weight), w.ptr(w.embedding.scales), w.ptr(w.embedding.biases), sc.h.ptr });
    for (w.blocks, 0..) |block, i| {
        try ru.norm(sc.h.ptr.?, if (i == 0) null else sc.r.ptr, block.input_norm, sc.x.ptr.?, total);
        switch (block.mixer) {
            .delta => |d| try delta(ru, d, &cache.blocks[i].delta, total),
            .attention => |a| try attention(ru, a, &cache.blocks[i].attention, cache.len, cache.capacity, total),
        }
        try ru.norm(sc.h.ptr.?, sc.r.ptr, block.post_norm, sc.x.ptr.?, total);
        try ru.projection(block.gate, sc.x.ptr.?, sc.gate.ptr.?, total);
        try ru.projection(block.up, sc.x.ptr.?, sc.up.ptr.?, total);
        try ru.go(.mlp_act, geo.mlpAct(ru.v(.mlp_act), T, c.intermediate), .{ sc.gate.ptr, sc.up.ptr, sc.act.ptr });
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
    try ru.go(.gdn_pre, geo.gdnPre(ru.v(.gdn_pre), T, 3 * c.linear_heads), .{
        sc.qkv.ptr, cache.conv.ptr, w.ptr(d.conv), @as(*anyopaque, @ptrCast(sc.windows().ptr)), sc.a.ptr, sc.b.ptr, w.ptr(d.a_log), w.ptr(d.dt_bias),
        sc.q.ptr, sc.k.ptr, sc.v.ptr, sc.g.ptr, sc.beta.ptr, sc.conv_snap.ptr,
    });
    // The recurrence is read into registers before it is written, one state row per work-item, so state_in == state_out is safe.
    try ru.go(.gdn_chain, geo.gdnChain(ru.v(.gdn_chain), c.linear_dim, c.linear_heads), .{
        sc.q.ptr, sc.k.ptr, sc.v.ptr, sc.g.ptr, sc.beta.ptr, cache.recurrence.ptr, sc.dims([2]i32{ @intCast(total), 0 }), sc.y.ptr, cache.recurrence.ptr,
    });
    try ru.s.queue.copy(cache.conv.ptr, off(sc.conv_snap, (total - 1) * xs.conv_bytes), xs.conv_bytes, null);
    try ru.go(.gdn_post, geo.gdnPost(ru.v(.gdn_post), T, c.linear_heads), .{ sc.y.ptr, sc.z.ptr, w.ptr(d.norm), c.eps, sc.mix.ptr });
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
    try ru.go(.head_norm, geo.layout(ru.v(.head_norm), .head_norm, @intCast(total * c.query_heads)), .{ sc.qraw.ptr, w.ptr(a.q_norm), sc.q.ptr, qn });
    const kn = sc.dims([4]u32{ T, c.kv_heads, c.kv_heads * c.head_dim, c.head_dim });
    try ru.go(.head_norm, geo.layout(ru.v(.head_norm), .head_norm, @intCast(total * c.kv_heads)), .{ sc.k.ptr, w.ptr(a.k_norm), sc.knorm.ptr, kn });
    try ru.go(.queries, geo.layout(ru.v(.queries), .queries, T), .{ sc.q.ptr, sc.queries.ptr, sc.dims([4]u32{ @intCast(len), T, 0, 0 }) });
    try ru.go(.keys, geo.layout(ru.v(.keys), .keys, T), .{ sc.knorm.ptr, sc.v.ptr, cache.keys.ptr, cache.values.ptr, sc.dims([4]u32{ @intCast(len), @intCast(capacity), 0, 0 }) });
    const nch: usize = (len + total + 127) / 128;
    const dims = sc.dims([8]i32{ @intCast(len), @intCast(total), @intCast(capacity), @intCast(nch), 0, @intCast(total), 0, 0 });
    try ru.go(.attn_partial, geo.attnPartial(ru.v(.attn_partial), @intCast(nch), T, c.kv_heads), .{
        sc.queries.ptr, cache.keys.ptr, cache.values.ptr, @as(f32, 1.0 / 16.0), dims, sc.pm.ptr, sc.pl.ptr, sc.po.ptr,
    });
    try ru.go(.attn_merge, geo.attnMerge(ru.v(.attn_merge), T, c.query_heads), .{ sc.pm.ptr, sc.pl.ptr, sc.po.ptr, dims, sc.y.ptr });
    try ru.go(.attention_gate, geo.layout(ru.v(.attention_gate), .attention_gate, T), .{ sc.y.ptr, sc.qraw.ptr, sc.mix.ptr });
    try ru.projection(a.out, sc.mix.ptr.?, sc.r.ptr.?, total);
}
