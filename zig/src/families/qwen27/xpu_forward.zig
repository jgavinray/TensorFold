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
pub const rp = @import("xpu_replay.zig");
const kbins = @import("xpu_qwen27_kernels");
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
pub const Error = rt.Error || std.mem.Allocator.Error || error{ QwenRowsExceeded, QwenContextFull, InvalidQwenToken, KernelBinaryMissing, ReplayNeedsOneRow, ReplayValueOutOfOrder, ReplayCannotProfile };
extern "c" fn getenv(name: [*:0]const u8) ?[*:0]const u8;

/// True when `name` is set and not "0" (used by `Kernels.load` for the TF_XPU_QWEN35_KERNELS override).
pub fn getenvBool(name: [*:0]const u8) bool {
    const v = getenv(name) orelse return false;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// Module keys (file names under shared/<key>/) and the kernels each provides.
const Module = enum { layout, layout27, norm, norm_nores, norm_qmm, qmm, qmm_pair, mlp_act, mlp_act_down, gdn_pre, gdn_chain, gdn_post, attn_partial, attn_merge, attn_merge27 };
pub const KernelId = enum {
    embed, head_norm, queries, keys, attention_gate, norm, norm_nores, norm_qmm, qmm, qmm_pair, mlp_act, mlp_act_down, gdn_pre, gdn_chain, gdn_post, attn_partial, attn_merge,

    fn module(self: KernelId) Module {
        return switch (self) {
            .embed, .head_norm, .keys, .attention_gate => .layout,
            .queries => .layout27,
            .attn_merge => .attn_merge27,
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

/// Per-kernel device time of one forward call, from Level Zero event timestamps (the same design as the
/// Qwen3.5-2B forward's `Profile`, plus a per-shape table for the qmm-family launches so the achieved
/// bandwidth of the 27B's largest projections can be read off directly).
/// Every launch gets its own timestamp event; `collect` reads them after the call has finished (the queue must
/// be synchronized) and rearms them. Copies (conv-state snapshot, logits) are not timed.
pub const Profile = struct {
    pool: rt.event.Pool,
    events: []rt.event.Event,
    ids: []KernelId,
    shapes: []u64,
    used: usize = 0,
    ns: [kernel_count]f64 = @splat(0),
    count: [kernel_count]usize = @splat(0),
    shape_ns: std.AutoHashMapUnmanaged(u64, f64) = .empty,
    shape_count: std.AutoHashMapUnmanaged(u64, usize) = .empty,

    pub fn init(s: *rt.session.Session, gpa: std.mem.Allocator, cap: u32) !Profile {
        var pool = try rt.event.Pool.init(&s.ctx, cap);
        errdefer pool.deinit();
        const events = try gpa.alloc(rt.event.Event, cap);
        errdefer gpa.free(events);
        const ids = try gpa.alloc(KernelId, cap);
        errdefer gpa.free(ids);
        const shapes = try gpa.alloc(u64, cap);
        errdefer gpa.free(shapes);
        for (events, 0..) |*e, i| e.* = try pool.event(@intCast(i));
        return .{ .pool = pool, .events = events, .ids = ids, .shapes = shapes };
    }
    pub fn deinit(self: *Profile, gpa: std.mem.Allocator) void {
        for (self.events) |*e| e.deinit();
        gpa.free(self.events);
        gpa.free(self.ids);
        gpa.free(self.shapes);
        self.shape_ns.deinit(gpa);
        self.shape_count.deinit(gpa);
        self.pool.deinit();
    }
    /// Shape key of a projection launch: `(outputs << 32) | inputs`, so both fit for every Qwen shape.
    pub fn shapeKey(outputs: usize, inputs: usize) u64 {
        return (@as(u64, outputs) << 32) | @as(u64, inputs);
    }
    pub fn shapeOutputs(key: u64) usize {
        return @intCast(key >> 32);
    }
    pub fn shapeInputs(key: u64) usize {
        return @intCast(key & 0xffffffff);
    }
    fn next(self: *Profile, id: KernelId, shape: u64) ?*const rt.event.Event {
        if (self.used >= self.events.len) return null;
        self.ids[self.used] = id;
        self.shapes[self.used] = shape;
        self.used += 1;
        return &self.events[self.used - 1];
    }
    /// Adds the launches since the last call to `ns`/`count`/`shape_*` and rearms their events.
    pub fn collect(self: *Profile, gpa: std.mem.Allocator) !void {
        for (self.events[0..self.used], self.ids[0..self.used], self.shapes[0..self.used]) |*e, id, shape| {
            const t = try e.kernelNs();
            self.ns[@intFromEnum(id)] += t;
            self.count[@intFromEnum(id)] += 1;
            if (shape != 0) {
                const gop = try self.shape_ns.getOrPut(gpa, shape);
                if (!gop.found_existing) gop.value_ptr.* = 0;
                gop.value_ptr.* += t;
                const goc = try self.shape_count.getOrPut(gpa, shape);
                if (!goc.found_existing) goc.value_ptr.* = 0;
                goc.value_ptr.* += 1;
            }
            try e.reset();
        }
        self.used = 0;
    }
};

pub const kernel_ids = std.enums.values(KernelId);

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
    /// Optional per-kernel device timing (Level Zero event timestamps); see `Profile`.
    prof: ?Profile = null,
    /// While a decode step is being recorded, every `dims` slot index lands here so the replay can rewrite the
    /// per-token scalars inside the captured arrays (see `xpu_replay.zig`). Null outside recording.
    record_slots: ?*std.ArrayList(usize) = null,
    /// Allocator for `record_slots` appends; the caller's, since the scratch does not own one otherwise.
    gpa: std.mem.Allocator = undefined,
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
        out.prof = null;
        out.record_slots = null;
        out.gpa = undefined;
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
    /// Copies `vals` into a fresh slot of the pool and returns its device-visible address. With a recorder active
    /// the slot's index is appended to `record_slots` so the replay can rewrite the per-token scalars inside it.
    fn dims(self: *Scratch, vals: anytype) *anyopaque {
        const bytes = std.mem.asBytes(&vals);
        std.debug.assert(bytes.len <= slot_bytes and self.dims_used < dims_slots);
        const p: [*]u8 = self.dimsBase() + self.dims_used * slot_bytes;
        @memcpy(p[0..bytes.len], bytes);
        if (self.record_slots) |list| list.append(self.gpa, self.dims_used) catch {};
        self.dims_used += 1;
        return @ptrCast(p);
    }
    /// The byte offset (from the pool's start) of the slot `dims` allocated last, for the replay recorder.
    fn lastDimsOffset(self: *Scratch) usize {
        const base = @intFromPtr(self.dimsBase()) - @intFromPtr(self.pool.ptr.?);
        return base + (self.dims_used - 1) * slot_bytes;
    }
};

/// A `dims` pool slot a recorded decode step reads, and which of its elements carries a value that moves per token
/// (the RoPE position and the cache length). `xpu_replay.zig` rewrites these before every replay.
pub const DimsSlot = struct {
    /// Byte offset of the slot from the pool's start.
    offset: usize,
    /// Index of the element that holds the RoPE position, or null.
    position_at: ?usize = null,
    /// Index of the element that holds the cache length, or null.
    cache_len_at: ?usize = null,
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
    /// When set, launches are recorded into this regular list instead of executed on the immediate queue; the
    /// caller closes and replays it (`xpu_replay.zig`). Kernel arguments are captured at record time.
    rec: ?*rt.cmdlist.Recorder = null,
    /// Which fusions this pass applies (see `Options.fuse`).
    fuse: Fuse = .none,

    fn go(self: Run, id: KernelId, g: geo.Geo, args: anytype) Error!void {
        try self.goShape(id, 0, g, args);
    }
    /// `shape` is the projection shape key for qmm-family launches (0 otherwise), used by the profiler.
    fn goShape(self: Run, id: KernelId, shape: u64, g: geo.Geo, args: anytype) Error!void {
        const global: Dim = .{ .x = g.gx, .y = g.gy, .z = g.gz };
        const local: Dim = .{ .x = g.lx, .y = g.ly, .z = g.lz };
        const launch_cfg = rt.launch.Config{ .global = global, .local = local };
        if (self.rec) |r| {
            // A recorded list cannot carry timestamps: profile only the eager path.
            if (self.sc.prof != null) return error.ReplayCannotProfile;
            try r.run(self.k.get(id), launch_cfg, args);
            return;
        }
        if (self.sc.prof) |*p| {
            const ev = p.next(id, shape) orelse return error.Invalid;
            try rt.launch.run(&self.s.queue, self.k.get(id), launch_cfg, args, ev);
            return;
        }
        try rt.launch.run(&self.s.queue, self.k.get(id), launch_cfg, args, null);
    }
    fn wp(self: Run, t: xw.Tensor) *anyopaque {
        return self.w.ptr(t);
    }
    fn projection(self: Run, l: xw.Linear, x: *anyopaque, y: *anyopaque, rows: usize) Error!void {
        try self.goShape(.qmm, Profile.shapeKey(l.outputs, l.inputs), geo.qmm(@intCast(rows), @intCast(l.outputs)), .{
            x, @as(i32, @intCast(rows)), self.wp(l.weight), self.wp(l.scales), self.wp(l.biases), y, @as(i32, @intCast(l.outputs)), @as(i32, @intCast(l.inputs)),
        });
    }
    /// Two same-shape projections of one activation row in a single launch (`qmm_pair`). Bit-identical to two
    /// `projection` calls (same fp32 accumulation order, same single bf16 RNE store); saves one launch for the
    /// launch-bound GatedDeltaNet in_proj_a/in_proj_b and the attention k_proj/v_proj pairs.
    fn projectionPair(self: Run, a: xw.Linear, a_out: *anyopaque, b: xw.Linear, b_out: *anyopaque, x: *anyopaque, rows: usize) Error!void {
        std.debug.assert(a.inputs == b.inputs and a.outputs == b.outputs);
        try self.goShape(.qmm_pair, Profile.shapeKey(a.outputs, a.inputs), geo.qmmPair(@intCast(rows), @intCast(a.outputs)), .{
            x,
            @as(i32, @intCast(rows)),
            self.wp(a.weight),
            self.wp(a.scales),
            self.wp(a.biases),
            self.wp(b.weight),
            self.wp(b.scales),
            self.wp(b.biases),
            a_out,
            b_out,
            @as(i32, @intCast(a.outputs)),
            @as(i32, @intCast(a.inputs)),
        });
    }
    /// `norm` + the next projection's input load in one launch (`norm_qmm`): the RMSNorm row is computed
    /// redundantly by each of the projection's work-groups, in norm.cl's own reduction partition, so the
    /// bf16 rounding points and the fp32 accumulation order are unchanged. `residual` null selects norm_nores.
    fn normProjection(self: Run, h: *anyopaque, residual: ?*anyopaque, weight: xw.Tensor, l: xw.Linear, y: *anyopaque, rows: usize) Error!void {
        if (residual) |r| {
            try self.goShape(.norm_qmm, Profile.shapeKey(l.outputs, l.inputs), geo.normQmm(@intCast(rows), @intCast(l.outputs), c.hidden), .{
                h, r, self.wp(weight), c.eps, h, self.wp(l.weight), self.wp(l.scales), self.wp(l.biases), y, @as(i32, @intCast(rows)), @as(i32, @intCast(l.outputs)), @as(i32, @intCast(l.inputs)),
            });
        } else {
            try self.goShape(.norm_qmm, Profile.shapeKey(l.outputs, l.inputs), geo.normQmm(@intCast(rows), @intCast(l.outputs), c.hidden), .{
                h, h, self.wp(weight), c.eps, h, self.wp(l.weight), self.wp(l.scales), self.wp(l.biases), y, @as(i32, @intCast(rows)), @as(i32, @intCast(l.outputs)), @as(i32, @intCast(l.inputs)),
            });
        }
    }
    /// SwiGLU + the down projection's input load in one launch (`mlp_act_down`): the activation is formed
    /// in-register from GATE and UP with mlp_act.cl's rounding point, so the round trip through `act` disappears.
    fn mlpDown(self: Run, gate: *anyopaque, up: *anyopaque, l: xw.Linear, y: *anyopaque, rows: usize) Error!void {
        try self.goShape(.mlp_act_down, Profile.shapeKey(l.outputs, l.inputs), geo.qmm(@intCast(rows), @intCast(l.outputs)), .{
            gate, up, self.wp(l.weight), self.wp(l.scales), self.wp(l.biases), y, @as(i32, @intCast(rows)), @as(i32, @intCast(l.outputs)), @as(i32, @intCast(l.inputs)),
        });
    }
    fn norm(self: Run, h: *anyopaque, residual: ?*anyopaque, weight: xw.Tensor, x: *anyopaque, rows: usize) Error!void {
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

/// Which fusions a forward pass applies. Every variant is bit-identical to the unfused pass (the kernels
/// preserve each bf16 rounding point and each fp32 accumulation order), so the exactness gates are the
/// arbiter: a variant whose output differs is a bug, never a tolerance question.
pub const Fuse = struct {
    /// `qmm_pair`: fuse the a/b and k/v projection pairs into one launch each (S02's 2B lever).
    pair: bool = false,
    /// `norm_qmm`: fuse each (residual + RMSNorm) into the projection that consumes its output.
    norm_qmm: bool = false,
    /// `mlp_act_down`: fuse SwiGLU into the down projection (removes the intermediate round trip).
    act_down: bool = false,

    pub const none: Fuse = .{};
    pub const all: Fuse = .{ .pair = true, .norm_qmm = true, .act_down = true };
};

pub const Options = struct {
    /// Stop after this layer's MLP (no final norm / head): the layer output is h + r in the scratch.
    stop_after: ?usize = null,
    head: Head = .last,
    /// Fusions to apply; `Fuse.none` is the unfused (gate-proven) path.
    fuse: Fuse = .none,
    /// When set, the forward *records* into this regular command list instead of executing: no work reaches the
    /// card and nothing is synchronized. The caller closes the list and replays it (see `xpu_replay.zig`). The
    /// buffers the launch arguments name are captured by address, so they must stay alive across replays; `rec`
    /// also forbids profiling and skips the cache-commit copies at the end.
    rec: ?*rt.cmdlist.Recorder = null,
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
    const ru = Run{ .s = s, .k = &m.kernels, .w = m.weights, .sc = sc, .rec = o.rec, .fuse = o.fuse };
    const w = m.weights;
    const T: u32 = @intCast(total);
    try ru.go(.embed, geo.layout(.embed, T, c.hidden, c.head_dim, c.heads, c.kv_heads), .{ sc.pool.ptr, w.ptr(w.embed.weight), w.ptr(w.embed.scales), w.ptr(w.embed.biases), sc.h.ptr });
    for (w.layers, 0..) |layer, i| {
        const residual: ?*anyopaque = if (i == 0) null else sc.r.ptr;
        // The layer's own norm writes `x` for the mixer; only the *post* norm's output can be fused into the
        // gate projection below (the mixer's projections read `x` before the post norm runs).
        switch (layer.mixer) {
            .linear => |d| {
                if (o.fuse.norm_qmm) {
                    try ru.normProjection(sc.h.ptr.?, residual, layer.input_norm, d.qkv, sc.qkv.ptr.?, total);
                } else {
                    try ru.norm(sc.h.ptr.?, residual, layer.input_norm, sc.x.ptr.?, total);
                }
                try delta(ru, d, &cache.blocks[i].linear, total);
            },
            .attention => |a| {
                if (o.fuse.norm_qmm) {
                    try ru.normProjection(sc.h.ptr.?, residual, layer.input_norm, a.q, sc.qraw.ptr.?, total);
                } else {
                    try ru.norm(sc.h.ptr.?, residual, layer.input_norm, sc.x.ptr.?, total);
                }
                try attention(ru, a, &cache.blocks[i].attention, cache.len, cache.capacity, total);
            },
        }
        if (o.fuse.norm_qmm) {
            try ru.normProjection(sc.h.ptr.?, sc.r.ptr, layer.post_norm, layer.mlp.gate, sc.gate.ptr.?, total);
        } else {
            try ru.norm(sc.h.ptr.?, sc.r.ptr, layer.post_norm, sc.x.ptr.?, total);
            try ru.projection(layer.mlp.gate, sc.x.ptr.?, sc.gate.ptr.?, total);
        }
        try ru.projection(layer.mlp.up, sc.x.ptr.?, sc.up.ptr.?, total);
        if (o.fuse.act_down) {
            // SwiGLU and the down projection in one launch: `act` is never written, so the round trip through
            // the intermediate buffer (17408 bf16 per row) disappears. The kernel forms each activation with
            // mlp_act.cl's rounding point, so the result is bit-identical to mlp_act + qmm.
            try ru.mlpDown(sc.gate.ptr.?, sc.up.ptr.?, layer.mlp.down, sc.r.ptr.?, total);
        } else {
            try ru.go(.mlp_act, geo.mlpAct(T, c.intermediate), .{ sc.gate.ptr, sc.up.ptr, sc.act.ptr });
            try ru.projection(layer.mlp.down, sc.act.ptr.?, sc.r.ptr.?, total);
        }
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
    // A recording pass stops here: the commit is a queued copy plus a cache-length bump, and neither belongs in a
    // replayed list (the caller commits once, after the last replay, from the eager path).
    if (o.rec != null) return;
    const row: usize = if (o.head == .all) total - 1 else 0;
    try s.queue.copy(cache.logits.ptr, off(sc.logits, row * c.vocab * 2), c.vocab * 2, null);
    try s.queue.synchronize();
    cache.len += total;
}

/// One decode token recorded into `rec` as a regular command list: the same launches as `run` with a single token
/// and `head = .last`, but no work reaches the card and nothing is synchronized. `slots` receives one entry per
/// `dims` array the step allocates, with the element index of every scalar that changes per token (the RoPE
/// position and the cache length), so a replay can rewrite exactly those before re-submitting the list.
/// `position` is the RoPE position (the cache length before this token). The caller closes the list.
pub fn recordDecode(
    rec: *rt.cmdlist.Recorder,
    m: *Model,
    sc: *Scratch,
    cache: *xs.Cache,
    tokens: []const u32,
    position: u32,
    slots: *std.ArrayList(DimsSlot),
    fuse: Fuse,
) Error!void {
    if (tokens.len != 1) return error.ReplayNeedsOneRow;
    // The forward allocates one `dims` slot per attention kernel; capture their pool indices in call order so the
    // per-token scalars can be classified below against the values this pass was recorded with.
    var capture: std.ArrayList(usize) = .empty;
    defer capture.deinit(sc.gpa);
    sc.record_slots = &capture;
    defer sc.record_slots = null;
    try run(nilSession(sc), m, sc, cache, tokens, .{ .head = .last, .rec = rec, .fuse = fuse });
    // Classify: a slot is `position` when it holds the RoPE position, `cache_len` when it holds the cache length.
    // Both appear in several slots (queries, keys and attn_partial all carry them), and every occurrence must be
    // rewritten, so each slot records its own first matching element.
    const pool: [*]u8 = @ptrCast(sc.pool.ptr.?);
    const base = @intFromPtr(sc.dimsBase()) - @intFromPtr(sc.pool.ptr.?);
    for (capture.items) |slot_i| {
        const d: [*]const i32 = @ptrCast(@alignCast(pool + base + slot_i * Scratch.slot_bytes));
        var s: DimsSlot = .{ .offset = base + slot_i * Scratch.slot_bytes };
        for (0..8) |j| {
            const v: u32 = @intCast(@max(d[j], 0));
            if (v == position and s.position_at == null) s.position_at = j;
            if (v == cache.len and s.cache_len_at == null) s.cache_len_at = j;
        }
        try slots.append(sc.gpa, s);
    }
}

/// The recording pass reaches no card, but `Run` still names the session (for its queue and context).
/// A recording pass never calls a launch path that uses it, so the sentinel is never dereferenced; keeping the
/// field optional would push a null check into every `goShape`.
fn nilSession(_: *Scratch) *rt.session.Session {
    return @ptrFromInt(@alignOf(rt.session.Session));
}

fn delta(ru: Run, d: xw.Gdn, cache: *xs.Gdn, total: usize) Error!void {
    const sc = ru.sc;
    const T: u32 = @intCast(total);
    try ru.projection(d.qkv, sc.x.ptr.?, sc.qkv.ptr.?, total);
    try ru.projection(d.z, sc.x.ptr.?, sc.z.ptr.?, total);
    // a and b are both N = v_heads (48) projections of the same x: one fused launch instead of two.
    if (ru.fuse.pair) {
        try ru.projectionPair(d.a, sc.a.ptr.?, d.b, sc.b.ptr.?, sc.x.ptr.?, total);
    } else {
        try ru.projection(d.a, sc.x.ptr.?, sc.a.ptr.?, total);
        try ru.projection(d.b, sc.x.ptr.?, sc.b.ptr.?, total);
    }
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

fn attention(ru: Run, a: xw.Attention, cache: *xs.Attention, len: usize, capacity: usize, total: usize) Error!void {
    const sc = ru.sc;
    const w = ru.w;
    const T: u32 = @intCast(total);
    // With `norm_qmm` the layer's norm already wrote `qraw` (the fused prologue), so q_proj must not run again.
    if (!ru.fuse.norm_qmm) try ru.projection(a.q, sc.x.ptr.?, sc.qraw.ptr.?, total);
    // k and v are both N = kv_heads*head_dim (1024) projections of the same x: one fused launch instead of two.
    if (ru.fuse.pair) {
        try ru.projectionPair(a.k, sc.k.ptr.?, a.v, sc.v.ptr.?, sc.x.ptr.?, total);
    } else {
        try ru.projection(a.k, sc.x.ptr.?, sc.k.ptr.?, total);
        try ru.projection(a.v, sc.x.ptr.?, sc.v.ptr.?, total);
    }
    // qraw holds q_proj's output: rows x qDim() bf16 (2 * heads * head_dim = 12288 at 27B, 4096 at 2B).
    // dims[2] is its element stride; the 2B value coincidentally equals 2*hidden, the 27B one does not.
    const qn = sc.dims([4]u32{ T, c.heads, @intCast(c.qDim()), 2 * c.head_dim });
    try ru.go(.head_norm, geo.layout(.head_norm, @intCast(total * c.heads), c.hidden, c.head_dim, c.heads, c.kv_heads), .{ sc.qraw.ptr, w.ptr(a.q_norm), sc.q.ptr, qn });
    const kn = sc.dims([4]u32{ T, c.kv_heads, c.kv_heads * c.head_dim, c.head_dim });
    try ru.go(.head_norm, geo.layout(.head_norm, @intCast(total * c.kv_heads), c.hidden, c.head_dim, c.heads, c.kv_heads), .{ sc.k.ptr, w.ptr(a.k_norm), sc.knorm.ptr, kn });
    // `queries` is strided by the whole round's rows (dims[1]) and this segment's base is dims[2]: the kernel's
    // QW_QROWS carries the round's row count (the launch geometry's y axis is heads*rows). The 2B passed its own
    // rows here, which is only correct while a round is a single segment; at 27B the same call would stride the
    // buffer by one row and every head but the first would read another head's query.
    try ru.go(.queries, geo.layout(.queries, T, c.hidden, c.head_dim, c.heads, c.kv_heads), .{ sc.q.ptr, sc.queries.ptr, sc.dims([4]u32{ @intCast(len), T, 0, 0 }) });
    try ru.go(.keys, geo.layout(.keys, T, c.hidden, c.head_dim, c.heads, c.kv_heads), .{ sc.knorm.ptr, sc.v.ptr, cache.keys.ptr, cache.values.ptr, sc.dims([4]u32{ @intCast(len), @intCast(capacity), 0, 0 }) });
    const nch: usize = (len + total + 127) / 128;
    const dims = sc.dims([8]i32{ @intCast(len), @intCast(total), @intCast(capacity), @intCast(nch), 0, @intCast(total), 0, 0 });
    try ru.go(.attn_partial, geo.attnPartial(@intCast(nch), T, c.heads), .{
        sc.queries.ptr, cache.keys.ptr, cache.values.ptr, @as(f32, 1.0 / 16.0), dims, sc.pm.ptr, sc.pl.ptr, sc.po.ptr,
    });
    try ru.go(.attn_merge, geo.attnMerge(T, c.heads), .{ sc.pm.ptr, sc.pl.ptr, sc.po.ptr, dims, sc.y.ptr });
    // attn_merge's OUT is strided by the model's attention row width: heads*head_dim = 6144 at 27B against
    // hidden 5120. The 2B's default (8*256 = 2048 = its hidden) hid the difference; the 27B passes it explicitly.
    try ru.go(.attention_gate, geo.layout(.attention_gate, T, c.hidden, c.head_dim, c.heads, c.kv_heads), .{ sc.y.ptr, sc.qraw.ptr, sc.mix.ptr });
    try ru.projection(a.o, sc.mix.ptr.?, sc.r.ptr.?, total);
}
