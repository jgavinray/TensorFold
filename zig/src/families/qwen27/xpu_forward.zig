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
    /// The value projection's width (in_proj_z / out_proj row count): QW_NV * QW_DV.
    pub fn gdnValueDim() usize {
        return v_heads * dv;
    }
    /// The GatedDeltaNet convolution's channel count: 2*QW_NK*QW_DK + QW_NV*QW_DV (gdn_pre.cl's QW_CC).
    pub fn gdnQkvDim() usize {
        return 2 * k_heads * dk + gdnValueDim();
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
    embed, head_norm, queries, keys, attention_gate, norm, norm_nores, qmm, qmm_xmx, qmm_xsum, qmm_split, qmm_q8, mlp_act, gdn_pre, gdn_chain, gdn_post, attn_partial, attn_partial_gqa, attn_merge,

    fn module(self: KernelId) Module {
        return switch (self) {
            .embed, .head_norm, .queries, .keys, .attention_gate => .layout,
            .qmm_xmx, .qmm_xsum, .qmm_split, .qmm_q8 => .qmm,
            .attn_partial_gqa => .attn_partial,
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

/// Every intermediate buffer's width in bf16 elements a row, as the writer that owns it demands. One entry per
/// `Scratch` buffer, in `Scratch.names` order, so `sizesOf` and the writer table cannot drift apart: the `names`
/// list and this table are indexed together, and the test below checks every entry against the widest writer.
pub const widths = struct {
    /// `pool`: ids (4 B each), conv windows (4 B each) and the `dims` slots (32 B each).
    pub fn pool(cfg_: cfgmod.Config, rows: usize) usize {
        return rows * 4 + rows * cfg_.conv_kernel * 4 + Scratch.dims_slots * Scratch.slot_bytes;
    }
    /// `h`, `x`, `r`, `z`: the residual stream, one `hidden`-wide row each.
    pub fn hidden(cfg_: cfgmod.Config, rows: usize) usize {
        return rows * cfg_.hidden * 2;
    }
    /// `v`, `y`, `mix`: the GDN value width (`v_heads*dv`), which is wider than `hidden` at 27B.
    pub fn gdnValue(cfg_: cfgmod.Config, rows: usize) usize {
        return rows * @max(cfg_.hidden, cfg_.gdnValueDim()) * 2;
    }
    /// `q`: `head_norm` writes `heads*head_dim` a row; the GDN path offsets it by `gdnValueDim()` a row, so the
    /// stride is the wider of the two.
    pub fn q(cfg_: cfgmod.Config, rows: usize) usize {
        return rows * @max(@max(cfg_.hidden, cfg_.heads * cfg_.head_dim), cfg_.gdnValueDim()) * 2;
    }
    /// `k` serves two writers: the GDN `gdn_pre` writes `k_heads*dk` a row (2048 at 27B) and attention's k
    /// projection writes `kvDim()` (`kv_heads*head_dim`, 1024). `hidden` (5120) covers both, so the buffer keeps the
    /// `hidden` width the original table gave it; sizing it by `kvDim()` alone would overrun at 27B.
    pub fn k(cfg_: cfgmod.Config, rows: usize) usize {
        return rows * @max(cfg_.hidden, @max(cfg_.kvDim(), cfg_.k_heads * cfg_.dk)) * 2;
    }
    /// `queries`: `layout.queries` writes `heads*head_dim` a row, which is the widest writer here (the attention
    /// path reads it back with the same row stride, so the buffer is `rows_q` wide as the original table had it).
    pub fn queries(cfg_: cfgmod.Config, rows: usize) usize {
        return rows * @max(cfg_.hidden, cfg_.heads * cfg_.head_dim) * 2;
    }
    /// `qkv`: `gdn_pre` reads the conv input `gdnQkvDim()` a row.
    pub fn qkv(cfg_: cfgmod.Config, rows: usize) usize {
        return rows * cfg_.gdnQkvDim() * 2;
    }
    /// `gate`, `up`, `act`: the MLP's `intermediate` width.
    pub fn intermediate(cfg_: cfgmod.Config, rows: usize) usize {
        return rows * cfg_.intermediate * 2;
    }
    /// `qraw`: the q projection's output, `qDim()` (`2*heads*head_dim`) a row.
    pub fn qraw(cfg_: cfgmod.Config, rows: usize) usize {
        return rows * cfg_.qDim() * 2;
    }
    /// `knorm`: `head_norm` writes `kv_heads*head_dim` a row.
    pub fn knorm(cfg_: cfgmod.Config, rows: usize) usize {
        return rows * cfg_.kvDim() * 2;
    }
    /// `a`, `b`, `beta`: one `v_heads`-wide row each (bf16).
    pub fn gdnHead(cfg_: cfgmod.Config, rows: usize) usize {
        return rows * cfg_.v_heads * 2;
    }
    /// `g`: the GDN gate, fp32, `v_heads` a row.
    pub fn gdnGate(cfg_: cfgmod.Config, rows: usize) usize {
        return rows * cfg_.v_heads * 4;
    }
    /// `pm`, `pl`: the attention partials, `heads * chunks` fp32 a row.
    pub fn attnPartial(cfg_: cfgmod.Config, rows: usize, chunks: usize) usize {
        return rows * cfg_.heads * chunks * 4;
    }
    /// `po`: the attention partial outputs, `heads * chunks * head_dim` fp32 a row.
    pub fn attnPartialOut(cfg_: cfgmod.Config, rows: usize, chunks: usize) usize {
        return rows * cfg_.heads * chunks * cfg_.head_dim * 4;
    }
    /// `logits`: one bf16 row a logit row.
    pub fn logits(cfg_: cfgmod.Config, logit_rows: usize) usize {
        return logit_rows * cfg_.vocab * 2;
    }
    /// `conv_snap`: the conv tail, `convBytes` a row.
    pub fn convSnap(cfg_: cfgmod.Config, rows: usize) usize {
        return rows * xs.convBytes(cfg_);
    }
    /// `xsum`: the XMX prefill matmul's per-(row, quantization group) input sums (fp32), for the widest input.
    pub fn xsum(cfg_: cfgmod.Config, rows: usize) usize {
        return rows * (@max(cfg_.intermediate, @max(cfg_.hidden, cfg_.gdnValueDim())) / 64) * 4;
    }
    /// `xq`: the decode matmul's int8 activations (qwen35_qmm_split), 2 x 64 bytes per (row, group), widest input.
    pub fn xq(cfg_: cfgmod.Config, rows: usize) usize {
        return rows * (@max(cfg_.intermediate, @max(cfg_.hidden, cfg_.gdnValueDim())) / 64) * 128;
    }
    /// `xscale`: the decode matmul's per-(row, group) int8 scale and exact input sum, fp32 each.
    pub fn xscale(cfg_: cfgmod.Config, rows: usize) usize {
        return rows * (@max(cfg_.intermediate, @max(cfg_.hidden, cfg_.gdnValueDim())) / 64) * 8;
    }
};

/// The byte size of every `Scratch` buffer, in `Scratch.names` order. Each entry is the widest writer's width (see
/// `widths`), never a coincidence of the 2B geometry: at 2B the GDN value width and the query width both happen to
/// equal `hidden`, so sizing by `hidden` was always enough there and silently 20% short at 27B.
pub fn sizesOf(config: cfgmod.Config, rows: usize, capacity: usize, logit_rows: usize) [Scratch.names.len]usize {
    const chunks = (capacity + 127) / 128;
    const w = widths;
    return .{
        w.pool(config, rows),
        w.hidden(config, rows),      w.hidden(config, rows),      w.hidden(config, rows),
        w.gdnValue(config, rows),    w.q(config, rows),           w.k(config, rows),
        w.gdnValue(config, rows),    w.gdnValue(config, rows),    w.gdnValue(config, rows),
        w.queries(config, rows),
        w.intermediate(config, rows), w.intermediate(config, rows), w.intermediate(config, rows),
        w.intermediate(config, rows),
        w.qraw(config, rows),        w.knorm(config, rows),
        w.gdnHead(config, rows),     w.gdnHead(config, rows),     w.gdnHead(config, rows),
        w.gdnGate(config, rows),
        w.attnPartial(config, rows, chunks), w.attnPartial(config, rows, chunks),
        w.attnPartialOut(config, rows, chunks),
        w.logits(config, logit_rows), w.convSnap(config, rows), w.xsum(config, rows),
        w.xq(config, rows),          w.xscale(config, rows),
    };
}

/// Per-kernel device time of forward calls (Level Zero kernel timestamps), ported from the 2B forward. Every launch gets
/// its own event; `collect` reads them after the queue has finished and rearms them. Copies are not timed.
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

    pub fn init(s: *rt.session.Session, gpa: std.mem.Allocator, capacity: u32) !Profile {
        var pool = try rt.event.Pool.init(&s.ctx, capacity);
        errdefer pool.deinit();
        const events = try gpa.alloc(rt.event.Event, capacity);
        errdefer gpa.free(events);
        const ids = try gpa.alloc(KernelId, capacity);
        errdefer gpa.free(ids);
        const shapes = try gpa.alloc(u64, capacity);
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
    pub fn shapeKey(outputs: usize, inputs: usize) u64 {
        return (@as(u64, outputs) << 32) | @as(u64, inputs);
    }
    fn next(self: *Profile, id: KernelId, shape: u64) ?*const rt.event.Event {
        if (self.used >= self.events.len) return null;
        self.ids[self.used] = id;
        self.shapes[self.used] = shape;
        self.used += 1;
        return &self.events[self.used - 1];
    }
    pub fn collect(self: *Profile, gpa: std.mem.Allocator) (rt.Error || error{OutOfMemory})!void {
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
    pub fn clear(self: *Profile) void {
        self.ns = @splat(0);
        self.count = @splat(0);
        self.shape_ns.clearRetainingCapacity();
        self.shape_count.clearRetainingCapacity();
    }
};

/// Intermediate buffers (device), plus a small host-visible pool for ids, conv windows and kernel `dims` arrays
/// (every launch gets its own slot, so the whole forward can be queued before the host waits).
pub const Scratch = struct {
    rows: usize,
    /// Rows a recording (verify) round may span: the recurrence snapshots hold this many rows per linear layer.
    /// Smaller than `rows` at 27B on purpose - see `snapshot_rows`. (Conv snapshots stay `rows` wide: gdn_pre writes
    /// one for every row of every forward, prefill chunks included.)
    snap_rows: usize,
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
    xsum: Buffer,
    xq: Buffer,
    xscale: Buffer,
    /// Optional per-kernel device timing; see `Profile`. Never set in serving.
    prof: ?Profile = null,
    /// Per-layer window snapshots for shared rounds, indexed by the absolute scratch row of the segment's last row.
    snaps_delta: [cfgmod.max_layers]?Buffer = @splat(null),
    snaps_conv: [cfgmod.max_layers]?Buffer = @splat(null),
    made: usize = 0,
    /// Snapshot buffers made so far, for `deinit`.
    made_snaps: usize = 0,
    dims_used: usize = 0,
    /// While a decode step is being recorded, every `dims` slot index lands here so the replay can rewrite the
    /// per-token scalars inside the captured arrays. Null outside recording.
    record_slots: ?*std.ArrayList(DimsSlot) = null,
    /// Allocator for `record_slots` appends; the caller's, since the scratch does not own one otherwise.
    gpa: std.mem.Allocator = undefined,

    const names = .{ "pool", "h", "x", "r", "z", "q", "k", "v", "y", "mix", "queries", "qkv", "gate", "up", "act", "qraw", "knorm", "a", "b", "beta", "g", "pm", "pl", "po", "logits", "conv_snap", "xsum", "xq", "xscale" };
    pub const dims_slots = 1024;
    pub const slot_bytes = 32;
    /// Recurrence-snapshot rows (a recording round's row budget). Each row costs one recurrence snapshot per linear layer:
    /// 48 x 3 MiB (v_heads*dv*dk fp32) = 147 MiB at 27B, so the 2B's 128 rows would be 18.4 GiB next to 15 GiB of
    /// weights on a 31.9 GiB card - an overcommit the driver resolves by evicting to system memory, which stalled
    /// decode forever (2026-10-19). 32 rows = two streams' 16-row windows (what `xpu_qualify` runs) = 4.6 GiB.
    pub const snapshot_rows = 2 * xs.window_rows;

    /// Device bytes of the snapshot buffers, per *real* linear layer (`config.layers` of them - never the `max_layers`
    /// capacity of the kinds table): `snap_rows` recurrence snapshots and `rows` conv tails.
    pub fn snapshotBytes(config: cfgmod.Config, rows: usize, snap_rows: usize) usize {
        var n: usize = 0;
        for (0..config.layers) |i| if (config.kind(i) == .linear) {
            n += snap_rows * xs.deltaBytes(config) + rows * xs.convBytes(config);
        };
        return n;
    }

    pub fn init(s: *rt.session.Session, config: cfgmod.Config, rows: usize, capacity: usize, logit_rows: usize) rt.Error!Scratch {
        return initGpa(std.heap.page_allocator, s, config, rows, capacity, logit_rows);
    }

    pub fn initGpa(gpa: std.mem.Allocator, s: *rt.session.Session, config: cfgmod.Config, rows: usize, capacity: usize, logit_rows: usize) rt.Error!Scratch {
        var out: Scratch = undefined;
        out.rows = rows;
        out.snap_rows = @min(rows, snapshot_rows);
        out.capacity = capacity;
        out.logit_rows = logit_rows;
        out.cfg = config;
        out.gpa = gpa;
        out.record_slots = null;
        out.prof = null;
        out.made = 0;
        out.made_snaps = 0;
        out.dims_used = 0;
        out.snaps_delta = @splat(null);
        out.snaps_conv = @splat(null);
        errdefer out.deinit();
        const sizes = sizesOf(config, rows, capacity, logit_rows);
        inline for (names, 0..) |n, i| {
            const kind: rt.memory.Kind = if (i == 0) .host else .device;
            @field(out, n) = try Buffer.alloc(&s.ctx, kind, sizes[i]);
            out.made = i + 1;
        }
        // One recurrence snapshot buffer (`snap_rows` rows: only a recording round writes it) and one conv snapshot
        // buffer (`rows` rows: gdn_pre writes a conv tail for every row of every forward) per real linear layer. Bounded
        // by `config.layers`: `kinds` defaults every slot past the model's layers to `.linear`, and iterating the whole
        // `max_layers` table allocated 64 phantom layers' snapshots (42.8 GiB at 27B, 2026-10-19).
        for (out.snaps_delta[0..config.layers], out.snaps_conv[0..config.layers], 0..) |*b, *cb, i| if (config.kind(i) == .linear) {
            b.* = try Buffer.alloc(&s.ctx, .device, out.snap_rows * xs.deltaBytes(config));
            out.made_snaps = i + 1;
            cb.* = try Buffer.alloc(&s.ctx, .device, rows * xs.convBytes(config));
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
    /// Bytes of the host pool the last forward wrote (ids, conv windows, its `dims` slots): a recorded step's image.
    pub fn poolUsed(self: *const Scratch) usize {
        return self.rows * 4 + self.rows * c.conv_kernel * 4 + self.dims_used * slot_bytes;
    }
    /// Copies `vals` into a fresh slot of the pool and returns its device-visible address. Its values are fixed for a
    /// recorded decode step; see `dimsMoving` for a slot holding the per-token position or cache length.
    fn dims(self: *Scratch, vals: anytype) *anyopaque {
        const bytes = std.mem.asBytes(&vals);
        std.debug.assert(bytes.len <= slot_bytes and self.dims_used < dims_slots);
        const p = self.dimsBase() + self.dims_used * slot_bytes;
        @memcpy(p[0..bytes.len], bytes);
        self.dims_used += 1;
        return @ptrCast(p);
    }
    /// `dims` for a slot whose element `position_at` holds the RoPE position and/or `cache_len_at` the cache length.
    /// While a decode step is recorded the slot and those indices go to `record_slots`, and the replay rewrites exactly
    /// them. (The indices used to be found by matching values against the position: any fixed element that happened
    /// to equal it - rows 1, kv heads 4, heads 24, ... for a prompt of that length - was rewritten every token.)
    fn dimsMoving(self: *Scratch, vals: anytype, position_at: ?usize, cache_len_at: ?usize) Error!*anyopaque {
        const p = self.dims(vals);
        if (self.record_slots) |list| try list.append(self.gpa, .{
            .offset = @intFromPtr(p) - @intFromPtr(self.pool.ptr.?),
            .position_at = position_at,
            .cache_len_at = cache_len_at,
        });
        return p;
    }

    /// The recurrence snapshot of `layer` at absolute scratch row `row` (`deltaBytes` long); `keep` reads it back.
    pub fn snapDelta(self: *Scratch, layer: usize, row: usize) *anyopaque {
        const b = self.snaps_delta[layer] orelse unreachable;
        std.debug.assert(row < self.snap_rows);
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
    /// A non-recording forward (prefill chunks, the logits tools): projections of >= 8 rows may use the XMX matmul.
    /// Recording rounds (verify windows) and recorded decode steps stay on the scalar kernel, byte-exact with one-row
    /// execution as the startup qualify requires.
    prefill: bool = false,

    fn go(self: Run, id: KernelId, g: geo.Geo, args: anytype) rt.Error!void {
        try self.goShape(id, 0, g, args);
    }
    /// `shape` is the projection's shape key (0 for other kernels), used by the profiler.
    fn goShape(self: Run, id: KernelId, shape: u64, g: geo.Geo, args: anytype) rt.Error!void {
        const global: Dim = .{ .x = g.gx, .y = g.gy, .z = g.gz };
        const local: Dim = .{ .x = g.lx, .y = g.ly, .z = g.lz };
        const launch_cfg = rt.launch.Config{ .global = global, .local = local };
        if (self.rec) |r| {
            if (self.sc.prof != null) return error.Invalid; // a recorded list carries no timestamps
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
    /// Prefill on the XMX engines: 5-7x the scalar kernel on the 27B shapes at the bf16 accuracy floor
    /// (exp/kernels/qmm_xmx/xmx_check.py). Narrow outputs (the GDN a/b heads) keep the scalar kernel.
    fn xmx(self: Run, l: xw.Linear, rows: usize) bool {
        return self.prefill and rows >= 8 and l.outputs >= 1024 and l.inputs % 64 == 0 and self.sc.cfg.group == 64;
    }
    fn projection(self: Run, l: xw.Linear, x: *anyopaque, y: *anyopaque, rows: usize) rt.Error!void {
        // The XMX kernel reads X with 2D block reads, which need a 64-byte-aligned base; any other X takes q8.
        if (self.xmx(l, rows) and @intFromPtr(x) % 64 == 0) {
            try self.go(.qmm_xsum, geo.qmmXsum(@intCast(rows), @intCast(l.inputs / 64)), .{
                x, @as(i32, @intCast(rows)), self.sc.xsum.ptr, @as(i32, @intCast(l.inputs)),
            });
            try self.goShape(.qmm_xmx, Profile.shapeKey(l.outputs, l.inputs), geo.qmmXmx(@intCast(rows), @intCast(l.outputs)), .{
                x, @as(i32, @intCast(rows)), self.wp(l.weight), self.wp(l.scales), self.wp(l.biases), y, @as(i32, @intCast(l.outputs)), @as(i32, @intCast(l.inputs)), self.sc.xsum.ptr,
            });
            return;
        }
        try self.q8(&.{.{ .l = l, .y = y }}, x, rows);
    }
    const Proj = struct { l: xw.Linear, y: *anyopaque };
    /// Up to four projections of one input `x` (same `inputs`), narrow ones listed first. Those the XMX path takes go
    /// one by one; the rest share one `q8` launch. A narrow projection alone is a few work-groups at latency (GDN a/b:
    /// ~55 us each, 96 per token); dispatched first inside its wide siblings' launch it costs nothing extra.
    fn projections(self: Run, ps: []const Proj, x: *anyopaque, rows: usize) rt.Error!void {
        std.debug.assert(ps.len <= 4);
        var rest: [4]Proj = undefined;
        var n: usize = 0;
        for (ps) |p| {
            std.debug.assert(p.l.inputs == ps[0].l.inputs);
            if (self.xmx(p.l, rows) and @intFromPtr(x) % 64 == 0) {
                try self.projection(p.l, x, p.y, rows);
            } else {
                rest[n] = p;
                n += 1;
            }
        }
        if (n > 0) try self.q8(rest[0..n], x, rows);
    }
    /// The decode matmul: W4A8x2 on dp4a (qwen35_qmm_split + qwen35_qmm_q8, shared/qmm/qmm.cl). At the 100 W cap decode is
    /// energy-bound and this does the math for ~0.4x the scalar kernel's energy: 281 vs 193 GB/s sustained on the widest
    /// projection, rms error 0.50x the bf16 floor (exp/kernels/qmm_dp4a, exp/kernels/power). Rows are independent.
    fn q8(self: Run, ps: []const Proj, x: *anyopaque, rows: usize) rt.Error!void {
        std.debug.assert(ps.len >= 1 and ps.len <= 4 and self.sc.cfg.group == 64);
        const k = ps[0].l.inputs;
        try self.go(.qmm_split, geo.qmmSplit(@intCast(rows), @intCast(k / 64)), .{
            x, @as(i32, @intCast(rows)), self.sc.xq.ptr, self.sc.xscale.ptr, @as(i32, @intCast(k)),
        });
        var seg: [4]Proj = undefined;
        var outs: [4]u32 = @splat(0);
        var nn: [4]i32 = @splat(0);
        var total: u32 = 0;
        for (0..4) |i| {
            seg[i] = ps[if (i < ps.len) i else 0]; // absent segments: N = 0, pointers never read
            if (i < ps.len) {
                outs[i] = @intCast(ps[i].l.outputs);
                nn[i] = @intCast(ps[i].l.outputs);
                total += outs[i];
            }
        }
        try self.goShape(.qmm_q8, Profile.shapeKey(total, k), geo.qmmQ8(@intCast(rows), outs), .{
            self.sc.xq.ptr,                                          self.sc.xscale.ptr,  @as(i32, @intCast(rows)), @as(i32, @intCast(k)),
            self.wp(seg[0].l.weight), self.wp(seg[0].l.scales), self.wp(seg[0].l.biases), seg[0].y, nn[0],
            self.wp(seg[1].l.weight), self.wp(seg[1].l.scales), self.wp(seg[1].l.biases), seg[1].y, nn[1],
            self.wp(seg[2].l.weight), self.wp(seg[2].l.scales), self.wp(seg[2].l.biases), seg[2].y, nn[2],
            self.wp(seg[3].l.weight), self.wp(seg[3].l.scales), self.wp(seg[3].l.biases), seg[3].y, nn[3],
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
    /// Receives every `dims` slot of the recorded step that holds a per-token scalar (position, cache length) and the
    /// element index of each, so the replay rewrites exactly those. Only used when `rec` is set.
    slots: ?*std.ArrayList(DimsSlot) = null,
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
    std.debug.assert(position == cache.len);
    try runSegments(s, m, sc, &.{.{ .cache = cache, .rows = 1 }}, tokens, false, .{ .head = .last, .rec = rec, .slots = slots });
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
    if (record and (segments.len > max_segments or total > sc.snap_rows)) return error.QwenRowsExceeded;
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
    const ru = Run{ .s = s, .k = &m.kernels, .w = m.weights, .sc = sc, .rec = o.rec, .prefill = !record and o.rec == null };
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
        try ru.projections(&.{ .{ .l = layer.mlp.gate, .y = sc.gate.ptr.? }, .{ .l = layer.mlp.up, .y = sc.up.ptr.? } }, sc.x.ptr.?, total);
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
    // A recording pass stops here, after the logits row lands where a plain decode leaves it (`cache.logits`, which the
    // backend reads); the cache-length bump is the replayer's. (Without the copy a replayed token's logits stayed in
    // the scratch and the caller read the previous token's.)
    if (o.rec) |r| {
        try r.copy(segments[0].cache.logits.ptr, sc.logits.ptr, c.vocab * 2);
        return;
    }
    base = 0;
    for (segments) |seg| {
        const row: usize = if (o.head == .all) base + seg.rows - 1 else 0;
        try s.queue.copy(seg.cache.logits.ptr, off(sc.logits, row * c.vocab * 2), c.vocab * 2, null);
        var rec: [cfg.max_layers]?*const anyopaque = @splat(null);
        var cv: [cfg.max_layers]?*const anyopaque = @splat(null);
        // Snapshots exist only for real linear layers and only a recording round wrote them.
        const n_layers = seg.cache.c.layers;
        if (record) for (rec[0..n_layers], cv[0..n_layers], 0..) |*r, *v, i| if (seg.cache.c.kind(i) == .linear) {
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
    try ru.projections(&.{
        .{ .l = d.a, .y = sc.a.ptr.? }, .{ .l = d.b, .y = sc.b.ptr.? }, .{ .l = d.qkv, .y = sc.qkv.ptr.? }, .{ .l = d.z, .y = sc.z.ptr.? },
    }, sc.x.ptr.?, total);
    const w = ru.w;
    var base: usize = 0;
    for (segments) |seg| {
        const cache = &seg.cache.blocks[layer].linear;
        // The conv snapshot is written at this segment's absolute scratch row.
        const snap_conv: *anyopaque = sc.snapConv(layer, base);
        // gdn_pre reads windows[w * taps + tap] for the segment's own rows w = 0..rows-1, and runSegments wrote this
        // segment's windows at scratch row base + w: hand it the window table shifted by `base`, as the qkv input is.
        // (Unshifted, every row past the first of a segment at base > 0 read another segment's window - a shared
        // round with a multi-row draft window diverged from the same stream served alone, 2026-10-19 stress test.)
        try ru.go(.gdn_pre, geo.gdnPre(@intCast(seg.rows), 2 * c.k_heads + c.v_heads), .{
            off(sc.qkv, base * sc.cfg.gdnQkvDim() * 2), cache.conv.ptr, w.ptr(d.conv), @as(*anyopaque, @ptrCast(sc.windows()[base * c.conv_kernel ..].ptr)),
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

fn attention(ru: Run, a: xw.Attention, layer: usize, segments: []const Segment, total: usize) Error!void {
    const sc = ru.sc;
    const w = ru.w;
    const T: u32 = @intCast(total);
    try ru.projections(&.{ .{ .l = a.k, .y = sc.k.ptr.? }, .{ .l = a.v, .y = sc.v.ptr.? }, .{ .l = a.q, .y = sc.qraw.ptr.? } }, sc.x.ptr.?, total);
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
        // dims[1] is the row stride of the shared `queries` buffer, dims[2] this segment's base. The kernels read input
        // row dims[2] + r themselves (layout.cl: m = dims[2] + r), so the inputs are the unshifted buffers - shifting
        // the pointer by `base` too read every segment past the first `base` rows late (and `v` at the wrong stride).
        try ru.go(.queries, geo.layout(.queries, rows, c.hidden, c.head_dim, c.heads, c.kv_heads), .{
            sc.q.ptr, sc.queries.ptr, try sc.dimsMoving([4]u32{ @intCast(seg.cache.len), T, @intCast(base), 0 }, 0, null),
        });
        try ru.go(.keys, geo.layout(.keys, rows, c.hidden, c.head_dim, c.heads, c.kv_heads), .{
            sc.knorm.ptr, sc.v.ptr, cache.keys.ptr, cache.values.ptr, try sc.dimsMoving([4]u32{ @intCast(seg.cache.len), @intCast(seg.cache.capacity), @intCast(base), 0 }, null, 0),
        });
        // A recorded decode step is replayed while the cache grows, and its launch geometry and dims are fixed at record
        // time: it covers the cache's whole capacity. Chunks past a row's last key write PL = 0, which the merge skips,
        // so a row's bits do not depend on the chunk count. (Sized to the length at record time, the replay ignored
        // every key past that chunk boundary.)
        const nch: usize = if (ru.rec != null) (seg.cache.capacity + 127) / 128 else (seg.cache.len + seg.rows + 127) / 128;
        const dims = try sc.dimsMoving([8]i32{ @intCast(seg.cache.len), @intCast(seg.rows), @intCast(seg.cache.capacity), @intCast(nch), 0, @intCast(total), @intCast(base), 0 }, null, 0);
        try ru.go(.attn_partial_gqa, geo.attnPartialGqa(@intCast(nch), rows, c.kv_heads, c.heads / c.kv_heads), .{
            sc.queries.ptr, cache.keys.ptr, cache.values.ptr, @as(f32, 1.0 / 16.0), dims, sc.pm.ptr, sc.pl.ptr, sc.po.ptr,
        });
        try ru.go(.attn_merge, geo.attnMerge(rows, c.heads), .{ sc.pm.ptr, sc.pl.ptr, sc.po.ptr, dims, off(sc.y, base * c.gdnValueDim() * 2) });
        base += seg.rows;
    }
    try ru.go(.attention_gate, geo.layout(.attention_gate, T, c.hidden, c.head_dim, c.heads, c.kv_heads), .{ sc.y.ptr, sc.qraw.ptr, sc.mix.ptr });
    try ru.projection(a.o, sc.mix.ptr.?, sc.r.ptr.?, total);
}

test "the serving scratch fits a B70 next to the 27B weights: real layers only, bounded snapshot rows" {
    // 2026-10-19: the serving scratch asked for ~44 GiB on a 31.9 GiB card (snapshots for 64 phantom layers past the
    // model's 64, at 128 rows each); the driver evicted to system memory and decode stalled forever. This pins the
    // arithmetic on the host so it cannot regress silently: the real layer schedule, the snapshot row budget, and
    // the whole non-weight footprint of the serving scratch (rows 128, context 32768, 256 logit rows).
    const g = cfgmod.Pinned;
    var config = cfgmod.Config{
        .hidden = g.hidden,
        .intermediate = g.intermediate,
        .layers = g.layers,
        .vocab = g.vocab,
        .heads = g.heads,
        .kv_heads = g.kv_heads,
        .head_dim = g.head_dim,
        .k_heads = g.k_heads,
        .v_heads = g.v_heads,
        .dk = g.dk,
        .dv = g.dv,
        .conv_kernel = g.conv_kernel,
        .eps = g.eps,
        .rope_dims = g.rotary_dim,
        .rope_theta = g.theta,
    };
    // As config.parse sets it: full attention every 4th layer. Slots past `layers` keep the table default.
    for (config.kinds[0..config.layers], 0..) |*k, i| k.* = if ((i + 1) % 4 == 0) .attention else .linear;
    var linear: usize = 0;
    for (0..config.layers) |i| linear += @intFromBool(config.kind(i) == .linear);
    try std.testing.expectEqual(@as(usize, 48), linear);
    try std.testing.expectEqual(48 * (Scratch.snapshot_rows * xs.deltaBytes(config) + 128 * xs.convBytes(config)),
        Scratch.snapshotBytes(config, 128, Scratch.snapshot_rows));
    // Static non-weight footprint of the serving scratch must leave room for 15 GiB of weights and per-stream caches
    // on a 31.9 GiB card: budget 8 GiB (it is ~5.6 GiB; the defect above made it ~44 GiB).
    const sizes = sizesOf(config, 128, 32768, xs.max_segments * xs.window_rows);
    var total: usize = Scratch.snapshotBytes(config, 128, @min(128, Scratch.snapshot_rows));
    for (sizes) |b| total += b;
    try std.testing.expect(total <= 8 << 30);
}

test "every scratch buffer is at least as wide as the widest writer that fills it" {
    // The 27B geometry. The GDN value width (v_heads*dv = 6144) and the attention query width (heads*head_dim =
    // 6144) both exceed `hidden` (5120); the GDN key width (k_heads*dk = 2048) exceeds the attention KV width
    // (kv_heads*head_dim = 1024). Sizing by the wrong one is a device-buffer overrun, so each buffer is checked
    // against every writer that touches it, at several row counts.
    const g = cfgmod.Pinned;
    const config = cfgmod.Config{
        .hidden = g.hidden,
        .intermediate = g.intermediate,
        .layers = g.layers,
        .vocab = g.vocab,
        .heads = g.heads,
        .kv_heads = g.kv_heads,
        .head_dim = g.head_dim,
        .k_heads = g.k_heads,
        .v_heads = g.v_heads,
        .dk = g.dk,
        .dv = g.dv,
        .conv_kernel = g.conv_kernel,
        .eps = g.eps,
        .rope_dims = g.rotary_dim,
        .rope_theta = g.theta,
    };
    for ([_]usize{ 1, 16, 128 }) |rows| {
        const sizes = sizesOf(config, rows, 4096, 16);
        const w = widths;
        // Indices are `Scratch.names`: 1 h, 4 z, 5 q, 6 k, 7 v, 8 y, 9 mix, 10 queries, 15 qraw, 16 knorm, 25 conv_snap.
        const need = .{
            .{ sizes[1], w.hidden(config, rows) },    .{ sizes[4], w.gdnValue(config, rows) },
            .{ sizes[5], w.q(config, rows) },         .{ sizes[6], w.k(config, rows) },
            .{ sizes[7], w.gdnValue(config, rows) },  .{ sizes[8], w.gdnValue(config, rows) },
            .{ sizes[9], w.gdnValue(config, rows) },  .{ sizes[10], w.queries(config, rows) },
            .{ sizes[15], w.qraw(config, rows) },     .{ sizes[16], w.knorm(config, rows) },
            .{ sizes[25], w.convSnap(config, rows) },
        };
        inline for (need) |pair| try std.testing.expectEqual(pair[1], pair[0]);
        // The widths the 27B geometry demands, stated so a future edit cannot quietly narrow a buffer below a writer.
        try std.testing.expect(sizes[5] >= rows * g.heads * g.head_dim * 2); // q: head_norm's query width
        try std.testing.expect(sizes[5] >= rows * config.gdnValueDim() * 2); // q: the GDN path's per-row stride
        try std.testing.expect(sizes[6] >= rows * g.k_heads * g.dk * 2); // k: the GDN key width
        try std.testing.expect(sizes[6] >= rows * config.kvDim() * 2); // k: the attention k projection
        try std.testing.expect(sizes[7] >= rows * config.gdnValueDim() * 2); // v: the GDN value width
        try std.testing.expect(sizes[9] >= rows * config.gdnValueDim() * 2); // mix: gdn_post + attention_gate
        try std.testing.expect(sizes[10] >= rows * g.heads * g.head_dim * 2); // queries: layout.queries' width
    }
}

test "the 2B geometry stays within the same buffer table" {
    // The 2B's coincidences (gdnValueDim == hidden == heads*head_dim) must keep every buffer exactly at `hidden`
    // or wider; a table that only fits 27B would break the 2B gates.
    const config = cfgmod.Config{
        .hidden = 2048,
        .intermediate = 6144,
        .layers = 24,
        .vocab = 248320,
        .heads = 8,
        .kv_heads = 2,
        .head_dim = 256,
        .k_heads = 16,
        .v_heads = 16,
        .dk = 128,
        .dv = 128,
        .conv_kernel = 4,
        .eps = 1e-6,
        .rope_dims = 64,
        .rope_theta = 10000000,
    };
    const rows: usize = 16;
    const sizes = sizesOf(config, rows, 4096, 16);
    try std.testing.expectEqual(rows * 2048 * 2, sizes[1]);
    try std.testing.expectEqual(rows * 2048 * 2, sizes[5]);
    try std.testing.expectEqual(rows * 2048 * 2, sizes[6]);
    try std.testing.expectEqual(rows * 16 * 128 * 2, sizes[4]);
}
