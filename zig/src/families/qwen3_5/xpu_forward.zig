//! Qwen3.5-2B forward pass on an Intel B70 (Level Zero + OpenCL kernels), mirroring forward.zig's op order and rounding
//! points (bf16 activations between every op, fp32 recurrence). Kernels are loaded by key from ocloc-compiled modules
//! embedded in the executable at build time (build option `xpu_qwen35_kernels`, @embedFile of the ocloc outputs), so the
//! binary works from any working directory. The kernels are the shared sources (zig/kernels/xpu/shared/<key>/) built with
//! their 2B-default shape defines; a kernel's launch geometry comes from xpu_geometry.zig, which takes the shape values.
//! TF_XPU_QWEN35_KERNELS names a directory of <module key>.bin files that replaces the embedded modules (development
//! override; the geometry still follows the running shapes).
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
pub const Error = rt.Error || std.mem.Allocator.Error || error{ QwenRowsExceeded, QwenContextFull, InvalidQwenToken, KernelBinaryMissing, ReplayCannotProfile, ReplayNeedsOneRow };
extern "c" fn getenv(name: [*:0]const u8) ?[*:0]const u8;

/// Module keys (file names under shared/<key>/) and the kernels each provides.
const Module = enum { layout, norm, norm_nores, qmm, qmm_pair, mlp_act, gdn_pre, gdn_chain, gdn_post, attn_partial, attn_merge };
pub const KernelId = enum {
    embed, head_norm, queries, keys, attention_gate, norm, norm_nores, qmm, qmm_pair, mlp_act, gdn_pre, gdn_chain, gdn_post, attn_partial, attn_merge,

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
        std.debug.print("xpu qwen3.5 kernels: shared ({d} modules)", .{self.loaded});
        inline for (std.enums.values(Module)) |m| std.debug.print(" {s}", .{@tagName(m)});
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
    /// Per-qmm-shape device time and launch count, keyed by `Profile.shapeKey(outputs, inputs)`. Every launch gets
    /// its shape recorded at queue time, so `collect` attributes each timestamp event to its own shape.
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
    /// Shape key of a qmm launch: `(outputs << 32) | inputs`, so both fit for every Qwen shape.
    pub fn shapeKey(outputs: usize, inputs: usize) u64 {
        return (@as(u64, outputs) << 32) | @as(u64, inputs);
    }
    pub fn shapeOutputs(key: u64) usize {
        return @intCast(key >> 32);
    }
    pub fn shapeInputs(key: u64) usize {
        return @intCast(key & 0xffffffff);
    }
    /// Reserves the next timestamp event for `id`; `shape` is the qmm shape key (0 for non-qmm kernels).
    fn next(self: *Profile, id: KernelId, shape: u64) ?*const rt.event.Event {
        if (self.used >= self.events.len) return null;
        self.ids[self.used] = id;
        self.shapes[self.used] = shape;
        self.used += 1;
        return &self.events[self.used - 1];
    }
    /// Adds the launches since the last call to `ns`/`count`/`shape_ns` and rearms their events.
    pub fn collect(self: *Profile, gpa: std.mem.Allocator) (rt.Error || error{OutOfMemory})!void {
        for (self.events[0..self.used], self.ids[0..self.used], self.shapes[0..self.used]) |*e, id, shape| {
            const t = try e.kernelNs();
            self.ns[@intFromEnum(id)] += t;
            self.count[@intFromEnum(id)] += 1;
            if (id == .qmm) {
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
    /// Per-layer window snapshots for shared rounds, indexed by the absolute scratch row of the segment's last row
    /// (as state.zig's Cache does): the recurrence and the conv tail are saved once per layer, so layers never
    /// overwrite each other's window state. A stream's `keep` reads its slot back; see `xpu_backend.Entry.keep`.
    snaps_delta: [c.layers]?Buffer = @splat(null),
    snaps_conv: [c.layers]?Buffer = @splat(null),
    /// Optional per-kernel device timing (Level Zero event timestamps); see `Profile`.
    prof: ?Profile = null,
    made: usize = 0,
    /// Snapshot buffers made so far (the `snaps_delta` entries up to this index), for `deinit`.
    made_snaps: usize = 0,
    dims_used: usize = 0,
    /// While a decode step is being recorded, every `dims` slot index lands here so the replay can rewrite the
    /// per-token scalars inside the captured arrays (see `xpu_replay.zig`). Null outside recording.
    record_slots: ?*std.ArrayList(usize) = null,
    /// Allocator for `record_slots` appends; the caller's, since the scratch does not own one otherwise.
    gpa: std.mem.Allocator = undefined,

    const names = .{ "pool", "h", "x", "r", "z", "q", "k", "v", "y", "mix", "queries", "qkv", "gate", "up", "act", "qraw", "knorm", "a", "b", "beta", "g", "pm", "pl", "po", "logits", "conv_snap" };
    pub const dims_slots = 1024;
    pub const slot_bytes = 32;

    pub fn init(gpa: std.mem.Allocator, s: *rt.session.Session, rows: usize, capacity: usize, logit_rows: usize) rt.Error!Scratch {
        var out: Scratch = undefined;
        out.rows = rows;
        out.capacity = capacity;
        out.logit_rows = logit_rows;
        out.gpa = gpa;
        out.record_slots = null;
        out.made = 0;
        out.dims_used = 0;
        out.prof = null;
        out.snaps_delta = @splat(null);
        out.snaps_conv = @splat(null);
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
        // One recurrence and one conv snapshot buffer per linear layer, `rows` rows wide (the scratch row of each
        // window row). Per layer, so one layer's window state never overwrites another's before `commit` reads it.
        for (&out.snaps_delta, 0..) |*b, i| if (c.linear(i)) {
            b.* = try Buffer.alloc(&s.ctx, .device, rows * xs.delta_bytes);
            out.made_snaps = i + 1;
        };
        for (&out.snaps_conv, 0..) |*b, i| if (c.linear(i)) {
            b.* = try Buffer.alloc(&s.ctx, .device, rows * xs.conv_bytes);
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
        return @as([*]i32, @ptrCast(@alignCast(@as([*]u8, @ptrCast(self.pool.ptr.?)) + self.rows * 4)))[0 .. self.rows * c.conv_taps];
    }
    fn dimsBase(self: *Scratch) [*]u8 {
        return @as([*]u8, @ptrCast(self.pool.ptr.?)) + self.rows * 4 + self.rows * c.conv_taps * 4;
    }
    /// Copies `vals` into a fresh slot of the pool and returns its device-visible address. With a recorder active the
    /// slot's index is appended to `record_slots` so the replay can rewrite the per-token scalars inside it.
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

    /// The recurrence snapshot of `layer` at absolute scratch row `row` (`delta_bytes` long); the kernel writes it
    /// when a segment records, and `xpu_backend.Entry.keep` reads it back.
    pub fn snapDelta(self: *Scratch, layer: usize, row: usize) *anyopaque {
        const b = self.snaps_delta[layer] orelse unreachable;
        std.debug.assert(row < self.rows);
        return @ptrFromInt(@intFromPtr(b.ptr.?) + row * xs.delta_bytes);
    }
    /// The conv-tail snapshot of `layer` at absolute scratch row `row` (`conv_bytes` long); the kernel writes it
    /// when a segment records, and `xpu_backend.Entry.keep` reads it back.
    pub fn snapConv(self: *Scratch, layer: usize, row: usize) *anyopaque {
        const b = self.snaps_conv[layer] orelse unreachable;
        std.debug.assert(row < self.rows);
        return @ptrFromInt(@intFromPtr(b.ptr.?) + row * xs.conv_bytes);
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
    /// When set, launches are recorded into this regular list instead of executed on the immediate queue; the
    /// caller closes and replays it (xpu_replay.zig). Kernel arguments are captured at record time.
    rec: ?*rt.cmdlist.Recorder = null,

    fn go(self: Run, id: KernelId, g: geo.Geo, args: anytype) Error!void {
        try self.goShape(id, 0, g, args);
    }
    /// `shape` is the qmm shape key for `.qmm` launches (0 otherwise), used by the profiler.
    fn goShape(self: Run, id: KernelId, shape: u64, g: geo.Geo, args: anytype) Error!void {
        const global: Dim = .{ .x = g.gx, .y = g.gy, .z = g.gz };
        const local: Dim = .{ .x = g.lx, .y = g.ly, .z = g.lz };
        const cfg = rt.launch.Config{ .global = global, .local = local };
        if (self.rec) |r| {
            // A recorded list cannot carry timestamps: profile only the eager path.
            if (self.sc.prof != null) return error.ReplayCannotProfile;
            try r.run(self.k.get(id), cfg, args);
            return;
        }
        if (self.sc.prof) |*p| {
            const ev = p.next(id, shape) orelse return error.Invalid;
            try rt.launch.run(&self.s.queue, self.k.get(id), cfg, args, ev);
            return;
        }
        try rt.launch.run(&self.s.queue, self.k.get(id), cfg, args, null);
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
    /// `projection` calls; saves one launch for the launch-bound N=16 GatedDeltaNet a/b pair and the N=512
    /// attention k/v pair.
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

pub const Options = struct {
    /// Stop after this layer's MLP (no final norm / head): the layer output is h + r in the scratch.
    stop_after: ?usize = null,
    head: Head = .last,
    /// When set, the forward *records* into this regular command list instead of executing: no work reaches the card
    /// and nothing is synchronized. The caller closes the list and replays it (see `xpu_replay.zig`). The buffers the
    /// launch arguments name are captured by address, so they must stay alive across replays; `rec` also forbids
    /// profiling (a recorded list carries no timestamps) and the cache-commit copies at the end are skipped.
    rec: ?*rt.cmdlist.Recorder = null,
    /// Receives the pool slot index of every `dims` array the recorded step allocates, so the replay can find the
    /// per-token scalars (position, cache length) inside them. Only used when `rec` is set.
    slots: ?*std.ArrayList(usize) = null,
};

/// A stream's rows in one shared round: its cache and how many rows it contributes, in segment order (each segment's
/// rows land in the scratch after the previous segment's). With `record` the forward saves the window's recurrence
/// and conv snapshots so a later `keep` can roll back; without it (the single-stream path) only the final state is kept.
pub const Segment = struct {
    cache: *xs.Cache,
    rows: usize,
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
/// array the step allocates, with the element index of every scalar that changes per token (the RoPE position and the
/// cache length), so a replay can rewrite exactly those before re-submitting the list. `position` is the RoPE position
/// (the cache length before this token). Caller closes the list (`Recorded.take`).
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
    // The forward records one `dims` slot per attention/linear kernel; capture their pool indices in call order so the
    // per-token scalars can be classified below against the values this pass was recorded with.
    var capture: std.ArrayList(usize) = .empty;
    defer capture.deinit(sc.gpa);
    sc.record_slots = &capture;
    defer sc.record_slots = null;
    try runSegments(s, m, sc, &.{.{ .cache = cache, .rows = 1 }}, tokens, false, .{ .head = .last, .rec = rec, .slots = &capture });
    // Classify: a slot is `position` when it holds the RoPE position, `cache_len` when it holds the cache length.
    // Both appear in several slots (queries, keys and attn_partial all carry them), and every occurrence must be
    // rewritten, so each slot records its own first matching element.
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
/// forward). Each segment's `base` is its first row in the scratch (computed here from the segment order). With
/// `record`, every row of a linear layer's recurrence and each segment's conv tail are saved to its `slot`'s snapshot
/// so a later `keep` can roll back; `segments.len == 1 and !record` is the plain single-stream path.
pub fn runSegments(s: *rt.session.Session, m: *Model, sc: *Scratch, segments: []const Segment, tokens: []const u32, record: bool, o: Options) Error!void {
    var total: usize = 0;
    for (segments) |seg| {
        if (seg.rows == 0 or seg.cache.len + seg.rows > seg.cache.capacity) return error.QwenContextFull;
        total += seg.rows;
    }
    if (total == 0 or total != tokens.len or total > sc.rows or (o.head == .all and total > sc.logit_rows)) return error.QwenRowsExceeded;
    if (record and (segments.len > max_segments or total > xs.max_segments * xs.window_rows)) return error.QwenRowsExceeded;
    if (!record and segments.len != 1) return error.QwenRowsExceeded;
    for (tokens) |id| if (id >= c.vocab) return error.InvalidQwenToken;
    @memcpy(sc.ids()[0..total], tokens);
    // Each segment's conv windows are relative to its own rows (row r of a segment sees taps r..r+TAPS-1), and its
    // scratch row is the running `base`, exactly as the Metal path lays them out.
    var base: usize = 0;
    for (segments) |seg| {
        for (0..seg.rows) |r| for (0..c.conv_taps) |tap| {
            sc.windows()[(base + r) * c.conv_taps + tap] = @intCast(r + tap);
        };
        base += seg.rows;
    }
    sc.dims_used = 0;
    sc.record_slots = o.slots;
    defer sc.record_slots = null;
    const ru = Run{ .s = s, .k = &m.kernels, .w = m.weights, .sc = sc, .rec = o.rec };
    const w = m.weights;
    const T: u32 = @intCast(total);
    try ru.go(.embed, geo.layout(.embed, T, c.hidden, c.head_dim, c.query_heads, c.kv_heads), .{ sc.pool.ptr, w.ptr(w.embedding.weight), w.ptr(w.embedding.scales), w.ptr(w.embedding.biases), sc.h.ptr });
    for (w.blocks, 0..) |block, i| {
        try ru.norm(sc.h.ptr.?, if (i == 0) null else sc.r.ptr, block.input_norm, sc.x.ptr.?, total);
        switch (block.mixer) {
            .delta => |d| try delta(ru, d, i, segments, record, total),
            .attention => |a| try attention(ru, a, i, segments, total),
        }
        try ru.norm(sc.h.ptr.?, sc.r.ptr, block.post_norm, sc.x.ptr.?, total);
        try ru.projection(block.gate, sc.x.ptr.?, sc.gate.ptr.?, total);
        try ru.projection(block.up, sc.x.ptr.?, sc.up.ptr.?, total);
        try ru.go(.mlp_act, geo.mlpAct(T, c.intermediate), .{ sc.gate.ptr, sc.up.ptr, sc.act.ptr });
        try ru.projection(block.down, sc.act.ptr.?, sc.r.ptr.?, total);
        if (o.stop_after) |n| if (i == n) {
            if (o.rec == null) try s.queue.synchronize();
            return;
        };
    }
    try ru.norm(sc.h.ptr.?, sc.r.ptr, w.norm, sc.x.ptr.?, total);
    if (o.head == .all) {
        try ru.projection(w.head(), sc.x.ptr.?, sc.logits.ptr.?, total);
    } else {
        try ru.projection(w.head(), off(sc.x, (total - 1) * c.hidden * 2), sc.logits.ptr.?, 1);
    }
    // Each segment's committed logits row: the last row of its range (head = .all keeps every row in sc.logits).
    // A recording pass stops here: the commit is a queued copy plus a cache-length bump, and neither belongs in a
    // replayed list (the caller commits once, after the last replay, from the eager path).
    if (o.rec != null) return;
    base = 0;
    for (segments) |seg| {
        const row: usize = if (o.head == .all) base + seg.rows - 1 else 0;
        try s.queue.copy(seg.cache.logits.ptr, off(sc.logits, row * c.vocab * 2), c.vocab * 2, null);
        var rec: [c.layers]?*const anyopaque = @splat(null);
        var cv: [c.layers]?*const anyopaque = @splat(null);
        for (&rec, &cv, 0..) |*r, *v, i| if (c.linear(i)) {
            r.* = sc.snapDelta(i, base + seg.rows - 1);
            v.* = sc.snapConv(i, base + seg.rows - 1);
        };
        try seg.cache.commit(s, base, seg.rows, record, rec, cv);
        base += seg.rows;
    }
    try s.queue.synchronize();
}

fn delta(ru: Run, d: xw.Delta, layer: usize, segments: []const Segment, record: bool, total: usize) Error!void {
    const sc = ru.sc;
    const T: u32 = @intCast(total);
    try ru.projection(d.qkv, sc.x.ptr.?, sc.qkv.ptr.?, total);
    try ru.projection(d.z, sc.x.ptr.?, sc.z.ptr.?, total);
    // a and b are both N = linear_heads (16) projections of the same x: one fused launch instead of two.
    try ru.projectionPair(d.a, sc.a.ptr.?, d.b, sc.b.ptr.?, sc.x.ptr.?, total);
    const w = ru.w;
    var base: usize = 0;
    for (segments) |seg| {
        const cache = &seg.cache.blocks[layer].delta;
        // The conv snapshot is written at this segment's absolute scratch row (its rows land at base..base+rows-1).
        const snap_conv: *anyopaque = sc.snapConv(layer, base);
        try ru.go(.gdn_pre, geo.gdnPre(@intCast(seg.rows), 3 * c.linear_heads), .{
            off(sc.qkv, base * c.conv_dim * 2), cache.conv.ptr, w.ptr(d.conv), @as(*anyopaque, @ptrCast(sc.windows().ptr)),
            off(sc.a, base * c.linear_heads * 2), off(sc.b, base * c.linear_heads * 2), w.ptr(d.a_log), w.ptr(d.dt_bias),
            off(sc.q, base * c.hidden * 2), off(sc.k, base * c.hidden * 2), off(sc.v, base * c.hidden * 2),
            off(sc.g, base * c.linear_heads * 4), off(sc.beta, base * c.linear_heads * 2), snap_conv,
        });
        // The recurrence is read into registers before it is written, one state row per work-item, so state_in == state_out is safe.
        // With `record`, the kernel saves every row to `layer`'s snapshot at this segment's absolute row; otherwise only
        // the final state, which it writes straight into `cache.recurrence` (state_in == state_out).
        const snap_rec: *anyopaque = if (record) sc.snapDelta(layer, base) else cache.recurrence.ptr.?;
        try ru.go(.gdn_chain, geo.gdnChain(c.linear_dim, c.linear_heads), .{
            off(sc.q, base * c.hidden * 2), off(sc.k, base * c.hidden * 2), off(sc.v, base * c.hidden * 2),
            off(sc.g, base * c.linear_heads * 4), off(sc.beta, base * c.linear_heads * 2), cache.recurrence.ptr,
            sc.dims([2]i32{ @intCast(seg.rows), @intFromBool(record) }), off(sc.y, base * c.hidden * 2), snap_rec,
        });
        // Without `record` the recurrence went straight into `cache.recurrence`; the conv tail is the segment's last row.
        if (!record) {
            if (ru.rec) |r| {
                try r.copy(cache.conv.ptr, sc.snapConv(layer, base + seg.rows - 1), xs.conv_bytes);
            } else {
                try ru.s.queue.copy(cache.conv.ptr, sc.snapConv(layer, base + seg.rows - 1), xs.conv_bytes, null);
            }
        }
        base += seg.rows;
    }
    try ru.go(.gdn_post, geo.gdnPost(T, c.linear_heads), .{ sc.y.ptr, sc.z.ptr, w.ptr(d.norm), c.eps, sc.mix.ptr });
    try ru.projection(d.out, sc.mix.ptr.?, sc.r.ptr.?, total);
}

fn attention(ru: Run, a: xw.Attention, layer: usize, segments: []const Segment, total: usize) Error!void {
    const sc = ru.sc;
    const w = ru.w;
    const T: u32 = @intCast(total);
    try ru.projection(a.q, sc.x.ptr.?, sc.qraw.ptr.?, total);
    // k and v are both N = kv_heads*head_dim (512) projections of the same x: one fused launch instead of two.
    try ru.projectionPair(a.k, sc.k.ptr.?, a.v, sc.v.ptr.?, sc.x.ptr.?, total);
    const qn = sc.dims([4]u32{ T, c.query_heads, 2 * c.hidden, 2 * c.head_dim });
    try ru.go(.head_norm, geo.layout(.head_norm, @intCast(total * c.query_heads), c.hidden, c.head_dim, c.query_heads, c.kv_heads), .{ sc.qraw.ptr, w.ptr(a.q_norm), sc.q.ptr, qn });
    const kn = sc.dims([4]u32{ T, c.kv_heads, c.kv_heads * c.head_dim, c.head_dim });
    try ru.go(.head_norm, geo.layout(.head_norm, @intCast(total * c.kv_heads), c.hidden, c.head_dim, c.query_heads, c.kv_heads), .{ sc.k.ptr, w.ptr(a.k_norm), sc.knorm.ptr, kn });
    // RoPE and the key/value cache rows are per segment: each writes at its own cache length, reading its own rows.
    var base: usize = 0;
    for (segments) |seg| {
        const cache = &seg.cache.blocks[layer].attention;
        const rows: u32 = @intCast(seg.rows);
        // dims[1] is the row stride of the shared `queries` buffer (all segments' rows), dims[2] this segment's base;
        // Metal passes the same pair. dims[1] = rows here would stride the buffer as if the segment were alone.
        try ru.go(.queries, geo.layout(.queries, rows, c.hidden, c.head_dim, c.query_heads, c.kv_heads), .{
            sc.q.ptr, sc.queries.ptr, sc.dims([4]u32{ @intCast(seg.cache.len), T, @intCast(base), 0 }),
        });
        try ru.go(.keys, geo.layout(.keys, rows, c.hidden, c.head_dim, c.query_heads, c.kv_heads), .{
            sc.knorm.ptr, sc.v.ptr, cache.keys.ptr, cache.values.ptr, sc.dims([4]u32{ @intCast(seg.cache.len), @intCast(seg.cache.capacity), @intCast(base), 0 }),
        });
        const nch: usize = (seg.cache.len + seg.rows + 127) / 128;
        const dims = sc.dims([8]i32{ @intCast(seg.cache.len), @intCast(seg.rows), @intCast(seg.cache.capacity), @intCast(nch), 0, @intCast(total), @intCast(base), 0 });
        try ru.go(.attn_partial, geo.attnPartial(@intCast(nch), rows, c.query_heads), .{
            sc.queries.ptr, cache.keys.ptr, cache.values.ptr, @as(f32, 1.0 / 16.0), dims, sc.pm.ptr, sc.pl.ptr, sc.po.ptr,
        });
        try ru.go(.attn_merge, geo.attnMerge(rows, c.query_heads), .{ sc.pm.ptr, sc.pl.ptr, sc.po.ptr, dims, off(sc.y, base * c.hidden * 2) });
        base += seg.rows;
    }
    try ru.go(.attention_gate, geo.layout(.attention_gate, T, c.hidden, c.head_dim, c.query_heads, c.kv_heads), .{ sc.y.ptr, sc.qraw.ptr, sc.mix.ptr });
    try ru.projection(a.out, sc.mix.ptr.?, sc.r.ptr.?, total);
}
