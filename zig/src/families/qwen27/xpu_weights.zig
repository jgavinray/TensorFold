//! Qwen 3.8 27B weights on an Intel B70 (Level Zero): the checkpoint's tensor bytes verbatim (MLX affine 4-bit, group 64),
//! packed into one device allocation at 256-byte alignment, with the Linear/Gdn/Attention/Layer structure and the declared
//! shapes and dtypes of the Metal path (gpu_weights.zig, widths.zig `shape`). This checkpoint is MULTI-SHARD: every shard
//! named by model.safetensors.index.json is opened and its `language_model.` tensors are read from wherever they live.
//! A host-side SHA-256 ledger of every tensor's bytes (taken from the files while loading) lets a device read-back be
//! checked exactly.
const std = @import("std");
const rt = @import("../../xpu/runtime.zig");
const sys = @import("../../xpu/sys.zig");
const st = @import("../../core/safetensors.zig");
const cfg = @import("config.zig");

pub const DType = st.DType;
pub const alignment = 256;
pub const chunk_bytes: usize = 64 << 20;
pub const max_shards = 32;
const Sha256 = std.crypto.hash.sha2.Sha256;

/// A tensor inside the weights arena: `offset` bytes from the arena's start (always a multiple of 256).
pub const Tensor = struct {
    offset: usize,
    bytes: usize,
    dtype: DType,
    shape: [st.max_rank]usize = @splat(1),
    rank: usize = 0,

    pub fn count(self: Tensor) usize {
        return self.bytes / self.dtype.size();
    }
};
pub const Linear = struct { weight: Tensor, scales: Tensor, biases: Tensor, inputs: usize, outputs: usize };
pub const Gdn = struct { qkv: Linear, z: Linear, b: Linear, a: Linear, out: Linear, conv: Tensor, norm: Tensor, a_log: Tensor, dt_bias: Tensor };
pub const Attention = struct { q: Linear, k: Linear, v: Linear, o: Linear, q_norm: Tensor, k_norm: Tensor };
pub const Mixer = union(cfg.Kind) { linear: Gdn, attention: Attention };
pub const Layer = struct {
    input_norm: Tensor,
    post_norm: Tensor,
    mlp: struct { gate: Linear, up: Linear, down: Linear },
    mixer: Mixer,
};

pub const LedgerEntry = struct { name: []const u8, tensor: Tensor, sha256: [Sha256.digest_length]u8 };

pub const Weights = struct {
    arena: rt.memory.Buffer,
    embed: Linear,
    /// This checkpoint always ships its own lm_head (`tie_word_embeddings: false`).
    head: Linear,
    norm: Tensor,
    layers: []Layer,
    /// Every language-model tensor, in arena order, with the hash of its host bytes.
    ledger: []LedgerEntry,
    total_bytes: usize,
    shards: usize,
    load_ns: u64,
    read_ns: u64 = 0,
    hash_ns: u64 = 0,
    upload_ns: u64 = 0,
    gpa: std.mem.Allocator,

    /// Device address of a tensor's first byte.
    pub fn ptr(self: *const Weights, t: Tensor) *anyopaque {
        return @ptrFromInt(@intFromPtr(self.arena.ptr.?) + t.offset);
    }

    pub fn deinit(self: *Weights) void {
        for (self.ledger) |e| self.gpa.free(e.name);
        self.gpa.free(self.ledger);
        self.gpa.free(self.layers);
        self.arena.free();
    }

    /// Reads every tensor back from the device and compares its SHA-256 with the ledger; returns the mismatch count.
    pub fn verify(self: *const Weights, s: *rt.session.Session, log: bool) !usize {
        var stage = try rt.memory.Buffer.alloc(&s.ctx, .host, chunk_bytes);
        defer stage.free();
        const buf = try stage.bytes();
        var bad: usize = 0;
        for (self.ledger) |e| {
            var h = Sha256.init(.{});
            var done: usize = 0;
            while (done < e.tensor.bytes) {
                const n = @min(chunk_bytes, e.tensor.bytes - done);
                try self.arena.download(&s.queue, e.tensor.offset + done, buf[0..n]);
                h.update(buf[0..n]);
                done += n;
            }
            var got: [Sha256.digest_length]u8 = undefined;
            h.final(&got);
            if (!std.mem.eql(u8, &got, &e.sha256)) {
                bad += 1;
                if (log) std.debug.print("MISMATCH {s} ({d} bytes)\n", .{ e.name, e.tensor.bytes });
            }
        }
        return bad;
    }
};

pub const LoadError = error{ UnexpectedQwenTensor, UnusedTensors, WeightsTooLarge, OpenFailed, ShortRead, OutOfMemory, BadSafetensors, BadShardIndex, TooManyShards, UnsupportedDType, RankTooHigh, Overflow } ||
    rt.Error || std.json.ParseError(std.json.Scanner);

const Shard = struct { fd: std.c.fd_t, data_start: usize, names: st.Header };

const Index = struct {
    names: [][]const u8,
    entries: []st.Entry,
    shard_of: []usize,
    placed: []Tensor,
    used: []bool,
    total: usize,

    fn find(self: *Index, name: []const u8) ?usize {
        for (self.names, 0..) |n, i| if (std.mem.eql(u8, n, name)) return i;
        return null;
    }

    fn expect(self: *Index, name: []const u8, dtype: DType, shape: []const usize) LoadError!Tensor {
        const i = self.find(name) orelse {
            std.debug.print("checkpoint has no tensor {s}\n", .{name});
            return error.UnexpectedQwenTensor;
        };
        const t = self.placed[i];
        if (t.dtype != dtype or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) {
            std.debug.print("{s}: unexpected Qwen tensor dtype or shape\n", .{name});
            return error.UnexpectedQwenTensor;
        }
        self.used[i] = true;
        return t;
    }

    fn tensor(self: *Index, prefix: []const u8, suffix: []const u8, dtype: DType, shape: []const usize) LoadError!Tensor {
        var buf: [256]u8 = undefined;
        return self.expect(std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix }) catch return error.UnexpectedQwenTensor, dtype, shape);
    }

    /// `module` is the fully-qualified leaf name; the affine triple is `<module>.weight/.scales/.biases`.
    fn projection(self: *Index, module: []const u8, outputs: usize, inputs: usize, group: usize) LoadError!Linear {
        return .{
            .weight = try self.tensor(module, ".weight", .u32, &.{ outputs, inputs / 8 }),
            .scales = try self.tensor(module, ".scales", .bf16, &.{ outputs, inputs / group }),
            .biases = try self.tensor(module, ".biases", .bf16, &.{ outputs, inputs / group }),
            .inputs = inputs,
            .outputs = outputs,
        };
    }
};

fn readAll(fd: std.c.fd_t, dest: []u8, at: usize) error{ShortRead}!void {
    var done: usize = 0;
    while (done < dest.len) {
        const n = std.c.pread(fd, dest.ptr + done, dest.len - done, @intCast(at + done));
        if (n <= 0) return error.ShortRead;
        done += @intCast(n);
    }
}

fn containsStr(haystack: []const []const u8, needle: []const u8) bool {
    for (haystack) |x| if (std.mem.eql(u8, x, needle)) return true;
    return false;
}

/// `<dir>/model.safetensors.index.json`'s parsed document; the caller reads `weight_map` from it.
fn readWeightMap(a: std.mem.Allocator, dir: []const u8) LoadError!std.json.Parsed(std.json.Value) {
    var pathbuf: [1024]u8 = undefined;
    const path = sys.zfmt(&pathbuf, "{s}/model.safetensors.index.json", .{dir}) orelse return error.OpenFailed;
    const text = sys.readFileAlloc(a, path, 64 << 20) orelse return error.OpenFailed;
    defer a.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    errdefer parsed.deinit();
    const root = switch (parsed.value) {
        .object => |o| o,
        else => return error.BadShardIndex,
    };
    _ = root.get("weight_map") orelse return error.BadShardIndex;
    return parsed;
}

/// Loads every `language_model.` tensor of a multi-shard MLX affine checkpoint onto the session's card.
/// `c` is config.json's parsed geometry; `group` is the checkpoint's affine group size (config.json's quantization).
pub fn load(gpa: std.mem.Allocator, s: *rt.session.Session, dir: []const u8, c: cfg.Config, group: usize) LoadError!Weights {
    const t0 = sys.nowNs();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    // 1. weight_map: name -> shard file, then open every named shard and index its language_model header.
    const index = try readWeightMap(a, dir);
    defer index.deinit();
    const map = index.value.object.get("weight_map").?.object;
    var shard_names = std.ArrayListUnmanaged([]const u8).empty;
    var shards: [max_shards]Shard = undefined;
    var n_shards: usize = 0;
    var it = map.iterator();
    while (it.next()) |kv| {
        const shard = switch (kv.value_ptr.*) {
            .string => |x| x,
            else => return error.BadShardIndex,
        };
        if (!containsStr(shard_names.items, shard)) try shard_names.append(a, shard);
    }
    if (shard_names.items.len == 0) return error.BadShardIndex;
    if (shard_names.items.len > max_shards) return error.TooManyShards;
    var opened: usize = 0;
    errdefer {
        for (shards[0..opened]) |sh| _ = std.c.close(sh.fd);
    }
    for (shard_names.items, 0..) |shard, i| {
        var pathbuf: [1024]u8 = undefined;
        const path = sys.zfmt(&pathbuf, "{s}/{s}", .{ dir, shard }) orelse return error.OpenFailed;
        const fd = std.c.open(path, .{ .ACCMODE = .RDONLY });
        if (fd < 0) {
            std.debug.print("cannot open shard {s}\n", .{path});
            return error.OpenFailed;
        }
        shards[i] = .{ .fd = fd, .data_start = 0, .names = .empty };
        opened += 1;
        var head8: [8]u8 = undefined;
        try readAll(fd, &head8, 0);
        const header_len: usize = @intCast(std.mem.readInt(u64, &head8, .little));
        const end = std.c.lseek(fd, 0, std.c.SEEK.END);
        if (end < 0 or header_len > @as(usize, @intCast(end)) - 8) return error.BadSafetensors;
        const data_start = 8 + header_len;
        const data_len: usize = @as(usize, @intCast(end)) - data_start;
        const header = try a.alloc(u8, header_len);
        try readAll(fd, header, 8);
        shards[i].data_start = data_start;
        shards[i].names = try st.parseHeaderPrefix(a, header, data_len, "language_model.");
        n_shards = i + 1;
    }
    defer {
        for (shards[0..n_shards]) |sh| _ = std.c.close(sh.fd);
    }

    // 2. One flat name table in a deterministic order (sorted), so the arena layout is reproducible run to run.
    var names = std.ArrayListUnmanaged([]const u8).empty;
    var entries = std.ArrayListUnmanaged(st.Entry).empty;
    var shard_of = std.ArrayListUnmanaged(usize).empty;
    for (shards[0..n_shards], 0..) |sh, si| for (sh.names.keys(), sh.names.values()) |name, e| {
        try names.append(a, name);
        try entries.append(a, e);
        try shard_of.append(a, si);
    };
    const order = try a.alloc(usize, names.items.len);
    for (order, 0..) |*o, i| o.* = i;
    const Ctx = struct {
        names: []const []const u8,
        fn lt(self: @This(), x: usize, y: usize) bool {
            return std.mem.lessThan(u8, self.names[x], self.names[y]);
        }
    };
    std.mem.sort(usize, order, Ctx{ .names = names.items }, Ctx.lt);

    var idx: Index = .{
        .names = try a.alloc([]const u8, order.len),
        .entries = try a.alloc(st.Entry, order.len),
        .shard_of = try a.alloc(usize, order.len),
        .placed = try a.alloc(Tensor, order.len),
        .used = try a.alloc(bool, order.len),
        .total = 0,
    };
    for (order, 0..) |src, dst| {
        idx.names[dst] = names.items[src];
        idx.entries[dst] = entries.items[src];
        idx.shard_of[dst] = shard_of.items[src];
    }
    @memset(idx.used, false);
    // Packing rule of the Metal split loader: each tensor aligned to 256 bytes, in arena order.
    var at: usize = 0;
    for (idx.entries, idx.placed) |e, *p| {
        at = std.mem.alignForward(usize, at, alignment);
        p.* = .{ .offset = at, .bytes = e.end - e.begin, .dtype = e.dtype, .shape = e.shape, .rank = e.rank };
        at += e.end - e.begin;
    }
    idx.total = at;

    // 3. Structure: exactly the geometry and formats the Metal path declares (widths.zig `shape`, gpu_weights.zig dtypes).
    if (c.v_heads % c.k_heads != 0 or c.heads % c.kv_heads != 0) return error.UnexpectedQwenTensor;
    const g = group;
    var w: Weights = .{
        .arena = undefined,
        .embed = try idx.projection("language_model.model.embed_tokens", c.vocab, c.hidden, g),
        .head = try idx.projection("language_model.lm_head", c.vocab, c.hidden, g),
        .norm = try idx.tensor("language_model.model.", "norm.weight", .bf16, &.{c.hidden}),
        .layers = try a.alloc(Layer, c.layers),
        .ledger = &.{},
        .total_bytes = idx.total,
        .shards = n_shards,
        .load_ns = 0,
        .gpa = gpa,
    };
    for (w.layers, 0..) |*layer, i| {
        var buf: [128]u8 = undefined;
        const p = std.fmt.bufPrint(&buf, "language_model.model.layers.{d}.", .{i}) catch unreachable;
        const mixer: Mixer = switch (c.kind(i)) {
            .linear => .{ .linear = .{
                .qkv = try idx.projection(try nameJoin(a, p, "linear_attn.in_proj_qkv"), c.gdnQkvDim(), c.hidden, g),
                .z = try idx.projection(try nameJoin(a, p, "linear_attn.in_proj_z"), c.gdnValueDim(), c.hidden, g),
                .b = try idx.projection(try nameJoin(a, p, "linear_attn.in_proj_b"), c.v_heads, c.hidden, g),
                .a = try idx.projection(try nameJoin(a, p, "linear_attn.in_proj_a"), c.v_heads, c.hidden, g),
                .out = try idx.projection(try nameJoin(a, p, "linear_attn.out_proj"), c.hidden, c.gdnValueDim(), g),
                .conv = try idx.tensor(p, "linear_attn.conv1d.weight", .bf16, &.{ c.gdnQkvDim(), c.conv_kernel, 1 }),
                .norm = try idx.tensor(p, "linear_attn.norm.weight", .bf16, &.{c.dv}),
                .a_log = try idx.tensor(p, "linear_attn.A_log", .bf16, &.{c.v_heads}),
                .dt_bias = try idx.tensor(p, "linear_attn.dt_bias", .bf16, &.{c.v_heads}),
            } },
            .attention => .{ .attention = .{
                .q = try idx.projection(try nameJoin(a, p, "self_attn.q_proj"), c.qDim(), c.hidden, g),
                .k = try idx.projection(try nameJoin(a, p, "self_attn.k_proj"), c.kvDim(), c.hidden, g),
                .v = try idx.projection(try nameJoin(a, p, "self_attn.v_proj"), c.kvDim(), c.hidden, g),
                .o = try idx.projection(try nameJoin(a, p, "self_attn.o_proj"), c.hidden, c.heads * c.head_dim, g),
                .q_norm = try idx.tensor(p, "self_attn.q_norm.weight", .bf16, &.{c.head_dim}),
                .k_norm = try idx.tensor(p, "self_attn.k_norm.weight", .bf16, &.{c.head_dim}),
            } },
        };
        layer.* = .{
            .input_norm = try idx.tensor(p, "input_layernorm.weight", .bf16, &.{c.hidden}),
            .post_norm = try idx.tensor(p, "post_attention_layernorm.weight", .bf16, &.{c.hidden}),
            .mlp = .{
                .gate = try idx.projection(try nameJoin(a, p, "mlp.gate_proj"), c.intermediate, c.hidden, g),
                .up = try idx.projection(try nameJoin(a, p, "mlp.up_proj"), c.intermediate, c.hidden, g),
                .down = try idx.projection(try nameJoin(a, p, "mlp.down_proj"), c.hidden, c.intermediate, g),
            },
            .mixer = mixer,
        };
    }
    var left: usize = 0;
    for (idx.used, idx.names) |u, n| if (!u) {
        if (left < 5) std.debug.print("unused checkpoint tensor {s}\n", .{n});
        left += 1;
    };
    if (left != 0) return error.UnusedTensors;

    // 4. Device arena, then stream every tensor through one pinned staging buffer: read, hash on the host, copy up.
    w.arena = try rt.memory.Buffer.alloc(&s.ctx, .device, @max(idx.total, 16));
    errdefer w.arena.free();
    var stage = try rt.memory.Buffer.alloc(&s.ctx, .host, chunk_bytes);
    defer stage.free();
    const sbuf = try stage.bytes();
    const ledger = try gpa.alloc(LedgerEntry, idx.entries.len);
    var made: usize = 0;
    errdefer {
        for (ledger[0..made]) |e| gpa.free(e.name);
        gpa.free(ledger);
    }
    for (idx.entries, idx.placed, idx.names, idx.shard_of, ledger) |e, p, name, si, *slot| {
        const sh = shards[si];
        var h = Sha256.init(.{});
        var done: usize = 0;
        while (done < p.bytes) {
            const n = @min(chunk_bytes, p.bytes - done);
            const t1 = sys.nowNs();
            try readAll(sh.fd, sbuf[0..n], sh.data_start + e.begin + done);
            const t2 = sys.nowNs();
            h.update(sbuf[0..n]);
            const t3 = sys.nowNs();
            try w.arena.upload(&s.queue, p.offset + done, sbuf[0..n]);
            const t4 = sys.nowNs();
            w.read_ns += t2 - t1;
            w.hash_ns += t3 - t2;
            w.upload_ns += t4 - t3;
            done += n;
        }
        slot.* = .{ .name = try gpa.dupe(u8, name), .tensor = p, .sha256 = undefined };
        made += 1;
        h.final(&slot.sha256);
    }
    w.ledger = ledger;
    w.load_ns = sys.nowNs() - t0;
    return w;
}

fn nameJoin(a: std.mem.Allocator, prefix: []const u8, leaf: []const u8) LoadError![]const u8 {
    return std.mem.concat(a, u8, &.{ prefix, leaf });
}

test "arena packing aligns every tensor and keeps the raw bytes contiguous" {
    // Two tensors of 3 and 5 bytes: the second starts at 256, and the arena is their aligned total.
    const a = std.testing.allocator;
    const names = [_][]const u8{ "a", "b" };
    const entries = [_]st.Entry{
        .{ .dtype = .u8, .rank = 1, .shape = .{ 3, 1, 1, 1, 1 }, .begin = 0, .end = 3 },
        .{ .dtype = .u8, .rank = 1, .shape = .{ 5, 1, 1, 1, 1 }, .begin = 3, .end = 8 },
    };
    var placed: [2]Tensor = undefined;
    var at: usize = 0;
    for (entries, &placed) |e, *p| {
        at = std.mem.alignForward(usize, at, alignment);
        p.* = .{ .offset = at, .bytes = e.end - e.begin, .dtype = e.dtype, .shape = e.shape, .rank = e.rank };
        at += e.end - e.begin;
    }
    try std.testing.expectEqual(@as(usize, 0), placed[0].offset);
    try std.testing.expectEqual(@as(usize, 256), placed[1].offset);
    try std.testing.expectEqual(@as(usize, 261), at);
    _ = a;
    _ = names;
}
