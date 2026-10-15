//! Qwen3.5-2B weights on an Intel B70 (Level Zero): the checkpoint's tensor bytes verbatim (MLX affine 4-bit, group 64),
//! packed into one device allocation at 256-byte alignment exactly as the Metal path's split loader packs its buffer
//! (core/checkpoint_metal.zig), with the same Linear/Delta/Attention/Layer structure as weights.zig. A host-side SHA-256
//! ledger of every tensor's bytes (taken from the file while loading) lets a device read-back be checked exactly.
const std = @import("std");
const rt = @import("../../xpu/runtime.zig");
const sys = @import("../../xpu/sys.zig");
const st = @import("core").safetensors;
const c = @import("config.zig");

pub const DType = st.DType;
pub const alignment = 256;
pub const chunk_bytes: usize = 64 << 20;
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
pub const Delta = struct { qkv: Linear, z: Linear, a: Linear, b: Linear, out: Linear, conv: Tensor, norm: Tensor, a_log: Tensor, dt_bias: Tensor };
pub const Attention = struct { q: Linear, k: Linear, v: Linear, out: Linear, q_norm: Tensor, k_norm: Tensor };
pub const Layer = struct {
    input_norm: Tensor,
    post_norm: Tensor,
    gate: Linear,
    up: Linear,
    down: Linear,
    mixer: union(enum) { delta: Delta, attention: Attention },
};

pub const LedgerEntry = struct { name: []const u8, tensor: Tensor, sha256: [Sha256.digest_length]u8 };

pub const Weights = struct {
    arena: rt.memory.Buffer,
    embedding: Linear,
    /// Present only when the checkpoint ships its own lm_head; otherwise the head is the packed embedding.
    separate_head: ?Linear,
    norm: Tensor,
    blocks: [c.layers]Layer,
    /// Every tensor of the language model, in arena order, with the hash of its host bytes.
    ledger: []LedgerEntry,
    total_bytes: usize,
    load_ns: u64,
    read_ns: u64 = 0,
    hash_ns: u64 = 0,
    upload_ns: u64 = 0,
    gpa: std.mem.Allocator,

    pub fn head(self: *const Weights) Linear {
        return self.separate_head orelse self.embedding;
    }

    /// Device address of a tensor's first byte.
    pub fn ptr(self: *const Weights, t: Tensor) *anyopaque {
        return @ptrFromInt(@intFromPtr(self.arena.ptr.?) + t.offset);
    }

    pub fn deinit(self: *Weights) void {
        for (self.ledger) |e| self.gpa.free(e.name);
        self.gpa.free(self.ledger);
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

pub const LoadError = error{ UnexpectedQwenTensor, QwenHeadTiednessMismatch, UnusedTensors, WeightsTooLarge, OpenFailed, ShortRead, OutOfMemory, BadSafetensors } ||
    rt.Error || std.json.ParseError(std.json.Scanner) || error{ UnsupportedDType, RankTooHigh, Overflow, SyntaxError, UnexpectedEndOfInput };

const Index = struct {
    names: []const []const u8,
    entries: []const st.Entry,
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

    fn projection(self: *Index, prefix: []const u8, suffix: []const u8, outputs: usize, inputs: usize) LoadError!Linear {
        var buf: [256]u8 = undefined;
        const name = std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix }) catch return error.UnexpectedQwenTensor;
        return .{
            .weight = try self.tensor(name, ".weight", .u32, &.{ outputs, inputs / 8 }),
            .scales = try self.tensor(name, ".scales", .bf16, &.{ outputs, inputs / c.group }),
            .biases = try self.tensor(name, ".biases", .bf16, &.{ outputs, inputs / c.group }),
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

/// Loads every `language_model.` tensor of `<dir>/model.safetensors` (single shard) onto the session's card.
/// `tied` is config.json's tie_word_embeddings.
pub fn load(gpa: std.mem.Allocator, s: *rt.session.Session, dir: []const u8, tied: bool) LoadError!Weights {
    const t0 = sys.nowNs();
    var pathbuf: [1024]u8 = undefined;
    const path = sys.zfmt(&pathbuf, "{s}/model.safetensors", .{dir}) orelse return error.OpenFailed;
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY });
    if (fd < 0) {
        std.debug.print("cannot open {s}\n", .{path});
        return error.OpenFailed;
    }
    defer _ = std.c.close(fd);
    var head8: [8]u8 = undefined;
    try readAll(fd, &head8, 0);
    const header_len: usize = @intCast(std.mem.readInt(u64, &head8, .little));
    const end = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (end < 0 or header_len > @as(usize, @intCast(end)) - 8) return error.BadSafetensors;
    const data_start = 8 + header_len;
    const data_len: usize = @as(usize, @intCast(end)) - data_start;

    var scratch = std.heap.ArenaAllocator.init(gpa);
    defer scratch.deinit();
    const a = scratch.allocator();
    const header = try a.alloc(u8, header_len);
    try readAll(fd, header, 8);
    const parsed = try st.parseHeaderPrefix(a, header, data_len, "language_model.");

    var idx: Index = .{
        .names = parsed.keys(),
        .entries = parsed.values(),
        .placed = try a.alloc(Tensor, parsed.count()),
        .used = try a.alloc(bool, parsed.count()),
        .total = 0,
    };
    @memset(idx.used, false);
    // Same packing rule as the Metal split loader: header order, each tensor aligned to 256 bytes.
    var at: usize = 0;
    for (idx.entries, idx.placed) |e, *p| {
        at = std.mem.alignForward(usize, at, alignment);
        p.* = .{ .offset = at, .bytes = e.end - e.begin, .dtype = e.dtype, .shape = e.shape, .rank = e.rank };
        at += e.end - e.begin;
    }
    idx.total = at;

    // Structure (and the refusal of unexpected dtypes, shapes, leftovers and head tiedness), as weights.zig.
    const ships_head = idx.find("language_model.lm_head.weight") != null;
    if (ships_head != !tied) {
        std.debug.print("config.json says tie_word_embeddings={} but the checkpoint {s} a language_model.lm_head projection\n", .{ tied, if (ships_head) "ships" else "lacks" });
        return error.QwenHeadTiednessMismatch;
    }
    var w: Weights = .{
        .arena = undefined,
        .embedding = try idx.projection("language_model.model.", "embed_tokens", c.vocab, c.hidden),
        .separate_head = if (ships_head) try idx.projection("language_model.", "lm_head", c.vocab, c.hidden) else null,
        .norm = try idx.tensor("language_model.model.", "norm.weight", .bf16, &.{c.hidden}),
        .blocks = undefined,
        .ledger = &.{},
        .total_bytes = idx.total,
        .load_ns = 0,
        .gpa = gpa,
    };
    for (&w.blocks, 0..) |*block, i| {
        var buf: [128]u8 = undefined;
        const p = std.fmt.bufPrint(&buf, "language_model.model.layers.{d}.", .{i}) catch unreachable;
        block.* = .{
            .input_norm = try idx.tensor(p, "input_layernorm.weight", .bf16, &.{c.hidden}),
            .post_norm = try idx.tensor(p, "post_attention_layernorm.weight", .bf16, &.{c.hidden}),
            .gate = try idx.projection(p, "mlp.gate_proj", c.intermediate, c.hidden),
            .up = try idx.projection(p, "mlp.up_proj", c.intermediate, c.hidden),
            .down = try idx.projection(p, "mlp.down_proj", c.hidden, c.intermediate),
            .mixer = undefined,
        };
        if (c.linear(i)) {
            block.mixer = .{ .delta = .{
                .qkv = try idx.projection(p, "linear_attn.in_proj_qkv", c.conv_dim, c.hidden),
                .z = try idx.projection(p, "linear_attn.in_proj_z", c.hidden, c.hidden),
                .a = try idx.projection(p, "linear_attn.in_proj_a", c.linear_heads, c.hidden),
                .b = try idx.projection(p, "linear_attn.in_proj_b", c.linear_heads, c.hidden),
                .out = try idx.projection(p, "linear_attn.out_proj", c.hidden, c.hidden),
                .conv = try idx.tensor(p, "linear_attn.conv1d.weight", .bf16, &.{ c.conv_dim, c.conv_taps, 1 }),
                .norm = try idx.tensor(p, "linear_attn.norm.weight", .bf16, &.{c.linear_dim}),
                .a_log = try idx.tensor(p, "linear_attn.A_log", .f32, &.{c.linear_heads}),
                .dt_bias = try idx.tensor(p, "linear_attn.dt_bias", .bf16, &.{c.linear_heads}),
            } };
        } else {
            block.mixer = .{ .attention = .{
                .q = try idx.projection(p, "self_attn.q_proj", 2 * c.hidden, c.hidden),
                .k = try idx.projection(p, "self_attn.k_proj", c.kv_heads * c.head_dim, c.hidden),
                .v = try idx.projection(p, "self_attn.v_proj", c.kv_heads * c.head_dim, c.hidden),
                .out = try idx.projection(p, "self_attn.o_proj", c.hidden, c.hidden),
                .q_norm = try idx.tensor(p, "self_attn.q_norm.weight", .bf16, &.{c.head_dim}),
                .k_norm = try idx.tensor(p, "self_attn.k_norm.weight", .bf16, &.{c.head_dim}),
            } };
        }
    }
    var left: usize = 0;
    for (idx.used, idx.names) |u, n| if (!u) {
        if (left < 5) std.debug.print("unused checkpoint tensor {s}\n", .{n});
        left += 1;
    };
    if (left != 0) return error.UnusedTensors;

    // Device arena, then stream every tensor through one pinned staging buffer: read, hash on the host, copy up.
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
    for (idx.entries, idx.placed, idx.names, ledger) |e, p, name, *slot| {
        var h = Sha256.init(.{});
        var done: usize = 0;
        while (done < p.bytes) {
            const n = @min(chunk_bytes, p.bytes - done);
            const t1 = sys.nowNs();
            try readAll(fd, sbuf[0..n], data_start + e.begin + done);
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
