//! Step 5 host gate (eaq-halfstep-spec-fable.md section 4, "Host oracle parity"):
//!
//!   T-HS-0  the unified E(p) = ((p + 1) * halves) >> 1 equals exl3_format.streamEnds at
//!           every p and every halves 2..16; the differenced b(p) is halves/2 at whole
//!           widths and ka + (p & 1) at odd ones; sum b = 128 * halves.
//!   T-HS-1  the oracle's lane-window test (exl3_format.zig:294-362) with the
//!           odd-halves `continue` removed and a_v generalized per spec 1.2 — both the
//!           floor form and the t1 av form — over every oracle tile (the fixture does
//!           carry odd-halves records: halves 3, 5, 7, and every even width 2..16) and
//!           over tiles generated from the step-1 producer's packStates of random
//!           closed sequences at every width 2..16, so the odd widths 9, 11, 13, 15
//!           the fixture lacks are pinned too.
//!   T-HS-2  the exl3_dequant128_c phase-A transcription against exl3_format.unpack:
//!           bit-identical fp16 on every tile at halves 2..16 (oracle tiles where the
//!           fixture has them, producer tiles otherwise, plus encodeTile-of-Gaussian
//!           tiles at every odd width).
//!   T-HS-3  the exl3_decode2_h / exl3_decode2_t1_h transcriptions on a [256, 256] part
//!           with random fp16 X: the extracted states equal exl3_format.tileStates
//!           exactly (the real assertion), the t1 output equals the general kernel's at
//!           T = 1 bit for bit, and the fp16 Y sits inside the max-ulp envelope of the
//!           fp64 reference that the same run measures at halves 4 and 6 — asserted at
//!           the odd widths 3, 5, 7, 9, 11, 13, 15.
//!   T-HS-6 host  the even-path identity (spec 1.2/section 6): at every even halves the
//!           generalized offset equals the pre-step-5 (v + 1) * bits template bit for
//!           bit, the host twin of the device golden half.
//!
//! Run: `zig build test-exl3-halfstep` (cwd the repo root for the oracle fixture).

const std = @import("std");
const core = @import("core");
const hs = core.exl3_halfstep;
const exl3_format = core.exl3_format;

const Width = exl3_format.Width;
const gpa = std.testing.allocator;

const oracle_path = "zig/tests/fixtures/exl3_oracle.bin";

// ---------------------------------------------------------------------------
// oracle fixture reader (the exl3_format.zig:217-244 layout, re-read here so the
// frozen oracle stays untouched)

const Fixture = struct {
    arena: std.heap.ArenaAllocator,
    bytes: []const u8,

    fn load() !Fixture {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, oracle_path, arena.allocator(), .limited(1 << 22));
        if (!std.mem.eql(u8, bytes[0..8], "EXL3ORAC")) return error.BadOracle;
        return .{ .arena = arena, .bytes = bytes };
    }

    fn deinit(f: *Fixture) void {
        f.arena.deinit();
    }

    fn count(f: Fixture) u64 {
        return std.mem.readInt(u64, f.bytes[8..16], .little);
    }

    const Record = struct {
        halves: u16,
        kt: u32,
        nt: u32,
        trellis: []const u8, // all tiles' words, k-major
        next: usize,
    };

    fn record(f: Fixture, start: usize) Record {
        const halves = std.mem.readInt(u16, f.bytes[start..][0..2], .little);
        const kt = std.mem.readInt(u32, f.bytes[start + 4 ..][0..4], .little);
        const nt = std.mem.readInt(u32, f.bytes[start + 8 ..][0..4], .little);
        const w = Width{ .halves = halves };
        const words = w.tileWords();
        const tbytes = kt * nt * words * 2;
        const sbytes = kt * nt * 256 * 4;
        return .{
            .halves = halves,
            .kt = kt,
            .nt = nt,
            .trellis = f.bytes[start + 12 ..][0..tbytes],
            .next = start + 12 + tbytes + sbytes,
        };
    }
};

/// Iterate the fixture's records; body sees each record plus a caller context.
fn forEachRecord(f: Fixture, ctx: anytype, comptime body: fn (@TypeOf(ctx), Fixture.Record) anyerror!void) !void {
    var at: usize = 16;
    for (0..f.count()) |_| {
        const rec = f.record(at);
        try body(ctx, rec);
        at = rec.next;
    }
}

fn tileBytes(rec: Fixture.Record, ti: usize, out: []i16) void {
    const halves: u32 = rec.halves;
    const words = @as(usize, halves) * 8;
    std.debug.assert(out.len == words);
    @memcpy(std.mem.sliceAsBytes(out), rec.trellis[ti * words * 2 ..][0 .. words * 2]);
}

// ---------------------------------------------------------------------------
// T-HS-0

test "T-HS-0: the unified E(p), b(p) and the a_v split hold at every halves 2..16" {
    for (2..17) |hh| {
        const halves: u32 = @intCast(hh);
        const w = Width{ .halves = @intCast(hh) };
        var ends: [256]u32 = undefined;
        exl3_format.streamEnds(w, &ends);
        var sum: u32 = 0;
        for (0..256) |p| {
            // the one closed form the spec 1.2 claims for both parities
            try std.testing.expectEqual(hs.streamEnd(halves, @intCast(p)), ends[p]);
            const prev: u32 = if (p == 0) 0 else ends[p - 1];
            const b = ends[p] - prev;
            if (hh % 2 == 0) {
                try std.testing.expectEqual(@as(u32, @intCast(hh / 2)), b);
            } else {
                const ka: u32 = @intCast((hh - 1) / 2);
                try std.testing.expectEqual(ka + (@as(u32, @intCast(p)) & 1), b);
            }
            sum += b;
        }
        try std.testing.expectEqual(@as(u32, 128) * halves, sum); // one tile's fresh bits
        try std.testing.expectEqual(@as(u32, 128) * halves, ends[255]); // E(255) = R
        // the a_v split: windowStart(halves, lane, v) == E(16*lane + v) - 16, and the
        // bounds that keep the kernels' wrap logic valid (spec 1.2's bullets)
        for ([_]u32{ 0, 3, 7, 15 }) |lane| {
            for ([_]u32{ 0, 1, 8, 15 }) |v| {
                const p = 16 * lane + v;
                try std.testing.expectEqual(@as(i32, @intCast(ends[p])) - 16, hs.windowStart(halves, lane, v));
            }
        }
        try std.testing.expectEqual(@as(i32, @intCast(halves >> 1)) - 16, hs.windowStart(halves, 0, 0)); // min
        try std.testing.expectEqual(@as(i32, @intCast(128 * halves)) - 16, hs.windowStart(halves, 15, 15)); // max
        try std.testing.expect(hs.windowStart(halves, 0, 0) >= -15); // floor form yields u >= -1
    }
}

// ---------------------------------------------------------------------------
// T-HS-6 host half: the even-path identity (spec 1.2, section 6). The template
// before step 5 computed lo = (16*lane + 1) * bits - 16 and a_v = lo + v * bits;
// the kernels now carry lb = 8*lane*halves - 16 and a_v = lb + (((v+1)*halves)>>1).
// At even halves (bits = halves/2) those must agree at every position — the
// claim that the shipped even path never moved, provable on any host. The device
// half (golden Y/slab capture and replay, halfstep_test.zig) is the hyper03 gate
// and the trap for a launch site still passing bits.

test "T-HS-6 host: every even halves reduces bit-for-bit to the (v + 1) * bits template" {
    for ([_]u32{ 2, 4, 6, 8, 10, 12, 14, 16 }) |halves| {
        const bits = halves / 2;
        try std.testing.expectEqual(@as(u32, 8) * bits, hs.ringWords(halves)); // nw = 4*halves = 8*bits
        try std.testing.expectEqual(@as(u32, 256) * bits, hs.ringBits(halves)); // R = 128*halves = 256*bits
        try std.testing.expectEqual(@as(u32, 16) * bits, @as(u32, halves) * 8); // tile stride in int16 words
        for (0..16) |lane_u| {
            const lane: i32 = @intCast(lane_u);
            const b: i32 = @intCast(bits);
            for (0..16) |v_u| {
                const v: i32 = @intCast(v_u);
                const legacy = (16 * lane + 1) * b - 16 + v * b;
                try std.testing.expectEqual(legacy, hs.windowStart(halves, @intCast(lane_u), @intCast(v_u)));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// T-HS-1: the oracle's lane-window test, generalized (the spec's removal of
// `if (rec.halves % 2 != 0) continue`), plus the packStates-of-closed-sequence
// sources at every width 2..16 (the fixture has no odd records beyond halves 7).

fn checkTileWindows(ring: []const u32, halves: u32, want: *const [256]u32) !void {
    for (0..16) |lane| {
        for (0..16) |v| {
            const p = 16 * lane + v;
            // floor form (exl3_decode2_h / the dequant's phase A)
            try std.testing.expectEqual(want[p], hs.stateFloor(ring, halves, @intCast(lane), @intCast(v)));
            // t1 av form (exl3_decode2_t1_h)
            try std.testing.expectEqual(want[p], hs.stateT1(ring, halves, @intCast(lane), @intCast(v)));
        }
    }
}

fn checkRecordWindows(ctx: *usize, rec: Fixture.Record) !void {
    const halves: u32 = rec.halves;
    const words = @as(usize, halves) * 8;
    const nw = hs.ringWords(halves);
    for (0..rec.kt * rec.nt) |ti| {
        var tile: [128]i16 = undefined;
        tileBytes(rec, ti, tile[0..words]);
        var states: [256]u32 = undefined;
        exl3_format.tileStates(.{ .halves = rec.halves }, tile[0..words], &states);
        var ring: [64]u32 = undefined;
        hs.ringOf(tile[0..words], ring[0..nw]);
        try checkTileWindows(ring[0..nw], halves, &states);
        ctx.* += 1;
    }
}

test "T-HS-1: both kernel window forms reproduce tileStates at every width, oracle and producer tiles" {
    var f = try Fixture.load();
    defer f.deinit();
    var oracle_tiles: usize = 0;
    try forEachRecord(f, &oracle_tiles, checkRecordWindows);
    try std.testing.expect(oracle_tiles >= 162); // the exl3_format oracle run covers 162 tiles

    // producer tiles: packStates of random closed sequences, every halves 2..16
    for (2..17) |hh| {
        const halves: u32 = @intCast(hh);
        var prng = std.Random.DefaultPrng.init(0x4853 + halves);
        const tiles: usize = 8;
        const words = @as(usize, halves) * 8;
        const tr = try gpa.alloc(i16, tiles * words);
        defer gpa.free(tr);
        hs.fillClosedTiles(prng.random(), halves, tiles, tr);
        const nw = hs.ringWords(halves);
        for (0..tiles) |ti| {
            const tile = tr[ti * words ..][0..words];
            var states: [256]u32 = undefined;
            exl3_format.tileStates(.{ .halves = @intCast(halves) }, tile, &states);
            var ring: [64]u32 = undefined;
            hs.ringOf(tile, ring[0..nw]);
            try checkTileWindows(ring[0..nw], halves, &states);
        }
    }
}

// ---------------------------------------------------------------------------
// T-HS-2: the dequant phase-A transcription vs the oracle's unpack.

fn checkDequant(tr: []const i16, halves: u32, tiles_k: u32, tiles_n: u32) !void {
    const words = @as(usize, halves) * 8;
    std.debug.assert(tr.len == @as(usize, tiles_k) * tiles_n * words);
    const k = tiles_k * 16;
    const n = tiles_n * 16;
    const got = try gpa.alloc(f16, k * n);
    defer gpa.free(got);
    const want = try gpa.alloc(f16, k * n);
    defer gpa.free(want);
    hs.dequantTranscribe(tr, halves, tiles_k, tiles_n, got);
    exl3_format.unpack(.{ .halves = @intCast(halves) }, .mul1, tr, tiles_k, tiles_n, want);
    for (got, want) |g, w| try std.testing.expectEqual(@as(u16, @bitCast(w)), @as(u16, @bitCast(g)));
}

fn checkRecordDequant(ctx: *usize, rec: Fixture.Record) !void {
    const halves: u32 = rec.halves;
    const words = @as(usize, halves) * 8;
    const tr = try gpa.alloc(i16, @as(usize, rec.kt) * rec.nt * words);
    defer gpa.free(tr);
    for (0..@as(usize, rec.kt) * rec.nt) |ti| tileBytes(rec, ti, tr[ti * words ..][0..words]);
    try checkDequant(tr, halves, rec.kt, rec.nt);
    ctx.* += 1;
}

test "T-HS-2: the dequant phase-A transcription is bit-identical to unpack at every halves 2..16" {
    // oracle tiles at every width the fixture carries
    var f = try Fixture.load();
    defer f.deinit();
    var oracle_records: usize = 0;
    try forEachRecord(f, &oracle_records, checkRecordDequant);
    try std.testing.expectEqual(@as(usize, 27), oracle_records);

    // producer tiles at the whole suite of widths 2..16: closed-sequence packs and,
    // at every odd width, tiles through the step-1 encoder's encodeTile of Gaussians
    for (2..17) |hh| {
        const halves: u32 = @intCast(hh);
        const tiles_k: u32 = 2;
        const tiles_n: u32 = 3;
        const words = @as(usize, halves) * 8;
        const count = @as(usize, tiles_k) * tiles_n;

        var prng = std.Random.DefaultPrng.init(0xDE00 + halves);
        const closed = try gpa.alloc(i16, count * words);
        defer gpa.free(closed);
        hs.fillClosedTiles(prng.random(), halves, count, closed);
        try checkDequant(closed, halves, tiles_k, tiles_n);

        if (halves % 2 == 1) {
            var prng2 = std.Random.DefaultPrng.init(0xC0DE * halves);
            const enc = try gpa.alloc(i16, count * words);
            defer gpa.free(enc);
            try hs.fillEncodedTiles(prng2.random(), gpa, halves, count, enc);
            try checkDequant(enc, halves, tiles_k, tiles_n);
        }
    }
}

// ---------------------------------------------------------------------------
// T-HS-3: the GEMV transcriptions on a [256, 256] part.

const K3: u32 = 256;
const N3: u32 = 256;
const TK3: u32 = K3 / 16;
const TN3: u32 = N3 / 16;

/// One seeded part: closed-sequence tiles, the expected states, and random fp16 X.
const Part3 = struct {
    tr: []i16,
    states: [TK3 * TN3 * 256]u32,
    x: [8 * K3]f16,

    fn init(seed: u64, halves: u32) !Part3 {
        var self: Part3 = undefined;
        const words = @as(usize, halves) * 8;
        self.tr = try gpa.alloc(i16, TK3 * TN3 * words);
        errdefer gpa.free(self.tr);
        var prng = std.Random.DefaultPrng.init(seed);
        hs.fillClosedTiles(prng.random(), halves, TK3 * TN3, self.tr);
        for (0..TK3 * TN3) |ti| {
            var s: [256]u32 = undefined;
            exl3_format.tileStates(.{ .halves = @intCast(halves) }, self.tr[ti * words ..][0..words], &s);
            @memcpy(self.states[ti * 256 ..][0..256], &s);
        }
        hs.fillRandomX(prng.random(), &self.x);
        return self;
    }

    fn deinit(self: *Part3) void {
        gpa.free(self.tr);
    }
};

fn measureUlp(part: *Part3, halves: u32, rows: u32) !u32 {
    const wf = try gpa.alloc(f64, K3 * N3);
    defer gpa.free(wf);
    hs.fastWeightMatrix(part.tr, halves, TK3, TN3, wf);
    const yref = try gpa.alloc(f64, rows * N3);
    defer gpa.free(yref);
    hs.gemvReference(part.x[0 .. rows * K3], rows, wf, K3, N3, yref);
    const y = try gpa.alloc(f16, rows * N3);
    defer gpa.free(y);
    const states = try gpa.alloc(u32, TK3 * TN3 * 256);
    defer gpa.free(states);
    hs.gemvTranscribe(part.tr, halves, K3, N3, N3, part.x[0 .. rows * K3], rows, y, states);
    // the real assertion: the extraction reproduces tileStates exactly
    try std.testing.expectEqualSlices(u32, &part.states, states);
    return hs.gemvUlpBound(y, N3, yref, rows, N3);
}

test "T-HS-3: the GEMV transcriptions reproduce tileStates and sit inside the even-half ulp envelope" {
    // the envelope: the same transcription's worst fp16-ulp distance at halves 4 and 6
    var bound: u32 = 0;
    for ([_]u32{ 4, 6 }) |halves| {
        var part = try Part3.init(0x9E37 + halves, halves);
        defer part.deinit();
        for (1..9) |rows_u| {
            bound = @max(bound, try measureUlp(&part, halves, @intCast(rows_u)));
        }
    }
    std.debug.print("T-HS-3: even-half envelope {d} ulp; asserting odd widths within it\n", .{bound});
    try std.testing.expect(bound < 32); // a broken transcription or reference is not ulps away

    for ([_]u32{ 3, 5, 7, 9, 11, 13, 15 }) |halves| {
        var part = try Part3.init(0x9E37 + halves, halves);
        defer part.deinit();
        for (1..9) |rows_u| {
            const rows: u32 = @intCast(rows_u);
            const worst = try measureUlp(&part, halves, rows);
            std.debug.print("T-HS-3: halves={d} rows={d} worst={d} ulp (bound {d})\n", .{ halves, rows, worst, bound });
            try std.testing.expect(worst <= bound);
        }
        // the t1 fast path equals the general kernel's row 0 bit for bit
        const y1 = try gpa.alloc(f16, N3);
        defer gpa.free(y1);
        hs.gemvT1Transcribe(part.tr, halves, K3, N3, N3, part.x[0..K3], y1);
        const y8 = try gpa.alloc(f16, N3);
        defer gpa.free(y8);
        hs.gemvTranscribe(part.tr, halves, K3, N3, N3, part.x[0..K3], 1, y8, null);
        for (y1, y8) |a, b| try std.testing.expectEqual(@as(u16, @bitCast(a)), @as(u16, @bitCast(b)));

        // the t1 extraction form agrees with the general one at every position
        const words = @as(usize, halves) * 8;
        const nw = hs.ringWords(halves);
        for (0..TK3 * TN3) |ti| {
            var ring: [64]u32 = undefined;
            hs.ringOf(part.tr[ti * words ..][0..words], ring[0..nw]);
            for (0..16) |lane| {
                for (0..16) |v| {
                    try std.testing.expectEqual(
                        hs.stateFloor(ring[0..nw], halves, @intCast(lane), @intCast(v)),
                        hs.stateT1(ring[0..nw], halves, @intCast(lane), @intCast(v)),
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// T-HS-8 (spec section 4): loader admission of odd halves and refusal of the
// shapes that lie about them, on the EAQ reader side — the loader that serves
// the odd-halves packs the step-5 kernels decode. The entries are fabricated
// in memory: map() reads only names, params and part shapes, so no pack
// directory or shard bytes are involved. The stock-path mirror (admitExl3's
// refusals and the histogram over a real pack) is the hyper03 re-gate run.

// (the named core module at the top owns exl3_format and the container; the T-HS-8
// block below reaches the reader and eaq_container through the named modules)
const reader = @import("exl3_eaq");
const eaqc = core.eaq_container;

fn trellisEntry(
    a: std.mem.Allocator,
    name: []const u8,
    bits: []const u8,
    dict: []const u8,
    k: usize,
    n: usize,
    lastdim: usize,
) !eaqc.Entry {
    const parts = try a.alloc(eaqc.Part, 3);
    const shapes = try a.alloc(usize, 5);
    parts[0] = .{ .name = "trellis", .tensor = "t", .dtype = .i16, .shape = shapes[0..3] };
    shapes[0] = k / 16; shapes[1] = n / 16; shapes[2] = lastdim;
    parts[1] = .{ .name = "suh", .tensor = "h", .dtype = .f16, .shape = shapes[3..4] };
    shapes[3] = k;
    parts[2] = .{ .name = "svh", .tensor = "v", .dtype = .f16, .shape = shapes[4..5] };
    shapes[4] = n;
    const text = try std.fmt.allocPrint(a, "{{\"bits\":{s},\"K\":{d},\"N\":{d},\"dictionary\":\"{s}\"}}", .{ bits, k, n, dict });
    const v = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
    if (v != .object) return error.BadManifest;
    return .{ .name = name, .format = .trellis, .params = v.object, .parts = parts, .shard = "model-00001.safetensors" };
}

fn fakePack(entries: []const eaqc.Entry) eaqc.Pack {
    return .{
        .arena = undefined, // map() never touches it; the test frees its own arena
        .manifest_sha256 = std.mem.zeroes([32]u8),
        .family = reader.family,
        .version = "eaq.pack/1",
        .features = &.{},
        .shards = &.{},
        .entries = entries,
        .n_tensors = entries.len * 3,
    };
}

test "T-HS-8: the reader admits a halves-5 mul1 head entry" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const entries = try a.alloc(eaqc.Entry, 1);
    entries[0] = try trellisEntry(a, "lm_head.weight", "2.5", "exl3.mul1.v1", 64, 128, 40);
    const pack = fakePack(entries);
    var table = try reader.map(gpa, &pack, null);
    defer table.deinit(gpa);
    const head = table.head orelse return error.TestUnexpectedResult;
    const q = head.q orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u8, 5), q.halves);
    try std.testing.expect(q.suh);
    try std.testing.expectEqual(exl3_format.Codebook.mul1, q.codebook);
}

test "T-HS-8: a fused slot mixes halves 5 and 6 parts once the agreement is K-only" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const entries = try a.alloc(eaqc.Entry, 3);
    // the stock 2.0bpw shape: layer 3's q at bits 2, k and v at bits 3, one fused
    // mixer_in slot (exl3_eaq.zig:366-382 admits it: same K, same mul1 codebook)
    entries[0] = try trellisEntry(a, "model.layers.3.self_attn.q_proj", "2.0", "exl3.mul1.v1", 1024, 3072, 32);
    entries[1] = try trellisEntry(a, "model.layers.3.self_attn.k_proj", "2.5", "exl3.mul1.v1", 1024, 512, 40);
    entries[2] = try trellisEntry(a, "model.layers.3.self_attn.v_proj", "3.0", "exl3.mul1.v1", 1024, 512, 48);
    const pack = fakePack(entries);
    var table = try reader.map(gpa, &pack, null);
    defer table.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 3), table.projections.len);
    for (table.projections, [_]u8{ 4, 5, 6 }) |p, want| {
        try std.testing.expectEqual(@as(usize, 3), p.layer);
        try std.testing.expectEqual(reader.Slot.mixer_in, p.slot);
        try std.testing.expectEqual(want, p.q.?.halves);
    }
}

test "T-HS-8: a fused part that moves K or codebook still refuses" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    {
        const entries = try a.alloc(eaqc.Entry, 2);
        entries[0] = try trellisEntry(a, "model.layers.3.self_attn.q_proj", "2.0", "exl3.mul1.v1", 1024, 3072, 32);
        entries[1] = try trellisEntry(a, "model.layers.3.self_attn.k_proj", "2.5", "exl3.mul1.v1", 512, 512, 40);
        const pack = fakePack(entries);
        var why: eaqc.Why = .{};
        try std.testing.expectError(reader.Error.MismatchedFusedK, reader.map(gpa, &pack, &why));
        try std.testing.expect(std.mem.indexOf(u8, why.text(), "K or codebook") != null);
    }
    {
        const entries = try a.alloc(eaqc.Entry, 2);
        entries[0] = try trellisEntry(a, "model.layers.3.self_attn.q_proj", "2.0", "exl3.mul1.v1", 1024, 3072, 32);
        entries[1] = try trellisEntry(a, "model.layers.3.self_attn.k_proj", "3.0", "exl3.mcg.v1", 1024, 512, 48);
        const pack = fakePack(entries);
        try std.testing.expectError(reader.Error.MismatchedFusedK, reader.map(gpa, &pack, null));
    }
}

test "T-HS-8: an odd-halves entry outside mul1 refuses at the codebook rule" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const entries = try a.alloc(eaqc.Entry, 1);
    entries[0] = try trellisEntry(a, "lm_head.weight", "2.5", "exl3.mcg.v1", 64, 128, 40);
    const pack = fakePack(entries);
    var why: eaqc.Why = .{};
    try std.testing.expectError(reader.Error.BadTrellisParams, reader.map(gpa, &pack, &why));
    try std.testing.expect(std.mem.indexOf(u8, why.text(), "odd halves") != null);
}

test "T-HS-8: a trellis last dim that is not 8 times its declared halves refuses" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const entries = try a.alloc(eaqc.Entry, 1);
    entries[0] = try trellisEntry(a, "lm_head.weight", "2.5", "exl3.mul1.v1", 64, 128, 44); // declares 40
    const pack = fakePack(entries);
    try std.testing.expectError(reader.Error.BadTrellisPart, reader.map(gpa, &pack, null));
}
