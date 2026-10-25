//! Host tests for the quant driver — no device, no card (`zig build xpu-eaq-quant-tests`).
//! (a) Progress save-then-load roundtrip: the hex-text regression (B1), the S3 fallback
//!     flag and the S5/P1-4 head and stream fields survive the JSON hop byte-exact;
//! (b) the head window merge against a direct stitched comparison (B2);
//! (c) the P1-3 gate: a synthetic pack whose GDN a/b gates carry the F16 header the
//!     publish now declares passes the reader's whole loadExl3 admission (openEaq plus
//!     admitExl3, not the container alone), and the pre-fix BF16-declared header is
//!     refused by that admission with Exl3NotDense.
const std = @import("std");
const quant = @import("quant.zig");
const q = @import("qwen27");
const core = @import("core");
const eaq_write = @import("eaq_write");

const alloc = std.heap.c_allocator;

// tmp dirs without std.testing.TmpDir: this root builds as an executable, where
// TmpDir.cleanup's comptime assert(builtin.is_test) does not hold
const io_ctx = std.Io.Threaded.global_single_threaded.io();
var tmp_seq: u32 = 0;
const Tmp = struct { path: []const u8 };
fn mkdirp(path: []const u8) void {
    var at: usize = 1;
    while (std.mem.indexOfScalarPos(u8, path, at, '/')) |sl| {
        const pre = alloc.dupeSentinel(u8, path[0..sl], 0) catch return;
        defer alloc.free(pre);
        _ = std.c.mkdir(pre.ptr, 0o755); // EEXIST is fine
        at = sl + 1;
    }
    const sent = alloc.dupeSentinel(u8, path, 0) catch return;
    defer alloc.free(sent);
    _ = std.c.mkdir(sent.ptr, 0o755);
}
fn mkTmp() Tmp {
    tmp_seq += 1;
    const p = std.fmt.allocPrint(alloc, "/tmp/eaq-quant-test-{d}-{d}", .{ std.c.getpid(), tmp_seq }) catch @panic("oom");
    mkdirp(p);
    return .{ .path = p };
}
fn rmTmp(t: Tmp) void {
    std.Io.Dir.cwd().deleteTree(io_ctx, t.path) catch {};
}

fn testProgressRoundtrip() !void {
    const dir = mkTmp();
    defer rmTmp(dir);
    var p: quant.Progress = .{
        .src = "/models/Qwen3.8-27B",
        .calib_sha = @splat('a'),
        .rows = 100,
        .cols = 2048,
        .bits = 3,
        .head_bits = 4,
        .seed = 0xE0,
        .lambda = 0.01,
        .scale = 1.0,
        .head_sha = @splat('c'),
        .head_idfb = true, // P1-4: the fallback verdict must ride the JSON hop
        .stream_sha = @splat('d'),
        .stream_i = 64,
    };
    try p.layers.append(alloc, .{ .i = 0, .sha = @splat('b'), .frag = "layers0.safetensors", .identity_fallback = false });
    try p.layers.append(alloc, .{ .i = 1, .sha = @splat('e'), .frag = "layers1.safetensors", .identity_fallback = true });
    try p.save(dir.path);
    const l = (try quant.Progress.load(dir.path)) orelse return error.NoProgress;
    try std.testing.expectEqualStrings(p.src, l.src);
    try std.testing.expectEqualStrings("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", &l.calib_sha); // B1: the reloaded bytes are the hex TEXT, not raw bytes
    try std.testing.expectEqual(@as(usize, 2), l.layers.items.len);
    try std.testing.expectEqual(@as(usize, 0), l.layers.items[0].i);
    try std.testing.expectEqualStrings("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", &l.layers.items[0].sha);
    try std.testing.expectEqualStrings("layers0.safetensors", l.layers.items[0].frag);
    try std.testing.expect(!l.layers.items[0].identity_fallback);
    try std.testing.expect(l.layers.items[1].identity_fallback);
    try std.testing.expectEqual(@as(u32, 100), l.rows);
    try std.testing.expectEqual(@as(u32, 2048), l.cols);
    try std.testing.expectEqual(@as(u32, 3), l.bits);
    try std.testing.expectEqual(@as(u32, 4), l.head_bits);
    try std.testing.expectEqual(@as(u64, 0xE0), l.seed);
    try std.testing.expectEqual(@as(f64, 0.01), l.lambda);
    try std.testing.expectEqual(@as(f64, 1.0), l.scale);
    if (l.head_sha) |hs| try std.testing.expectEqualStrings("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc", &hs) else return error.HeadShaLost;
    try std.testing.expect(l.head_idfb);
    if (l.stream_sha) |ss| try std.testing.expectEqualStrings("dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd", &ss) else return error.StreamShaLost;
    try std.testing.expectEqual(@as(usize, 64), l.stream_i);
}

fn testWindowMerge() !void {
    // Three 16-wide windows over a k = 32 part: chunk w's block-row blk word x carries
    // (blk*words + x)*10 + w; the merge must land it at column tile w of block row blk.
    const k: usize = 32;
    const words: usize = 2;
    const win_n: u32 = 16;
    const nt: usize = 48;
    const per = (k / 16) * (win_n / 16) * words;
    var trbuf: [3][per]i16 = undefined;
    var chunks: [3]quant.PartWork = undefined;
    for (&chunks, 0..) |*ch, w| {
        for (&trbuf[w], 0..) |*v, i| v.* = @intCast((i * 10) +% w); // i == blk*words + x within the chunk
        ch.* = .{
            .spec = .{ .base = "b", .tail = "t", .n = 16, .n0 = @intCast(w * 16) },
            .win_n = win_n,
            .tr = &trbuf[w],
            .suh = &.{},
            .svh = &.{},
            .halves = 6,
            .rec = .{ .prefix = "p" },
        };
    }
    const merged = try quant.mergeWindowTrellises(&chunks, k, nt, words);
    defer alloc.free(merged);
    var exp: [(k / 16) * (nt / 16) * words]i16 = undefined;
    for (0..k / 16) |blk| for (0..3) |w| for (0..words) |x| {
        exp[(blk * (nt / 16) + w) * words + x] = @intCast(((blk * words + x) * 10) +% w);
    };
    try std.testing.expectEqualSlices(i16, &exp, merged);

    // Uneven windows (16 then 32 columns): the second chunk's two tiles shift one over.
    const win2: [2]u32 = .{ 16, 32 };
    var tr2: [2][32]i16 = undefined; // sized for the wider chunk
    var ch2: [2]quant.PartWork = undefined;
    for (&ch2, 0..) |*c, w| {
        const cnt: usize = win2[w] / 16;
        for (&tr2[w], 0..) |*v, i| v.* = if (i < (k / 16) * cnt * words) @intCast(i * 3 +% w) else 0;
        c.* = .{
            .spec = .{ .base = "b", .tail = "t", .n = win2[w], .n0 = if (w == 0) 0 else 16 },
            .win_n = win2[w],
            .tr = tr2[w][0..(k / 16) * cnt * words],
            .suh = &.{},
            .svh = &.{},
            .halves = 6,
            .rec = .{ .prefix = "p" },
        };
    }
    const merged2 = try quant.mergeWindowTrellises(&ch2, k, nt, words);
    defer alloc.free(merged2);
    for (0..k / 16) |blk| {
        for (0..words) |x| { // chunk 0: tile 0 <- its tr[blk*1*words + x]
            try std.testing.expectEqual(tr2[0][(blk * words) + x], merged2[(blk * (nt / 16)) * words + x]);
        }
        for (0..2) |t| {
            for (0..words) |x| { // chunk 1: tiles 1,2 <- its tr[(blk*2 + t)*words + x]
                try std.testing.expectEqual(tr2[1][(blk * 2 + t) * words + x], merged2[(blk * (nt / 16) + 1 + t) * words + x]);
            }
        }
    }
}

// ---------------------------------------------------------------- the gate pack (P1-3)

const D: usize = 5120; // q.D, pinned so the 48 x D gate test fails loudly if the model spec moves
const LayerSpec = struct { in: []const []const u8, out: []const []const u8, gdn: bool };

fn layerSpec(i: usize) LayerSpec {
    return if (q.isAttention(i))
        .{ .in = &.{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj" }, .out = &.{"self_attn.o_proj"}, .gdn = false }
    else
        .{ .in = &.{ "linear_attn.in_proj_qkv", "linear_attn.in_proj_z" }, .out = &.{"linear_attn.out_proj"}, .gdn = true };
}

fn fill(n: usize, seed: u8) []const u8 {
    const b = alloc.alloc(u8, n) catch @panic("oom");
    for (b, 0..) |*v, i| v.* = @intCast(((i *% 251) +% seed) & 0xff);
    return b;
}

fn bundle(list: *std.ArrayList(eaq_write.Input), base: []const u8) !void {
    const trellis = try std.fmt.allocPrint(alloc, "{s}.trellis", .{base});
    const suh = try std.fmt.allocPrint(alloc, "{s}.suh", .{base});
    const svh = try std.fmt.allocPrint(alloc, "{s}.svh", .{base});
    const mul1 = try std.fmt.allocPrint(alloc, "{s}.mul1", .{base});
    const t = try alloc.alloc(u8, 8 * 8 * 16 * 2); // I16 [8, 8, 16]: k = n = 128 (the admission's minimum), halves = 2
    @memset(t, 0x11);
    const s = try alloc.alloc(u8, 16); // f16 [8]
    @memset(s, 0x22);
    const w: u32 = core.exl3_format.Mul1Mul; // the I32 codebook marker: the magic word
    const m = try alloc.dupe(u8, std.mem.asBytes(&w));
    // payloads must outlive this frame: the writer reads them at pack time, so they live on the bump arena
    try list.append(alloc, .{ .name = trellis, .dtype = .i16, .shape = &.{ 8, 8, 16 }, .payload = t });
    try list.append(alloc, .{ .name = suh, .dtype = .f16, .shape = &.{8}, .payload = s });
    try list.append(alloc, .{ .name = svh, .dtype = .f16, .shape = &.{8}, .payload = s });
    try list.append(alloc, .{ .name = mul1, .dtype = .i32, .shape = &.{}, .payload = m });
}

fn gateBytes(alloc_: std.mem.Allocator) ![]const u8 {
    return alloc_.alloc(u8, 48 * D * 2);
}

fn buildGatePack(a: std.mem.Allocator, gate_dtype: eaq_write.DType) !std.ArrayList(eaq_write.Input) {
    var list: std.ArrayList(eaq_write.Input) = .empty;
    try list.append(a, .{ .name = "model.embed_tokens.weight", .dtype = .f16, .shape = &.{ 2, 2 }, .payload = fill(8, 'e') });
    try list.append(a, .{ .name = "lm_head.weight", .dtype = .f16, .shape = &.{ 2, 2 }, .payload = fill(8, 'h') }); // the seed head's F16 route (the trellis head must be 5120 x 248320 — impossible in a synthetic)
    for (0..64) |i| {
        const sp = layerSpec(i);
        const p = try std.fmt.allocPrint(a, "model.layers.{d}.", .{i});
        if (sp.gdn) { // the GDN gates belong to GDN layers only; attention layers carry their own projections
            const gname = try std.fmt.allocPrint(a, "model.layers.{d}.linear_attn.in_proj_a.weight", .{i});
            const bname = try std.fmt.allocPrint(a, "model.layers.{d}.linear_attn.in_proj_b.weight", .{i});
            for ([_]struct { name: []const u8, shape: []const usize, payload: []const u8 }{
                .{ .name = gname, .shape = &.{ 48, D }, .payload = try gateBytes(a) },
                .{ .name = bname, .shape = &.{ 48, D }, .payload = try gateBytes(a) },
            }) |gd| try list.append(a, .{ .name = gd.name, .dtype = gate_dtype, .shape = gd.shape, .payload = gd.payload });
        }
        for (sp.in) |part| try bundle(&list, try std.fmt.allocPrint(a, "{s}{s}", .{ p, part }));
        for (sp.out) |part| try bundle(&list, try std.fmt.allocPrint(a, "{s}{s}", .{ p, part }));
        for ([_][]const u8{ "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj" }) |part|
            try bundle(&list, try std.fmt.allocPrint(a, "{s}{s}", .{ p, part }));
    }
    return list;
}

fn openPack(dir: []const u8) !q.Checkpoint {
    var ck: q.Checkpoint = .{};
    errdefer ck.deinit();
    try ck.openEaq(dir);
    return ck;
}

fn testGateAdmission() !void {
    const dir = mkTmp();
    defer rmTmp(dir);
    const io = std.Io.Threaded.global_single_threaded.io();

    // The fixed driver's shape: the converted gates wear the F16 header — the whole
    // loadExl3 admission, not just the container, must pass.
    const good = try buildGatePack(alloc, .f16);
    var why: core.eaq_container.Why = .{};
    _ = eaq_write.write(alloc, io, dir.path, good.items, .{ .config = null, .shards = 4 }, &why) catch |e| {
        std.debug.print("gate-good: {s}: {s}\n", .{ @errorName(e), why.text() });
        return e;
    };
    var ck = try openPack(dir.path);
    defer ck.deinit();
    const adm = try q.loader.admitExl3(&ck);
    try std.testing.expect(adm.summary.groups > 0);
    // The header the pack actually declares is F16 (the pre-fix driver said BF16 here).
    var gdn: ?usize = null;
    for (0..64) |i| if (!q.isAttention(i)) {
        gdn = i;
        break;
    };
    if (gdn) |i| {
        const g = try ck.get(try std.fmt.allocPrint(alloc, "model.layers.{d}.linear_attn.in_proj_a.weight", .{i}));
        try std.testing.expectEqualStrings("F16", g.dtype);
        try std.testing.expectEqual(@as(usize, 48 * D * 2), g.bytes.len);
    } else return error.NoGdnLayer;

    // The pre-fix defect: right bytes, BF16-declared header. openEaq alone admits it;
    // the loadExl3 admission must refuse it with Exl3NotDense.
    const dir2 = mkTmp();
    defer rmTmp(dir2);
    const bad = try buildGatePack(alloc, .bf16);
    var why2: core.eaq_container.Why = .{};
    _ = try eaq_write.write(alloc, io, dir2.path, bad.items, .{ .config = null, .shards = 4 }, &why2);
    var ck2 = try openPack(dir2.path);
    defer ck2.deinit();
    try std.testing.expectError(error.Exl3NotDense, q.loader.admitExl3(&ck2));
}
/// The Q3 binding: the writer takes the driver's report bytes, hashes them into the
/// staged shards' metadata, and refuses anything that misdescribes the pack — a failed
/// tile, an identity-fallback part, a missing contract, a part whose trellis digest
/// does not match the staged bytes. The report builder below mirrors exactly what the
/// driver's buildReport emits per part, so this is the driver's side of the contract
/// under test, not just the writer's.
fn gateReport(a: std.mem.Allocator, list: []const eaq_write.Input, failed: u32, idfb: bool, first_wrong_sha: bool) ![]u8 {
    var sb: std.ArrayList(u8) = .empty;
    try sb.appendSlice(a, "{\"contract\":\"c\",\"encoder\":\"xpu-kernel\",\"parts\":[");
    var n: usize = 0;
    for (list) |t| {
        if (!std.mem.endsWith(u8, t.name, ".trellis")) continue;
        if (n > 0) try sb.appendSlice(a, ",");
        var dg: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(t.payload, &dg, .{});
        var hx: [64]u8 = std.fmt.bytesToHex(dg, .lower);
        if (first_wrong_sha and n == 0) hx[0] ^= 0x20;
        const one = try std.fmt.allocPrint(a,
            "{{\"prefix\":\"{s}\",\"encoder\":\"xpu-kernel\",\"k\":{d},\"n\":{d},\"halves\":{d},\"failed\":{d},\"identity_fallback\":{s},\"trellis_sha256\":\"{s}\"}}",
            .{
                t.name[0 .. t.name.len - ".trellis".len],
                t.shape[0] * 16, t.shape[1] * 16, t.shape[2] / 8,
                failed, if (idfb) "true" else "false", hx[0..],
            });
        try sb.appendSlice(a, one);
        n += 1;
    }
    try sb.appendSlice(a, "]}");
    return sb.items;
}

fn testReportBinding() !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    const list = try buildGatePack(alloc, .f16);

    const dir = mkTmp();
    defer rmTmp(dir);
    var why: core.eaq_container.Why = .{};
    const good = try gateReport(alloc, list.items, 0, false, false);
    _ = try eaq_write.write(alloc, io, dir.path, list.items, .{ .config = null, .shards = 4, .report = good }, &why);
    var ck = try openPack(dir.path); // the bound pack still opens
    ck.deinit();

    const dir_m = mkTmp();
    defer rmTmp(dir_m);
    var why_m: core.eaq_container.Why = .{};
    try std.testing.expectError( // one part digests other bytes than the staged trellis
        error.ReportMismatch,
        eaq_write.write(alloc, io, dir_m.path, list.items, .{ .config = null, .shards = 4, .report = try gateReport(alloc, list.items, 0, false, true) }, &why_m),
    );

    const dir_b = mkTmp();
    defer rmTmp(dir_b);
    var why_b: core.eaq_container.Why = .{};
    try std.testing.expectError(
        error.ReportFailed,
        eaq_write.write(alloc, io, dir_b.path, list.items, .{ .config = null, .shards = 4, .report =
            \\{"contract":"c","encoder":"xpu-kernel","parts":[{"failed":1}]}
        }, &why_b),
    );

    const dir_c = mkTmp();
    defer rmTmp(dir_c);
    var why_c: core.eaq_container.Why = .{};
    try std.testing.expectError(
        error.ReportFailed,
        eaq_write.write(alloc, io, dir_c.path, list.items, .{ .config = null, .shards = 4, .report =
            \\{"contract":"c","encoder":"xpu-kernel","parts":[{"failed":0,"identity_fallback":true}]}
        }, &why_c),
    );

    const dir_d = mkTmp();
    defer rmTmp(dir_d);
    var why_d: core.eaq_container.Why = .{};
    try std.testing.expectError(
        error.BadReport, // no contract
        eaq_write.write(alloc, io, dir_d.path, list.items, .{ .config = null, .shards = 4, .report =
            \\{"encoder":"xpu-kernel","parts":[{"failed":0}]}
        }, &why_d),
    );
}


pub fn main() !void {
    try testProgressRoundtrip();
    std.debug.print("ok: progress save-then-load roundtrip (B1, S3, S5, P1-4)\n", .{});
    try testWindowMerge();
    std.debug.print("ok: head window merge matches the direct stitch (B2)\n", .{});
    try testGateAdmission();
    std.debug.print("ok: synthetic pack — F16 gates pass the loadExl3 admission, BF16-declared gates refuse (P1-3)\n", .{});
    try testReportBinding();
    std.debug.print("ok: the writer binds a clean report and refuses failed/identity/contract-less reports (Q3)\n", .{});
    std.debug.print("xpu-eaq-quant-tests: all green\n", .{});
}
