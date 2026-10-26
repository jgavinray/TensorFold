//! fragcheck: the pack-side verifier for the T-LDLQ-5 controversy (QuantFix 2026-10-26).
//! Reads the WRITTEN fragment file - every tensor its own header names, hashed - and
//! decodes the file's .trellis/.suh/.svh through core/exl3_format.dequantize (the
//! probe's oracle code path, host-only, no device), scoring against the source bf16
//! tensor exactly as ProbeJob.run does: column-major source read, E = |decode - source|
//! / |svh[col]|, weight-space Pearson. Prints the whole-part RMS and corr, the
//! G4-style table for a named block, the forensics 128-row band, and the global worst -
//! the numbers the two decodes disagree on, decided on the file's own bytes.
//! Usage: xpu-fragcheck FRAG.safetensors MODEL_DIR [BLOCK]
const std = @import("std");
const core = @import("core");
const q = @import("qwen27");
const devm = q.dev;
const Checkpoint = devm.Checkpoint;
const efmt = core.exl3_format;

const alloc = std.heap.c_allocator;
extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn lseek(fd: c_int, off: i64, whence: c_int) i64;
extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) ?*anyopaque;
extern "c" fn munmap(addr: *anyopaque, len: usize) c_int;
const PROT_READ = 1;
const MAP_PRIVATE = 2;

const MappedFile = struct {
    bytes: []const u8,
    pub fn openFile(path: []const u8) !MappedFile {
        const sent = try alloc.dupeSentinel(u8, path, 0);
        defer alloc.free(sent);
        const fd = open(sent.ptr, 0);
        if (fd < 0) return error.FileNotFound;
        defer _ = close(fd);
        const size: usize = @intCast(lseek(fd, 0, 2));
        const base = mmap(null, size, PROT_READ, MAP_PRIVATE, fd, 0) orelse return error.MmapFailed;
        if (@intFromPtr(base) == @as(usize, @bitCast(@as(isize, -1)))) return error.MmapFailed;
        return .{ .bytes = @as([*]const u8, @ptrCast(base))[0..size] };
    }
    pub fn unmap(self: *MappedFile) void {
        if (self.bytes.len == 0) return;
        _ = munmap(@ptrCast(@constCast(self.bytes.ptr)), self.bytes.len);
        self.bytes = &.{};
    }
};

fn sha256Hex(bytes: []const u8) [64]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(bytes);
    const d = h.finalResult();
    const hex = "0123456789abcdef";
    var out: [64]u8 = undefined;
    for (d, 0..) |byte, i| {
        out[i * 2] = hex[byte >> 4];
        out[i * 2 + 1] = hex[byte & 15];
    }
    return out;
}

fn sha256File(path: []const u8) ![64]u8 {
    var mf = try MappedFile.openFile(path);
    defer mf.unmap();
    return sha256Hex(mf.bytes);
}

fn readFile(path: []const u8) ![]u8 {
    var mf = try MappedFile.openFile(path);
    defer mf.unmap();
    return alloc.dupe(u8, mf.bytes);
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(alloc);
    if (args.len < 3) {
        std.debug.print("usage: xpu-fragcheck FRAG.safetensors MODEL_DIR [BLOCK]\n", .{});
        return error.Usage;
    }
    const frag_path = args[1];
    const model_dir = args[2];
    const report_block: usize = if (args.len > 3) try std.fmt.parseInt(usize, args[3], 10) else 1087;

    const sha = try sha256File(frag_path);
    std.debug.print("fragcheck: {s}\n  file sha256 {s}\n", .{ frag_path, sha[0..] });

    // the header's own enumeration: names, dtypes, shapes, payload size and hash of each
    var frag = Checkpoint{};
    defer frag.deinit();
    const f0 = try alloc.dupeSentinel(u8, frag_path, 0);
    defer alloc.free(f0);
    try frag.addShard(f0);
    var total: u64 = 0;
    for (frag.tensors.items) |t| {
        total += t.bytes.len;
        std.debug.print("  {s} {s} {any} {d} B sha {s}\n", .{
            t.name, t.dtype, t.shape[0..t.rank], t.bytes.len, sha256Hex(t.bytes)[0..16],
        });
    }
    var mfm = try MappedFile.openFile(frag_path);
    defer mfm.unmap();
    const fsize: u64 = mfm.bytes.len;
    std.debug.print("fragcheck: {d} tensors, payload sum {d} B, file {d} B\n", .{ frag.tensors.items.len, total, fsize });

    const tr_t = try frag.get("model.layers.0.mlp.down_proj.trellis");
    const suh_t = try frag.get("model.layers.0.mlp.down_proj.suh");
    const svh_t = try frag.get("model.layers.0.mlp.down_proj.svh");
    const tnk = tr_t.shape[0]; // k/16
    const tnn = tr_t.shape[1]; // n/16
    const words = tr_t.shape[2];
    const halves: u8 = @intCast(words / 8);
    const w: efmt.Width = .{ .halves = halves };
    if (words != w.tileWords()) return error.BadDims;
    const k: usize = tnk * 16;
    const n: usize = tnn * 16;
    if (tr_t.bytes.len != tnk * tnn * words * 2) return error.BadBytes;
    const tr: []const i16 = @alignCast(std.mem.bytesAsSlice(i16, tr_t.bytes));
    const suh: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, suh_t.bytes));
    const svh: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, svh_t.bytes));
    std.debug.print("fragcheck: down_proj k {d} n {d} halves {d} words {d} (trellis sha {s})\n", .{
        k, n, halves, words, sha256Hex(tr_t.bytes)[0..16],
    });

    // the forensic tiles' raw words, for the dossiers' p-65/p-68 predictions
    for ([2][2]usize{ .{ 476, 163 }, .{ 550, 249 } }) |tile| {
        const base = (tile[0] * tnn + tile[1]) * words;
        const wl = tr[base..][0..words];
        std.debug.print("fragcheck: tile ({d},{d}) words:", .{ tile[0], tile[1] });
        for (wl, 0..) |word, p| {
            if (p == 65 or p == 68 or p < 2) std.debug.print(" p{d}={x:0>4}", .{ p, word });
        }
        std.debug.print("\n  full: {any}\n", .{wl});
    }

    // the source shard for down_proj, straight from the model's index
    const idxp = try std.fmt.allocPrint(alloc, "{s}/model.safetensors.index.json", .{model_dir});
    const idx = try readFile(idxp);
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, idx, .{});
    const wm = parsed.value.object.get("weight_map").?;
    const wname = "model.language_model.layers.0.mlp.down_proj.weight";
    const shard = wm.object.get(wname).?.string;
    var src = Checkpoint{};
    defer src.deinit();
    const sp = try std.fmt.allocPrintSentinel(alloc, "{s}/{s}", .{ model_dir, shard }, 0);
    try src.addShard(sp);
    const wsrc = try src.get(wname);
    std.debug.print("fragcheck: source {s} from {s}: {s} {any}\n", .{ wname, shard, wsrc.dtype, wsrc.shape[0..wsrc.rank] });
    if (wsrc.bytes.len != k * n * 2) return error.BadSource;

    // the probe's oracle split: 128-row strips, block-local rotations
    const out = try alloc.alloc(f64, k * n);
    defer alloc.free(out);
    const scratch = try alloc.alloc(f64, 2 * 128 * n);
    defer alloc.free(scratch);
    var strip: usize = 0;
    while (strip < k / 128) : (strip += 1) {
        const row0 = strip * 128;
        const s = tr[(row0 / 16) * tnn * words ..][0..8 * tnn * words];
        efmt.dequantize(w, .mul1, s, 8, tnn, suh[row0..][0..128], svh, out[row0 * n ..][0..128 * n], scratch);
    }

    // score exactly as ProbeJob.run: column-major source ((cc*k)+rr)*2, /|svh[cc]|,
    // weight-space Pearson. The down part has zero dead columns (the site printed
    // dead 0), so the zc skip is a no-op here and every element is scored.
    const kblks = k / 16;
    const blk_sq = try alloc.alloc(f64, kblks);
    defer alloc.free(blk_sq);
    @memset(blk_sq, 0);
    var blk_worst = try alloc.alloc(f64, kblks);
    defer alloc.free(blk_worst);
    @memset(blk_worst, 0);
    var blk_wr = try alloc.alloc(usize, kblks);
    defer alloc.free(blk_wr);
    var blk_wc = try alloc.alloc(usize, kblks);
    defer alloc.free(blk_wc);
    var sq_err: f64 = 0;
    var st: f64 = 0;
    var sq: f64 = 0;
    var stq: f64 = 0;
    var stt: f64 = 0;
    var sqq: f64 = 0;
    var worst: f64 = 0;
    var wr: usize = 0;
    var wc: usize = 0;
    var worst_abs: f64 = 0;
    var wa_r: usize = 0;
    var wa_c: usize = 0;
    var ncnt: u64 = 0;
    for (0..k) |rr| {
        const jb = rr / 16;
        for (0..n) |cc| {
            const qv: f32 = @floatCast(out[rr * n + cc]);
            const idxb = (cc * k + rr) * 2;
            const sbits: u32 = @as(u32, wsrc.bytes[idxb]) | (@as(u32, wsrc.bytes[idxb + 1]) << 8);
            const sv_src: f32 = @as(f32, @bitCast(sbits << 16));
            const ef: f64 = @abs(@as(f64, qv) - @as(f64, sv_src)) / @max(@as(f64, @abs(svh[cc])), 1e-30);
            blk_sq[jb] += ef * ef;
            if (ef > blk_worst[jb]) {
                blk_worst[jb] = ef;
                blk_wr[jb] = rr;
                blk_wc[jb] = cc;
            }
            sq_err += ef * ef;
            if (ef > worst) {
                worst = ef;
                wr = rr;
                wc = cc;
            }
            const ae = @abs(@as(f64, qv) - @as(f64, sv_src));
            if (ae > worst_abs) {
                worst_abs = ae;
                wa_r = rr;
                wa_c = cc;
            }
            st += sv_src;
            sq += qv;
            stq += @as(f64, sv_src) * qv;
            stt += @as(f64, sv_src) * sv_src;
            sqq += @as(f64, qv) * qv;
            ncnt += 1;
        }
    }
    const cnt: f64 = @floatFromInt(ncnt);
    const corr = (stq - st * sq / cnt) / @max(@sqrt((stt - st * st / cnt) * (sqq - sq * sq / cnt)), 1e-300);
    std.debug.print("fragcheck: whole-part err RMS {d:.6} of scale; corr(pack, source) {d:.6}\n", .{ @sqrt(sq_err / cnt), corr });
    std.debug.print("  (probe's in-memory table says 0.1357 / 0.55 @(17405,4911) corr 0.9899;\n   forensics' numpy file-decode says 1.4012 / 6.77 @(17395,2264) corr -0.000123, whole 1.421647)\n", .{});

    const jb = report_block;
    std.debug.print("fragcheck: block {d} (rows {d}..{d}): RMS {d:.4} worst {d:.2} @({d},{d})\n", .{
        jb, jb * 16, (jb + 1) * 16 - 1, @sqrt(blk_sq[jb] / @as(f64, @floatFromInt(16 * n))), blk_worst[jb], blk_wr[jb], blk_wc[jb],
    });
    { // the forensics band, rows 17280..17407
        var b2: f64 = 0;
        var w2: f64 = 0;
        var r2: usize = 0;
        var c2: usize = 0;
        for (17280..17408) |rr| {
            for (0..n) |cc| {
            const qv: f32 = @floatCast(out[rr * n + cc]);
            const idxb = (cc * k + rr) * 2;
            const sbits: u32 = @as(u32, wsrc.bytes[idxb]) | (@as(u32, wsrc.bytes[idxb + 1]) << 8);
            const sv_src: f32 = @as(f32, @bitCast(sbits << 16));
            const ef: f64 = @abs(@as(f64, qv) - @as(f64, sv_src)) / @max(@as(f64, @abs(svh[cc])), 1e-30);
            b2 += ef * ef;
            if (ef > w2) {
                w2 = ef;
                r2 = rr;
                c2 = cc;
            }
        }
        }
        std.debug.print("fragcheck: rows 17280..17407: RMS {d:.4} worst {d:.2} @({d},{d})\n", .{ @sqrt(b2 / @as(f64, @floatFromInt(128 * n))), w2, r2, c2 });
    }
    std.debug.print("fragcheck: global worst {d:.2} @({d},{d}); absolute worst {d:.4} @({d},{d})\n", .{ worst, wr, wc, worst_abs, wa_r, wa_c });
    std.debug.print("  (forensics predicts 22.98 @(7617,2610) and 1.0585 @(8800,3994) for the file's bytes)\n", .{});
}
