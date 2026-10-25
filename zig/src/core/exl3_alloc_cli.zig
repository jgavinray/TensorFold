//! Host CLI for the EXL3 bit allocator: emit a ready-to-serve plan file for the
//! 27B run (docs/eaq-review/eaq-halfstep-spec-fable.md 3.2: the driver consumes
//! `--plan <file>`). Builds the real kept-851 inventory, allocates at the target
//! (head fixed at head_bits, outside the body budget), renders the plan JSON through
//! `planToJson`, writes it, then re-reads it through `planFromJson` and proves the
//! resume identity: the embedded plan_sha equals the sha recomputed over the
//! canonical serialization of BOTH the emitted and the parsed-back plan.
//!
//!   zig run zig/src/core/exl3_alloc_cli.zig -- <target_bpw> <out.json> [head_bits]
//!   e.g. ... -- 2.9 ~/xpu-eaq/plan-2.9.json        (head_bits defaults to 4)

const std = @import("std");
const alloc = @import("exl3_alloc.zig");
const Io = std.Io;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3 or args.len > 4) {
        std.debug.print("usage: zig run exl3_alloc_cli.zig -- <target_bpw> <out.json> [head_bits]\n", .{});
        std.process.exit(1);
    }
    const target = try std.fmt.parseFloat(f64, args[1]);
    const out_path = args[2];
    const head_bits: u4 = if (args.len == 4) try std.fmt.parseInt(u4, args[3], 10) else 4;

    const ts = try alloc.qwen27Inventory(gpa);
    defer alloc.freeInventory(gpa, ts);
    const opts = alloc.Options{ .target_bits = target, .head_bits = head_bits };
    var al = try alloc.allocate(gpa, ts, opts);
    defer al.deinit(gpa);
    var p = try alloc.makePlan(gpa, ts, al, opts);
    defer alloc.freePlan(gpa, &p);

    const json = try alloc.planToJson(gpa, p);
    defer gpa.free(json);

    // write (creating the directory) and hash the exact bytes on disk
    var atomic = try Io.Dir.cwd().createFileAtomic(io, out_path, .{ .make_path = true, .replace = true });
    defer atomic.deinit(io);
    try atomic.file.writePositionalAll(io, json, 0);
    try atomic.replace(io);
    const on_disk = try Io.Dir.cwd().readFileAlloc(io, out_path, gpa, .limited(1 << 24));
    defer gpa.free(on_disk);
    var file_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(on_disk, &file_sha, .{});

    // reader path: parse back, re-serialize, re-hash; the resume identity must hold
    var rp = try alloc.planFromJson(gpa, on_disk);
    defer {
        for (rp.parts) |part| gpa.free(@constCast(part.base));
        gpa.free(@constCast(rp.parts));
        gpa.free(rp.method);
    }
    if (rp.parts.len != p.parts.len) return error.PartsMismatch;
    for (rp.parts, p.parts) |x, y| {
        if (!std.mem.eql(u8, x.base, y.base) or x.halves != y.halves) return error.PartsMismatch;
    }
    const sha_emitted = try alloc.planShaWith(gpa, p);
    if (!std.mem.eql(u8, &p.plan_sha, &sha_emitted)) return error.ShaMismatch;
    const sha_parsed = try alloc.planShaWith(gpa, rp);
    if (!std.mem.eql(u8, &p.plan_sha, &rp.plan_sha)) return error.ShaMismatch;
    if (!std.mem.eql(u8, &rp.plan_sha, &sha_parsed)) return error.ShaMismatch;

    var odd_body: usize = 0;
    for (p.parts) |part| {
        if (part.halves % 2 == 1) odd_body += 1;
    }
    var sha_hex: [64]u8 = undefined;
    var file_hex: [64]u8 = undefined;
    hex64(&sha_hex, p.plan_sha);
    hex64(&file_hex, file_sha);
    std.debug.print(
        "{s} @ body {d} bpw (head {d}): achieved {d:.6}, total {d:.6}, odd {d} of {d} body parts, " ++
            "attn avg {d:.3}, gdn avg {d:.3}, mlp avg {d:.3}, head {d:.1}\n" ++
            "  plan_sha      {s}\n" ++
            "  file sha256   {s}\n" ++
            "  reparse: parts {d}, sha over canonical bytes of emitted+parsed plans identical\n",
        .{
            out_path,
            target,
            al.report.head_bits,
            al.report.achieved_bits,
            al.report.trellis_bits_total,
            odd_body,
            al.report.quantized,
            al.report.per_class[@intFromEnum(alloc.Class.attn)].avg_bits,
            al.report.per_class[@intFromEnum(alloc.Class.gdn)].avg_bits,
            al.report.per_class[@intFromEnum(alloc.Class.mlp)].avg_bits,
            al.report.head_bits,
            &sha_hex,
            &file_hex,
            rp.parts.len,
        },
    );
}

/// lowercase hex of a sha256, printed byte by byte (no float/array formatter paths)
fn hex64(buf: *[64]u8, sha: [32]u8) void {
    const hex = "0123456789abcdef";
    for (sha, 0..) |b, i| {
        buf[i * 2] = hex[b >> 4];
        buf[i * 2 + 1] = hex[b & 15];
    }
}

