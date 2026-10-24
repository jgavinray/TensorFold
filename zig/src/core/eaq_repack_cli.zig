//! CLI over core.eaq_repack.repack: turn a stock EXL3 checkpoint directory into an EAQ pack,
//! then prove it with the XPU reader. `tf-eaq-repack <src-dir> <dst-dir>`; a refusal prints
//! the note and exits nonzero (1 repack, 2 openPack, 3 map, 4 incomplete table).
const std = @import("std");
const core = @import("core");
const eaq = core.eaq_container;
const repack = core.eaq_repack;
const r = @import("exl3_eaq");

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3) return error.Usage;

    var why: eaq.Why = .{};
    const got = repack.repack(gpa, io, args[1], args[2], &why) catch |e| {
        std.debug.print("repack refused: {t}: {s}\n", .{ e, why.text() });
        std.process.exit(1);
    };
    std.debug.print(
        "shards {d} (trellis entries {d}, dense entries {d}), entries {d}\n" ++
            "tensors kept {d}, dropped {d}, distinct .mul1 payloads {d}\n" ++
            "manifest sha256 {s}\n",
        .{ got.shards, got.n_trellis, got.n_dense, got.entries, got.kept_tensors, got.dropped_tensors, got.mul1_values, got.manifest_sha256 },
    );

    var pack = r.openPack(gpa, io, args[2], &why) catch |e| {
        std.debug.print("openPack refused: {t}: {s}\n", .{ e, why.text() });
        std.process.exit(2);
    };
    defer pack.deinit();
    var table = r.map(gpa, &pack, &why) catch |e| {
        std.debug.print("map refused: {t}: {s}\n", .{ e, why.text() });
        std.process.exit(3);
    };
    defer table.deinit(gpa);
    std.debug.print(
        "table: {d} projections, head {s}, embed {s}, others {d}, complete {}\n",
        .{ table.projections.len, if (table.head) |h| h.name else "(none)", if (table.embed) |x| x.name else "(none)", table.others.len, table.complete },
    );
    if (!table.complete) std.process.exit(4);
}
