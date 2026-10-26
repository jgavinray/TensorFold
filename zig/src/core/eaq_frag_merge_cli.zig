//! tf-eaq-frag-merge: the mixed half-step pack (docs/eaq-review/eaq-halfstep-spec-fable.md
//! 1, 5 item 4) — per-layer trellis fragments merged over a full-width dense source into a
//! complete EAQ pack, host-only: the quantizer and the card are never touched, and the
//! repacker and the container are not modified.
//!
//!   tf-eaq-frag-merge <src-dir> <out-dir> --frag-dir DIR [--frag-dir DIR]... [--config FILE]
//!
//! <src-dir> is the stock source directory the quantizer itself reads (config.json,
//! model.safetensors.index.json and the model*.safetensors shards): the dense side —
//! embeds, norms, conv1d, A_log, dt_bias, the a/b gates — is taken from it exactly as the
//! quantizer's publish takes it, the same drop rules and the same BF16->F16 gate
//! conversion, mirrored from zig/src/xpu/quant.zig's denseRegistry/quantizedName/isKept/
//! packName/bf16ToF16 (cited at each) and read through the host checkpoint reader
//! (core/checkpoint.zig, the one exl3_load scans with). --frag-dir holds
//! frag_layer<N>.safetensors fragments (the quantizer's own work-dir shape); repeat it to
//! mix populations, e.g. the halves-5 work dir first and the step-4 3.0 work dir second:
//! dirs are scanned in order and the FIRST fragment found for a layer wins; a shadowed
//! copy is compared by size and reported — a shadow that differs warns loudly, the
//! driver's S5 refusal to ship a fragment whose bytes moved, in spirit. All 64 layers plus
//! a frag_head.safetensors are required; the head fragment replaces the dense
//! lm_head.weight, as the driver's publish does.
//!
//! The writer runs with shards = 8 (the stock convention the reader and the repacker use)
//! and allow_odd_bits = true (T-HS-10 always passes: halves 5 is the point of the pack,
//! and the step-5 loader admits the odd counts). On success the Result line prints, then
//! the reader's own openEaq, map and Table close the loop as tf-eaq-repack's proof does
//! (eaq_repack_cli.zig:28-42), and core.exl3_load.scan prints the admission histogram in
//! the serve's own lines (xpu/load.zig:510-512) — a refused group, a Table that is not
//! complete, or a halves-5 count that is not 26 exits nonzero
//! (1 merge/write, 2 openPack, 3 map, 4 incomplete Table, 5 histogram, 6 degenerate trellis).

const std = @import("std");
const core = @import("core");
const eaq = core.eaq_container;
const eaq_write = @import("eaq_write"); // a module root (xpu_quant.zig:115), never a file of core
const exl3_eaq = @import("exl3_eaq");
const Checkpoint = core.checkpoint.Checkpoint;
const Io = std.Io;

const LAYERS = 64;
const head_name = "lm_head.weight";

fn quantizedName(a: std.mem.Allocator, i: usize, s: []const u8) bool { // quant.zig:516-531, the same rule
    const p = std.fmt.allocPrint(a, "model.language_model.layers.{d}.", .{i}) catch return false;
    const tails = [_][]const u8{
        "self_attn.q_proj.weight",     "self_attn.k_proj.weight",        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",     "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
        "linear_attn.out_proj.weight", "mlp.gate_proj.weight",           "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    };
    for (tails) |t| {
        const full = std.fmt.allocPrint(a, "{s}{s}", .{ p, t }) catch return false;
        if (std.mem.eql(u8, s, full)) return true;
    }
    return false;
}

fn isKept(name: []const u8) bool { // quant.zig:533-535
    return !(std.mem.startsWith(u8, name, "model.visual.") or std.mem.startsWith(u8, name, "mtp."));
}

fn packName(a: std.mem.Allocator, name: []const u8) ![]const u8 { // quant.zig:244-250
    if (std.mem.startsWith(u8, name, "model.language_model.layers."))
        return std.fmt.allocPrint(a, "model.layers.{s}", .{name["model.language_model.layers.".len..]});
    if (std.mem.eql(u8, name, "model.language_model.embed_tokens.weight")) return a.dupe(u8, "model.embed_tokens.weight");
    if (std.mem.eql(u8, name, "model.language_model.norm.weight")) return a.dupe(u8, "model.norm.weight");
    return a.dupe(u8, name);
}

/// quant.zig:338-344 verbatim. The writer gates the a/b payloads for F16 and this CLI is
/// the only converter on the merge path; duplicating the rounding rule here is the house
/// cite-the-source pattern, and the pack's decode through the reader's admission is the
/// guard against drift.
fn bf16ToF16(src: []const u8, out: []u16) void {
    for (out, 0..) |*o, e| {
        // bf16 is the top half of fp32: expand by shifting left, then narrow (the twin of the kernel-side bf2f)
        const bits: u32 = @as(u32, src[2 * e]) | (@as(u32, src[2 * e + 1]) << 8);
        o.* = @bitCast(@as(f16, @floatCast(@as(f32, @bitCast(bits << 16)))));
    }
}

const Frag = struct { path: []const u8, size: u64 };

fn statSize(io: Io, path: []const u8) ?u64 {
    const st = Io.Dir.cwd().statFile(io, path, .{}) catch return null;
    if (st.kind != .file) return null;
    return st.size;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 4) return error.Usage;
    const src_dir = args[1];
    const out_dir = args[2];
    var frag_dirs: std.ArrayList([]const u8) = .empty;
    var cfg_path: ?[]const u8 = null;
    var ai: usize = 3;
    while (ai < args.len) : (ai += 1) {
        if (std.mem.eql(u8, args[ai], "--frag-dir")) {
            ai += 1;
            if (ai >= args.len) return error.Usage;
            try frag_dirs.append(a, args[ai]);
        } else if (std.mem.eql(u8, args[ai], "--config")) {
            ai += 1;
            if (ai >= args.len) return error.Usage;
            cfg_path = args[ai];
        } else return error.Usage;
    }
    if (frag_dirs.items.len == 0) return error.Usage;

    // ---- the dense side: the stock directory's shards mapped through the host checkpoint
    // reader, then the kept-851 gate (quant.zig:90, openSource's count at 2603-2610)
    var dense = Checkpoint.openModel(gpa, io, src_dir) catch |e| {
        std.debug.print("tf-eaq-frag-merge: {s}: {t} — the source must be the stock directory the quantizer reads\n", .{ src_dir, e });
        std.process.exit(1);
    };
    defer dense.close();
    {
        var kept: usize = 0;
        for (dense.files.items) |*f| for (f.names.keys()) |n| {
            if (isKept(n)) kept += 1;
        };
        if (kept != 851) {
            std.debug.print("tf-eaq-frag-merge: {s} keeps {d} tensors, expected 851 (850 language_model + lm_head after the visual/mtp drop)\n", .{ src_dir, kept });
            std.process.exit(1);
        }
    }

    // ---- fragment placement: dirs in order, first hit wins; a shadowed copy is compared
    // by size — an equal size stays silent, a different one warns loudly
    var frags: std.ArrayList(Frag) = .empty;
    try frags.ensureTotalCapacity(a, LAYERS);
    {
        var li: usize = 0;
        while (li < LAYERS) : (li += 1) {
            const nm = try std.fmt.allocPrint(a, "frag_layer{d}.safetensors", .{li});
            var chosen: ?Frag = null;
            for (frag_dirs.items) |fd| {
                const p = try std.fs.path.join(a, &.{ fd, nm });
                const sz = statSize(io, p) orelse continue;
                if (chosen) |c| {
                    if (c.size != sz) std.debug.print("tf-eaq-frag-merge: layer {d}: using {s} ({d} bytes); shadowed by a DIFFERENT-SIZED copy ({d} bytes) at {s}\n", .{ li, c.path, c.size, sz, p });
                    continue;
                }
                try frags.append(a, .{ .path = p, .size = sz });
                chosen = .{ .path = p, .size = sz };
            }
            if (chosen == null) {
                std.debug.print("tf-eaq-frag-merge: layer {d}: no {s} in any --frag-dir — every layer needs a fragment\n", .{ li, nm });
                std.process.exit(1);
            }
        }
    }
    var head_path: ?[]const u8 = null;
    for (frag_dirs.items) |fd| {
        const p = try std.fs.path.join(a, &.{ fd, "frag_head.safetensors" });
        if (statSize(io, p) != null) {
            head_path = p;
            break;
        }
    }
    if (head_path == null) {
        std.debug.print("tf-eaq-frag-merge: no frag_head.safetensors in any --frag-dir — the pack needs the trellis head; the dense head is no substitute for this writer\n", .{});
        std.process.exit(1);
    }

    // ---- open the fragments the way publish does (quant.zig:2211-2244): one checkpoint
    // per layer plus the head, mapped for the whole life of the write
    var fcks: std.ArrayList(Checkpoint) = .empty;
    for (frags.items) |f| {
        const dir = std.fs.path.dirname(f.path).?;
        const name = std.fs.path.basename(f.path);
        var ck: Checkpoint = .{ .gpa = gpa, .io = io };
        ck.add(dir, name) catch |e| {
            std.debug.print("tf-eaq-frag-merge: {s}: {t}\n", .{ f.path, e });
            std.process.exit(1);
        };
        try fcks.append(a, ck);
    }
    var hck: Checkpoint = .{ .gpa = gpa, .io = io };
    hck.add(std.fs.path.dirname(head_path.?).?, std.fs.path.basename(head_path.?)) catch |e| {
        std.debug.print("tf-eaq-frag-merge: {s}: {t}\n", .{ head_path.?, e });
        std.process.exit(1);
    };

    // ---- degenerate smoke (the l0fix lesson): a halves-5 trellis whose every word is
    // one value never closed the codebook — refuse before staging rather than ship it
    for (fcks.items, 0..) |*ck, li| for (ck.files.items) |*file| for (file.names.keys()) |n| {
        if (!std.mem.endsWith(u8, n, ".trellis")) continue;
        const t = file.get(n).?;
        if (t.rank != 3 or t.shape[2] % 16 != 8) continue; // only the odd-halves parts are the new content
        const words: []const i16 = @alignCast(@ptrCast(t.bytes[0 .. t.bytes.len & ~@as(usize, 1)]));
        if (words.len == 0) continue;
        var same = true;
        for (words) |w| if (w != words[0]) {
            same = false;
            break;
        };
        if (same) {
            std.debug.print("tf-eaq-frag-merge: layer {d}: {s} is one repeated int16 word — a degenerate trellis, the pack refuses\n", .{ li, n });
            std.process.exit(6);
        }
    };

    // ---- inputs, the publish order: dense (source order) then the fragments, head last
    var items: std.ArrayList(eaq_write.Input) = .empty;
    for (dense.files.items) |*file| for (file.names.keys()) |n| {
        if (!isKept(n)) continue;
        if (std.mem.eql(u8, n, head_name)) continue; // replaced by the trellis head
        var qn = false;
        for (0..LAYERS) |i| if (quantizedName(a, i, n)) {
            qn = true;
            break;
        };
        if (qn) continue;
        const t = file.get(n).?;
        const gate = std.mem.endsWith(u8, n, "linear_attn.in_proj_a.weight") or
            std.mem.endsWith(u8, n, "linear_attn.in_proj_b.weight");
        const convert = gate and t.dtype == .bf16;
        const payload: []const u8 = if (convert) blk: {
            const out = try a.alloc(u16, t.bytes.len / 2);
            bf16ToF16(t.bytes, out);
            break :blk std.mem.sliceAsBytes(out);
        } else t.bytes;
        try items.append(a, .{
            .name = try packName(a, n),
            .dtype = if (convert) .f16 else t.dtype,
            .shape = try a.dupe(usize, t.shape[0..t.rank]),
            .payload = payload,
        });
    };
    for (fcks.items) |*ck| for (ck.files.items) |*file| for (file.names.keys()) |n| {
        const t = file.get(n).?;
        try items.append(a, .{
            .name = n,
            .dtype = t.dtype,
            .shape = try a.dupe(usize, t.shape[0..t.rank]),
            .payload = t.bytes,
        });
    };
    for (hck.files.items) |*file| for (file.names.keys()) |n| {
        const t = file.get(n).?;
        try items.append(a, .{
            .name = n,
            .dtype = t.dtype,
            .shape = try a.dupe(usize, t.shape[0..t.rank]),
            .payload = t.bytes,
        });
    };

    const cfg: ?[]const u8 = if (cfg_path) |cp|
        try Io.Dir.cwd().readFileAlloc(io, cp, a, .limited(1 << 20))
    else blk: {
        const jp = try std.fs.path.join(a, &.{ src_dir, "config.json" });
        break :blk try Io.Dir.cwd().readFileAlloc(io, jp, a, .limited(1 << 20));
    };

    std.debug.print("tf-eaq-frag-merge: {d} inputs, fragments from {d} dirs, writing {s}\n", .{ items.items.len, frag_dirs.items.len, out_dir });
    var why: eaq.Why = .{};
    const got = eaq_write.write(gpa, io, out_dir, items.items, .{
        .config = cfg,
        .shards = 8,
        .allow_odd_bits = true,
    }, &why) catch |e| {
        std.debug.print("tf-eaq-frag-merge: write refused: {t}: {s}\n", .{ e, why.text() });
        std.process.exit(1);
    };
    std.debug.print(
        "shards {d} (trellis entries {d}, dense entries {d}), entries {d}\nmanifest sha256 {s}\n",
        .{ got.shards, got.n_trellis, got.n_dense, got.entries, got.manifest_sha256 },
    );

    // ---- the reader's own gate, as tf-eaq-repack proves (eaq_repack_cli.zig:28-42)
    var pack = exl3_eaq.openPack(gpa, io, out_dir, &why) catch |e| {
        std.debug.print("openPack refused: {t}: {s}\n", .{ e, why.text() });
        std.process.exit(2);
    };
    defer pack.deinit();
    var table = exl3_eaq.map(gpa, &pack, &why) catch |e| {
        std.debug.print("map refused: {t}: {s}\n", .{ e, why.text() });
        std.process.exit(3);
    };
    defer table.deinit(gpa);
    std.debug.print(
        "table: {d} projections, head {s}, embed {s}, others {d}, complete {}\n",
        .{ table.projections.len, if (table.head) |h| h.name else "(none)", if (table.embed) |x| x.name else "(none)", table.others.len, table.complete },
    );
    if (!table.complete) std.process.exit(4);

    // ---- the admission histogram, in the serve's own lines (xpu/load.zig:510-512)
    var sc = try core.exl3_load.scan(gpa, io, out_dir);
    defer sc.deinit();
    std.debug.print("  exl3: {d} groups admitted (mul1 {d}, inst3 {d}, mcg {d}): scales suh/svh {d}, su/sv {d}, bias {d}\n", .{ sc.summary.groups, sc.summary.mul1, sc.summary.inst3, sc.summary.mcg, sc.summary.suh_svh, sc.summary.su_sv, sc.summary.bias });
    std.debug.print("  bits:", .{}); // the histogram keys halves; print the bits each index means (2.5b style, one decimal)
    for (sc.summary.bits, 0..) |n, h| if (n != 0) std.debug.print(" {d}.{d}b:{d}", .{ h / 2, (h % 2) * 5, n });
    std.debug.print("\n", .{});
    if (sc.bad != 0 or sc.summary.bits[5] != 26) {
        std.debug.print("tf-eaq-frag-merge: admission refuses {d} trellises and the halves-5 count is {d}, expected 26 — not the T-HS-10 pack\n", .{ sc.bad, sc.summary.bits[5] });
        std.process.exit(5);
    }
    std.debug.print("tf-eaq-frag-merge: {s} opens, Table complete, 2.5b:26 admitted\n", .{out_dir});
}
