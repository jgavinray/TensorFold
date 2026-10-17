//! tf-xpu-qwen27-greedy <model dir> <tokens.npy> <n>: prints n greedy token ids of Qwen 3.8 27B on a B70,
//! whitespace separated, on stdout. The prompt comes from a .npy of uint32; each generated token is the argmax of the
//! previous row's bf16 logits (ties go to the lowest id, matching the fixture's argmax convention), appended to the
//! cache as a one-row segment. n counts generated tokens, not prompt tokens.
//! Exit status 0 only when n tokens were produced; errors go to stderr with a nonzero status.
const std = @import("std");
// "qwen27_forward" (zig/src/xpu_qwen27_forward.zig) re-exports the runtime, config, weights and state modules,
// so the tools share one instance of every file instead of importing them a second time.
const fwd = @import("qwen27_forward");
const rt = fwd.rt;
const sys = fwd.sys;
const cfg = fwd.cfg;
const xw = fwd.xw;
const xs = fwd.xs;

const gpa = std.heap.page_allocator;

/// Rows the state is sized for: prompt length + n is bounded by this; the runtime lock makes one big allocation cheap.
const capacity = 512;

fn readNpyU32(path: [*:0]const u8) ![]u32 {
    const bytes = sys.readFileAlloc(gpa, path, 1 << 30) orelse {
        std.debug.print("cannot read {s}\n", .{path});
        return error.NoFile;
    };
    defer gpa.free(bytes);
    if (bytes.len < 10 or !std.mem.eql(u8, bytes[0..6], "\x93NUMPY")) return error.BadNpy;
    const hl = std.mem.readInt(u16, bytes[8..10], .little);
    if (10 + hl > bytes.len) return error.BadNpy;
    const data = bytes[10 + hl ..];
    if (data.len % 4 != 0) return error.BadNpy;
    const out = try gpa.alloc(u32, data.len / 4);
    for (out, 0..) |*o, i| o.* = std.mem.readInt(u32, data[i * 4 ..][0..4], .little);
    return out;
}

fn loadGroup(dir: []const u8) !usize {
    var cb: [1024]u8 = undefined;
    const cpath = sys.zfmt(&cb, "{s}/config.json", .{dir}) orelse return error.PathTooLong;
    const text = sys.readFileAlloc(gpa, cpath, 1 << 22) orelse return error.NoConfig;
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const root = switch (parsed.value) {
        .object => |o| o,
        else => return error.BadConfig,
    };
    for ([_][]const u8{ "quantization", "quantization_config" }) |key| {
        const v = root.get(key) orelse continue;
        if (v != .object) continue;
        if (v.object.get("group_size")) |gs| {
            if (gs != .integer or gs.integer < 1) return error.BadAffine;
            return @intCast(gs.integer);
        }
    }
    return error.MissingAffineFormat;
}

fn loadConfig(dir: []const u8) !cfg.Config {
    var cb: [1024]u8 = undefined;
    const cpath = sys.zfmt(&cb, "{s}/config.json", .{dir}) orelse return error.PathTooLong;
    const text = sys.readFileAlloc(gpa, cpath, 1 << 22) orelse return error.NoConfig;
    defer gpa.free(text);
    var gen: ?[]u8 = null;
    var gb: [1024]u8 = undefined;
    if (sys.zfmt(&gb, "{s}/generation_config.json", .{dir})) |gp| gen = sys.readFileAlloc(gpa, gp, 1 << 22);
    defer if (gen) |g| gpa.free(g);
    return cfg.parse(gpa, text, gen);
}

fn bf(w: u16) f32 {
    return @bitCast(@as(u32, w) << 16);
}

/// Argmax over a bf16 row; ties keep the lower id (strict >), as the fixtures' argmax does.
fn argmax(row: []const u16) usize {
    var best: usize = 0;
    var bv = bf(row[0]);
    for (row, 0..) |w, i| {
        const v = bf(w);
        if (v > bv) {
            bv = v;
            best = i;
        }
    }
    return best;
}

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    if (argv.len != 4) {
        std.debug.print("usage: tf-xpu-qwen27-greedy <model dir> <tokens.npy> <n>\n", .{});
        return 2;
    }
    const dir = argv[1];
    const ids = try readNpyU32(argv[2]);
    const n = std.fmt.parseInt(usize, argv[3], 10) catch {
        std.debug.print("bad n {s}\n", .{argv[3]});
        return 2;
    };
    if (ids.len == 0 or n == 0) {
        std.debug.print("need a non-empty prompt and n >= 1\n", .{});
        return 1;
    }
    if (ids.len + n > capacity) {
        std.debug.print("prompt {d} + {d} tokens exceeds the {d}-row state\n", .{ ids.len, n, capacity });
        return 1;
    }

    var s: rt.session.Session = undefined;
    s.open(.{}) catch |e| {
        std.debug.print("cannot open a card: {s} ({s})\n", .{ @errorName(e), s.diag.message() });
        return 1;
    };
    defer s.close();

    const config = loadConfig(dir) catch |e| {
        std.debug.print("config error: {s}\n", .{@errorName(e)});
        return 1;
    };
    const group = loadGroup(dir) catch |e| {
        std.debug.print("quantization error: {s}\n", .{@errorName(e)});
        return 1;
    };
    var w = try xw.load(gpa, &s, dir, config, group);
    defer w.deinit();
    var m: fwd.Model = .{ .weights = &w, .kernels = undefined };
    try m.kernels.load(&s, gpa);
    defer m.kernels.deinit();

    // The scratch must hold the whole prompt in one prefill segment (rows = prompt length); the one-row decode
    // steps reuse the same buffers, since `run` only needs rows >= that call's token count. `logit_rows` stays 1
    // because every call here asks for the last row only.
    var sc = try fwd.Scratch.init(&s, config, ids.len, capacity, 1);
    defer sc.deinit();
    var cache = try xs.Cache.init(gpa, &s, config, capacity);
    defer cache.deinit();

    // Prefill the prompt in one segment, then one row per generated token.
    try fwd.run(&s, &m, &sc, &cache, ids, .{});
    const row = try gpa.alloc(u16, config.vocab);
    defer gpa.free(row);

    var out = std.ArrayList(u8).empty;
    defer out.deinit(gpa);
    var next: [1]u32 = undefined;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        try cache.logits.download(&s.queue, 0, std.mem.sliceAsBytes(row));
        const t: u32 = @intCast(argmax(row));
        const sep: u8 = if (i + 1 == n) '\n' else ' ';
        const piece = try std.fmt.allocPrint(gpa, "{d}{c}", .{ t, sep });
        defer gpa.free(piece);
        try out.appendSlice(gpa, piece);
        if (i + 1 == n) break;
        next[0] = t;
        try fwd.run(&s, &m, &sc, &cache, &next, .{});
    }
    try std.Io.File.stdout().writeStreamingAll(init.io, out.items);
    return 0;
}
