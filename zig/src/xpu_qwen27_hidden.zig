//! tf-xpu-qwen27-hidden <model dir> <tokens.npy> <positions> <out.bf16>: Qwen 3.8 27B per-layer residual stream at
//! chosen positions, teacher-forced in 128-row prefill chunks as the server prefills (diagnostics: compare layer by
//! layer with an MLX reference, exp/ref/mlx_positions.py). <positions> is comma separated. Writes, per position in
//! the order given and per layer, the residual h and the layer's MLP output r (bf16, hidden each): the layer output
//! is h + r. Layout: [position][layer][h | r][hidden].
const std = @import("std");
const fwd = @import("qwen27_forward");
const rt = fwd.rt;
const sys = fwd.sys;
const cfg = fwd.cfg;
const xw = fwd.xw;
const xs = fwd.xs;

const gpa = std.heap.page_allocator;
const chunk = 128;

fn readNpyU32(path: []const u8) ![]u32 {
    var pb: [1024]u8 = undefined;
    const zp = sys.zfmt(&pb, "{s}", .{path}) orelse return error.PathTooLong;
    const bytes = sys.readFileAlloc(gpa, zp, 1 << 30) orelse return error.NoFile;
    defer gpa.free(bytes);
    if (bytes.len < 10 or !std.mem.eql(u8, bytes[0..6], "\x93NUMPY")) return error.BadNpy;
    const hl = std.mem.readInt(u16, bytes[8..10], .little);
    const data = bytes[10 + hl ..];
    const out = try gpa.alloc(u32, data.len / 4);
    @memcpy(std.mem.sliceAsBytes(out), data[0 .. out.len * 4]);
    return out;
}

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    if (argv.len != 5) {
        std.debug.print("usage: tf-xpu-qwen27-hidden <model dir> <tokens.npy> <positions> <out.bf16>\n", .{});
        return 2;
    }
    const ids = try readNpyU32(argv[2]);
    var picks: std.ArrayList(usize) = .empty;
    var it = std.mem.splitScalar(u8, argv[3], ',');
    while (it.next()) |t| try picks.append(arena, try std.fmt.parseInt(usize, t, 10));
    for (picks.items) |p| if (p >= ids.len) return error.PositionPastInput;

    var s: rt.session.Session = undefined;
    s.open(.{}) catch |e| {
        std.debug.print("cannot open a card: {s} ({s})\n", .{ @errorName(e), s.diag.message() });
        return 1;
    };
    defer s.close();
    var cb: [1024]u8 = undefined;
    const cpath = sys.zfmt(&cb, "{s}/config.json", .{argv[1]}) orelse return error.PathTooLong;
    const text = sys.readFileAlloc(gpa, cpath, 1 << 22) orelse return error.NoConfig;
    const config = try cfg.parse(gpa, text, null);
    var w = try xw.load(gpa, &s, argv[1], config);
    defer w.deinit();
    var m: fwd.Model = .{ .weights = &w, .kernels = undefined };
    try m.kernels.load(&s, gpa);
    defer m.kernels.deinit();
    const capacity = @max(512, (ids.len + 127) / 128 * 128);
    var sc = try fwd.Scratch.init(&s, config, chunk, capacity, 1);
    defer sc.deinit();
    var cache = try xs.Cache.init(gpa, &s, config, capacity);
    defer cache.deinit();

    const per = config.layers * 2 * config.hidden; // bf16 words per position
    var tapbuf = try rt.memory.Buffer.alloc(&s.ctx, .device, picks.items.len * per * 2);
    defer tapbuf.free();
    const host = try gpa.alloc(u16, picks.items.len * per);
    defer gpa.free(host);
    var at: usize = 0;
    while (at < ids.len) : (at += chunk) {
        const end = @min(ids.len, at + chunk);
        var rows: std.ArrayList(usize) = .empty;
        var which: std.ArrayList(usize) = .empty;
        for (picks.items, 0..) |p, k| if (p >= at and p < end) {
            try rows.append(arena, p - at);
            try which.append(arena, k);
        };
        try fwd.run(&s, &m, &sc, &cache, ids[at..end], .{ .taps = if (rows.items.len > 0) .{ .rows = rows.items, .out = tapbuf.ptr.? } else null });
        if (rows.items.len > 0) {
            const tmp = try gpa.alloc(u16, rows.items.len * per);
            defer gpa.free(tmp);
            try tapbuf.download(&s.queue, 0, std.mem.sliceAsBytes(tmp));
            // tap layout in the forward: [layer][tap][h | r][hidden]; output: [position][layer][h | r][hidden]
            for (which.items, 0..) |k, tpi| for (0..config.layers) |li| {
                const src = tmp[((li * rows.items.len + tpi) * 2) * config.hidden ..][0 .. 2 * config.hidden];
                @memcpy(host[(k * config.layers + li) * 2 * config.hidden ..][0 .. 2 * config.hidden], src);
            };
        }
    }
    var ob: [1024]u8 = undefined;
    const op = sys.zfmt(&ob, "{s}", .{argv[4]}) orelse return error.PathTooLong;
    if (!sys.writeFile(op, std.mem.sliceAsBytes(host))) return 1;
    std.debug.print("wrote {d} positions x {d} layers\n", .{ picks.items.len, config.layers });
    return 0;
}
