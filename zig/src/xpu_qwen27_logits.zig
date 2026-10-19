//! tf-xpu-qwen27-logits <model dir> <tokens.npy> <out.bf16>: teacher-forced logits of Qwen 3.8 27B on a B70.
//! Writes len(tokens) x vocab bf16 words (raw, little-endian) to <out.bf16>, one row per input token, in order.
//! Tokens come from a .npy of uint32 (the fixture format the gates use); every id must be < vocab.
//! The forward is families/qwen27/xpu_forward.zig with head = .all, prefilled in `chunk`-row pieces as the server
//! prefills (any length: the state capacity is sized to the input), so the row of token i is the logits that predict
//! token i+1 (the teacher-forced layout the reference uses).
//! Exit status 0 only when the file was written; errors go to stderr with a nonzero status.
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

/// Rows per forward: the server's prefill chunk. A single forward over thousands of rows would size every scratch
/// buffer (attention partials: rows x heads x chunks x head_dim fp32) by the whole input.
const chunk = 128;

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

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    if (argv.len != 4) {
        std.debug.print("usage: tf-xpu-qwen27-logits <model dir> <tokens.npy> <out.bf16>\n", .{});
        return 2;
    }
    const dir = argv[1];
    const ids = try readNpyU32(argv[2]);
    const out_path = argv[3];
    if (ids.len == 0) {
        std.debug.print("no tokens in {s}\n", .{argv[2]});
        return 1;
    }

    var s: rt.session.Session = undefined;
    s.open(.{}) catch |e| {
        std.debug.print("cannot open a card: {s} ({s})\n", .{ @errorName(e), s.diag.message() });
        return 1;
    };
    defer s.close();
    std.debug.print("card {s} ({s})\n", .{ s.bdf(), s.ctx.dev.name() });

    const config = loadConfig(dir) catch |e| {
        std.debug.print("config error: {s}\n", .{@errorName(e)});
        return 1;
    };
    std.debug.print("config: hidden {d}, layers {d}, heads {d}, kv {d}, intermediate {d}, group {d}\n", .{ config.hidden, config.layers, config.heads, config.kv_heads, config.intermediate, config.group });
    var w = try xw.load(gpa, &s, dir, config);
    defer w.deinit();
    std.debug.print("weights loaded: {d} tensors, {d:.1} MiB\n", .{ w.ledger.len, @as(f64, @floatFromInt(w.total_bytes)) / (1 << 20) });
    var m: fwd.Model = .{ .weights = &w, .kernels = undefined };
    try m.kernels.load(&s, gpa);
    defer m.kernels.deinit();
    std.debug.print("kernels loaded\n", .{});

    const rows = ids.len;
    // The KV/recurrence state holds the whole input; at least 512 as before.
    const capacity = @max(512, (rows + 127) / 128 * 128);
    const step = @min(rows, chunk);
    std.debug.print("scratch: rows {d}, capacity {d}\n", .{ step, capacity });
    var sc = try fwd.Scratch.init(&s, config, step, capacity, step);
    defer sc.deinit();
    std.debug.print("scratch ok ({d} buffers)\n", .{sc.made});
    var cache = try xs.Cache.init(gpa, &s, config, capacity);
    defer cache.deinit();
    std.debug.print("cache ok ({d} buffers)\n", .{cache.bufferCount()});

    // Every row's bf16 logits, then one raw file of len(tokens) x vocab words.
    const words = rows * config.vocab;
    const logits = try gpa.alloc(u16, words);
    defer gpa.free(logits);
    const t0 = sys.nowNs();
    std.debug.print("forward...\n", .{});
    var at: usize = 0;
    while (at < rows) : (at += step) {
        const end = @min(rows, at + step);
        try fwd.run(&s, &m, &sc, &cache, ids[at..end], .{ .head = .all });
        try sc.logits.download(&s.queue, 0, std.mem.sliceAsBytes(logits[at * config.vocab .. end * config.vocab]));
    }
    std.debug.print("forward {d} tokens in {d:.2} s ({d:.2} tokens/s teacher-forced)\n", .{
        rows,
        @as(f64, @floatFromInt(sys.nowNs() - t0)) / 1e9,
        @as(f64, @floatFromInt(rows)) / (@as(f64, @floatFromInt(sys.nowNs() - t0)) / 1e9),
    });

    var finite = true;
    for (logits) |w16| {
        const f: f32 = @bitCast(@as(u32, w16) << 16);
        if (!std.math.isFinite(f)) {
            finite = false;
            break;
        }
    }
    std.debug.print("logits: {d} words ({d} x {d}), finite={}\n", .{ words, rows, config.vocab, finite });

    var wb: [1024]u8 = undefined;
    const path = sys.zfmt(&wb, "{s}", .{out_path}) orelse return error.PathTooLong;
    if (!sys.writeFile(path, std.mem.sliceAsBytes(logits))) {
        std.debug.print("cannot write {s}\n", .{out_path});
        return 1;
    }
    return 0;
}
