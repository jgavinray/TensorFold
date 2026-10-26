//! xpu-eaq-quant — the fresh-quant pipeline CLI (eaq-ldlq-spec-fable.md §5.5).
//!
//!   xpu-eaq-quant SRC OUT --calib rows.i32 [--rows 100] [--cols 2048] [--bits 3]
//!     [--head-bits 4] --seed N [--lambda 0.025] [--scale 1.0] [--layers a..b]
//!     [--save-h DIR] [--proxy] [--allow-identity-fallback] [--work DIR]
//!     [--resume] [--dry-run] [--plan FILE]
//!
//! SRC is the bf16 source directory (the 52 GB Qwen3.8-27B shard set; the driver keeps the
//! 851 non-visual non-mtp tensors); OUT is the pack directory the run publishes last. The
//! pipeline and its checkpointing live in zig/src/xpu/quant.zig; this file only turns the
//! command line into Options and reports the verdict.

const std = @import("std");
const quant = @import("quant.zig");

const usage =
    \\usage: xpu-eaq-quant SRC OUT --calib FILE [options]
    \\  --rows N                 calibration rows (default 100)
    \\  --cols N                 tokens per row (default 2048, max 4096)
    \\  --calib FILE             token ids, rows x cols decimals
    \\  --bits N                 quantized bits (default 3)
    \\  --head-bits N            lm_head bits (default 4)
    \\  --seed N                 run seed for the su/sv draws (required unless --resume)
    \\  --lambda X               damping fraction (default 0.025)
    \\  --scale X                svh gain (default 1.0)
    \\  --layers a..b            layer window (default 0..63)
    \\  --save-h DIR             copy every captured site H here as fp32 K x K
    \\  --proxy                  per-part proxy loss, D-ratio, and (single-layer) T-LDLQ-4
    \\  --allow-identity-fallback  encode through a Cholesky failure (never publishes)
    \\  --work DIR               checkpoint dir (default OUT-work): fragments, H, stream
    \\  --resume                 skip the layers the work dir's progress file records
    \\  --plan FILE              allocator plan JSON (target_bpw, head_halves, method,
    \\                           parts [{base, halves}], plan_sha); default: uniform --bits
    \\  --device-check             probe the card (Level Zero free bytes vs this run's
    \\                             estimate): exit 0 fits, 3 short — the run script's pre-flight
    \\  --dry-run                run the gates, write no pack, record no progress
    \\
;


fn need(argv: []const [:0]const u8, i: *usize) ![]const u8 {
    if (i.* + 1 >= argv.len) return error.Usage;
    i.* += 1;
    return argv[i.*];
}
pub fn main(init: std.process.Init) !void {
    const c = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(c);
    if (argv.len < 3) {
        std.debug.print("{s}", .{usage});
        std.process.exit(2);
    }
    var o: quant.Options = .{
        .src = argv[1],
        .out = argv[2],
    };
    var i: usize = 3;
    while (i < argv.len) : (i += 1) {
        const a = argv[i];
        if (std.mem.eql(u8, a, "--proxy")) {
            o.proxy = true;
        } else if (std.mem.eql(u8, a, "--allow-identity-fallback")) {
            o.allow_identity_fallback = true;
        } else if (std.mem.eql(u8, a, "--resume")) {
            o.resume_run = true;
        } else if (std.mem.eql(u8, a, "--dry-run")) {
            o.dry_run = true;
        } else if (std.mem.eql(u8, a, "--device-check")) {
            o.device_check = true;
        } else if (std.mem.eql(u8, a, "--help") or std.mem.eql(u8, a, "-h")) {
            std.debug.print("{s}", .{usage});
            std.process.exit(0);
        } else {
            const next = need(argv, &i) catch {
                std.debug.print("xpu-eaq-quant: {s} wants a value\n{s}", .{ a, usage });
                std.process.exit(2);
            };
            if (std.mem.eql(u8, a, "--calib")) {
                o.calib_path = next;
            } else if (std.mem.eql(u8, a, "--rows")) {
                o.rows = try std.fmt.parseInt(u32, next, 10);
            } else if (std.mem.eql(u8, a, "--cols")) {
                o.cols = try std.fmt.parseInt(u32, next, 10);
            } else if (std.mem.eql(u8, a, "--bits")) {
                o.bits = try std.fmt.parseInt(u32, next, 10);
            } else if (std.mem.eql(u8, a, "--head-bits")) {
                o.head_bits = try std.fmt.parseInt(u32, next, 10);
            } else if (std.mem.eql(u8, a, "--seed")) {
                o.seed = try std.fmt.parseInt(u64, next, 0); // base 0: decimal or a 0x/0o/0b prefix — the spec's 0xE0 form
                o.seed_given = true;
            } else if (std.mem.eql(u8, a, "--lambda")) {
                o.lambda = try std.fmt.parseFloat(f64, next);
            } else if (std.mem.eql(u8, a, "--scale")) {
                o.scale = try std.fmt.parseFloat(f64, next);
            } else if (std.mem.eql(u8, a, "--layers")) {
                const sep = std.mem.indexOf(u8, next, "..") orelse return error.Usage;
                o.layer_a = try std.fmt.parseInt(usize, next[0..sep], 10);
                o.layer_b = try std.fmt.parseInt(usize, next[sep + 2 ..], 10);
            } else if (std.mem.eql(u8, a, "--save-h")) {
                o.save_h = next;
            } else if (std.mem.eql(u8, a, "--work")) {
                o.work = next;
            } else if (std.mem.eql(u8, a, "--plan")) {
                o.plan_path = next;
            } else {
                std.debug.print("xpu-eaq-quant: unknown option {s}\n{s}", .{ a, usage });
                std.process.exit(2);
            }
        }
    }
    if (o.device_check) {
        std.process.exit(quant.deviceCheck(o) catch 3);
    }
    quant.run(o) catch |e| {
        std.debug.print("xpu-eaq-quant: {s}\n", .{@errorName(e)});
        // the device-memory gate's class (the sysexits ruling): EX_TEMPFAIL, so the run
        // script can retry it as a transient condition instead of burning a crash attempt
        if (e == error.InsufficientDeviceMemory) std.process.exit(75);
        std.process.exit(1);
    };
}
