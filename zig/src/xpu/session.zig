//! One job on one card: lock -> preflight -> runtime -> device -> context -> in-order queue, torn down in reverse.
const std = @import("std");
const runtime = @import("runtime.zig");
const device = @import("device.zig");
const lock = @import("lock.zig");
const preflight = @import("preflight.zig");
const Context = @import("context.zig").Context;
const Queue = @import("queue.zig").Queue;

pub const Options = struct {
    lock_dir: []const u8 = lock.default_dir,
    /// Index into device.cards, tried first.
    prefer: ?usize = null,
    /// Index into device.cards: use exactly this card (waits for it) instead of any free one.
    only: ?usize = null,
    wait_ms: u64 = 3_600_000,
    loader: []const u8 = runtime.loader_name,
    paths: preflight.Paths = .{},
};

pub const OpenError = runtime.Error || lock.Error || preflight.Refusal;

/// Heap-free but address-stable: open() fills `self` in place (Context and Queue point into it).
pub const Session = struct {
    held: lock.Held,
    rt: runtime.Runtime,
    ctx: Context,
    queue: Queue,
    diag: preflight.Diag,

    pub fn open(self: *Session, opts: Options) OpenError!void {
        self.diag = .{};
        self.held = try lock.acquireOfPaths(opts.lock_dir, opts.prefer, opts.only, opts.wait_ms, opts.paths);
        errdefer self.held.release();
        try preflight.check(opts.paths, self.held.bdf(), &self.diag);
        self.rt = try runtime.Runtime.openPath(opts.loader);
        errdefer self.rt.close();
        const dev = try device.select(&self.rt, self.held.bdf());
        self.ctx = try Context.init(dev);
        errdefer self.ctx.deinit();
        self.queue = try Queue.init(&self.ctx);
    }

    pub fn close(self: *Session) void {
        self.queue.deinit();
        self.ctx.deinit();
        self.rt.close();
        self.held.release();
    }

    pub fn bdf(self: *const Session) []const u8 {
        return self.held.bdf();
    }
};
