//! Modules from ocloc native binaries (ze_module_format NATIVE) and their kernels.
const std = @import("std");
const abi = @import("abi.zig");
const runtime = @import("runtime.zig");
const Context = @import("context.zig").Context;

pub const Module = struct {
    ctx: *const Context,
    handle: abi.Module,

    /// `binary` is the output of `ocloc compile -device bmg`. On a build failure the driver's log is copied to
    /// `log` (when given) and ZeFailed is returned.
    pub fn load(ctx: *const Context, binary: []const u8, log: ?*BuildLog) runtime.Error!Module {
        if (binary.len == 0) return error.Invalid;
        var h: abi.Module = null;
        var blog: abi.BuildLog = null;
        const desc: abi.ModuleDesc = .{ .format = abi.module_format_native, .inputSize = binary.len, .pInputModule = binary.ptr };
        const rc = ctx.api().zeModuleCreate(ctx.handle, ctx.dev.handle, &desc, &h, &blog);
        defer if (blog != null) {
            _ = ctx.api().zeModuleBuildLogDestroy(blog);
        };
        if (log) |l| {
            l.len = 0;
            if (blog != null) {
                var n: usize = l.buf.len;
                if (ctx.api().zeModuleBuildLogGetString(blog, &n, &l.buf) == abi.success) l.len = @min(n, l.buf.len);
            }
        }
        try runtime.check(rc);
        if (h == null) return error.Invalid;
        return .{ .ctx = ctx, .handle = h };
    }

    pub fn unload(self: *Module) void {
        _ = self.ctx.api().zeModuleDestroy(self.handle);
        self.* = undefined;
    }

    pub fn kernel(self: *const Module, name: [:0]const u8) runtime.Error!Kernel {
        var k: abi.Kernel = null;
        const desc: abi.KernelDesc = .{ .pKernelName = name.ptr };
        try runtime.check(self.ctx.api().zeKernelCreate(self.handle, &desc, &k));
        if (k == null) return error.Invalid;
        return .{ .ctx = self.ctx, .handle = k };
    }
};

pub const BuildLog = struct {
    buf: [4096]u8 = undefined,
    len: usize = 0,
    pub fn text(self: *const BuildLog) []const u8 {
        return self.buf[0..self.len];
    }
};

pub const Kernel = struct {
    ctx: *const Context,
    handle: abi.Kernel,

    pub fn deinit(self: *Kernel) void {
        _ = self.ctx.api().zeKernelDestroy(self.handle);
        self.* = undefined;
    }

    /// Set argument `index` from a value of the kernel's C parameter type (pointers: pass Buffer.ptr).
    pub fn setArg(self: *Kernel, index: u32, value: anytype) runtime.Error!void {
        const T = @TypeOf(value);
        comptime std.debug.assert(@sizeOf(T) != 0);
        try self.setArgBytes(index, std.mem.asBytes(&value));
    }

    pub fn setArgBytes(self: *Kernel, index: u32, bytes: []const u8) runtime.Error!void {
        try runtime.check(self.ctx.api().zeKernelSetArgumentValue(self.handle, index, bytes.len, bytes.ptr));
    }

    pub fn setGroupSize(self: *Kernel, x: u32, y: u32, z: u32) runtime.Error!void {
        if (x == 0 or y == 0 or z == 0) return error.Invalid;
        try runtime.check(self.ctx.api().zeKernelSetGroupSize(self.handle, x, y, z));
    }
};
