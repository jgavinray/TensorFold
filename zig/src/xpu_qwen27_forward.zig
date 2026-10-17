//! The Qwen 3.8 27B XPU forward's module root: a thin re-export so every file under zig/src/ belongs to exactly one
//! module. The forward itself lives in families/qwen27/xpu_forward.zig; this file exists only so the build can set a
//! module root at zig/src/ (module paths may not escape their root, and the forward imports ../../xpu/** and
//! ../qwen3_5/xpu_geometry.zig). The tools and any test binary import "qwen27_forward" and see the forward plus the
//! runtime, config, weights and state modules it re-exports.
pub const forward = @import("families/qwen27/xpu_forward.zig");
pub const rt = forward.rt;
pub const sys = forward.sys;
pub const cfg = forward.cfg;
pub const xw = forward.xw;
pub const xs = forward.xs;
pub const Model = forward.Model;
pub const Scratch = forward.Scratch;
pub const Options = forward.Options;
pub const Head = forward.Head;
pub const Error = forward.Error;
pub const Kernels = forward.Kernels;
pub const KernelId = forward.KernelId;
pub const kernel_ids = forward.kernel_ids;
pub const run = forward.run;
pub const setTrace = forward.setTrace;
pub const getenvBool = forward.getenvBool;
