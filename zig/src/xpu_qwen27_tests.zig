//! Host-side tests for the Qwen 3.8 27B XPU pieces (arena packing, state sizes): no card, no driver.
//! The on-card gates live in `tf-xpu-qwen27-test` (`zig build xpu-qwen27-test`).
const std = @import("std");

test {
    _ = @import("families/qwen27/xpu_weights.zig");
    _ = @import("families/qwen27/xpu_state.zig");
    _ = @import("families/qwen27/xpu_forward.zig");
}
