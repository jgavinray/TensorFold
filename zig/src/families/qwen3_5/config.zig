//! Admission for the Qwen3.5-2B MLX affine checkpoint: pinned geometry and formats, tied or untied head; other geometries require qualification.
const std = @import("std");

pub const hidden = 2048;
pub const vocab = 248320;
pub const layers = 24;
pub const intermediate = 6144;
pub const linear_heads = 16;
pub const linear_dim = 128;
pub const conv_dim = 6144;
pub const conv_taps = 4;
pub const query_heads = 8;
pub const kv_heads = 2;
pub const head_dim = 256;
pub const rotary_dim = 64;
pub const group = 64;
pub const eps: f32 = 1e-6;
pub const theta: f32 = 10000000;

/// The geometry `text_config` states, before admission. `admitted` pins it to one of the two shapes this engine runs:
/// the Qwen3.5-2B checkpoint (`qwen3_5`'s own constants) or the Qwen 3.8 27B one (`qwen27`'s layout). Anything else is
/// refused, so a checkpoint cannot slip a third shape into a forward built for these two.
pub const Geometry = struct {
    hidden: usize,
    vocab: usize,
    layers: usize,
    intermediate: usize,
    linear_heads: usize,
    linear_value_heads: usize,
    linear_dim: usize,
    linear_value_dim: usize,
    conv_taps: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    full_attention_interval: usize,

    /// Whether layer `index` is a full-attention layer under this checkpoint's interval.
    pub fn fullAttention(self: Geometry, index: usize) bool {
        return (index + 1) % self.full_attention_interval == 0;
    }

    /// The Qwen 3.8 27B layout (`qwen27/config.zig` `Pinned`): hidden 5120, 64 layers, 24 query / 4 kv heads,
    /// 16 key / 48 value GDN heads at dim 128, interval 4.
    pub const qwen27 = Geometry{
        .hidden = 5120,
        .vocab = 248320,
        .layers = 64,
        .intermediate = 17408,
        .linear_heads = 16,
        .linear_value_heads = 48,
        .linear_dim = 128,
        .linear_value_dim = 128,
        .conv_taps = 4,
        .query_heads = 24,
        .kv_heads = 4,
        .head_dim = 256,
        .full_attention_interval = 4,
    };

    /// The Qwen3.5-2B layout (the constants above).
    pub const qwen35 = Geometry{
        .hidden = hidden,
        .vocab = vocab,
        .layers = layers,
        .intermediate = intermediate,
        .linear_heads = linear_heads,
        .linear_value_heads = linear_heads,
        .linear_dim = linear_dim,
        .linear_value_dim = linear_dim,
        .conv_taps = conv_taps,
        .query_heads = query_heads,
        .kv_heads = kv_heads,
        .head_dim = head_dim,
        .full_attention_interval = 4,
    };

    /// Whether this geometry is one of the two the engine runs. Every dimension is checked: an arm matches whole or
    /// not at all, so a partially matching checkpoint (the exact class of bug this validator exists to catch) is
    /// refused rather than run through a forward sized for another shape.
    pub fn admitted(self: Geometry) bool {
        inline for ([_]Geometry{ qwen35, qwen27 }) |arm| {
            if (self.eql(arm)) return true;
        }
        return false;
    }

    pub fn eql(self: Geometry, other: Geometry) bool {
        return self.hidden == other.hidden and self.vocab == other.vocab and self.layers == other.layers and
            self.intermediate == other.intermediate and self.linear_heads == other.linear_heads and
            self.linear_value_heads == other.linear_value_heads and self.linear_dim == other.linear_dim and
            self.linear_value_dim == other.linear_value_dim and self.conv_taps == other.conv_taps and
            self.query_heads == other.query_heads and self.kv_heads == other.kv_heads and self.head_dim == other.head_dim and
            self.full_attention_interval == other.full_attention_interval;
    }
};

pub const Config = struct {
    context: usize,
    /// Whether the head is the packed input embedding (true) or a separate lm_head projection (false).
    tied: bool,
    /// The checkpoint's own geometry, read from `text_config` rather than assumed. `parse` admits the two shapes this
    /// engine runs (the 2B above, and the 27B in `qwen27`); every other dimension is still refused.
    hidden: usize,
    vocab: usize,
    layers: usize,
    intermediate: usize,
    linear_heads: usize,
    linear_value_heads: usize,
    linear_dim: usize,
    linear_value_dim: usize,
    conv_taps: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    /// Layers between two full-attention layers: layer `i` is full attention when `(i + 1) % interval == 0`.
    full_attention_interval: usize,

    /// The checkpoint's geometry as a `Geometry`, so callers can compare whole layouts (the family router does).
    pub fn geometry(self: Config) Geometry {
        return .{
            .hidden = self.hidden,
            .vocab = self.vocab,
            .layers = self.layers,
            .intermediate = self.intermediate,
            .linear_heads = self.linear_heads,
            .linear_value_heads = self.linear_value_heads,
            .linear_dim = self.linear_dim,
            .linear_value_dim = self.linear_value_dim,
            .conv_taps = self.conv_taps,
            .query_heads = self.query_heads,
            .kv_heads = self.kv_heads,
            .head_dim = self.head_dim,
            .full_attention_interval = self.full_attention_interval,
        };
    }

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(bytes);
        return parse(gpa, bytes);
    }
};

/// Whether layer `index` of the Qwen3.5-2B schedule is a GatedDeltaNet (linear-attention) layer. The 2B interval is
/// 4, which is also the 27B's, so this serves both admitted shapes; `Geometry.fullAttention` is the general form.
pub fn linear(index: usize) bool {
    return index % 4 != 3;
}

fn object(v: std.json.Value) !std.json.ObjectMap {
    return if (v == .object) v.object else error.BadQwenConfig;
}

fn field(o: std.json.ObjectMap, key: []const u8) !std.json.Value {
    return o.get(key) orelse error.BadQwenConfig;
}

fn number(v: std.json.Value) !f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadQwenConfig,
    };
}

fn integer(o: std.json.ObjectMap, key: []const u8) !usize {
    const v = try field(o, key);
    if (v != .integer or v.integer < 1) return error.BadQwenConfig;
    return @intCast(v.integer);
}

fn string(o: std.json.ObjectMap, key: []const u8, expected: []const u8) !void {
    const v = try field(o, key);
    if (v != .string or !std.mem.eql(u8, v.string, expected)) return error.UnsupportedQwenConfig;
}

fn boolean(o: std.json.ObjectMap, key: []const u8, expected: bool) !void {
    const v = try field(o, key);
    if (v != .bool or v.bool != expected) return error.UnsupportedQwenConfig;
}

fn flag(o: std.json.ObjectMap, key: []const u8) !bool {
    const v = try field(o, key);
    if (v != .bool) return error.BadQwenConfig;
    return v.bool;
}

fn quantization(v: std.json.Value) !void {
    const o = try object(v);
    if (try integer(o, "bits") != 4 or try integer(o, "group_size") != group) return error.UnsupportedQwenQuantization;
    if (o.get("mode")) |mode| {
        if (mode != .string or !std.mem.eql(u8, mode.string, "affine")) return error.UnsupportedQwenQuantization;
    }
    var it = o.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        if (!std.mem.eql(u8, k, "bits") and !std.mem.eql(u8, k, "group_size") and !std.mem.eql(u8, k, "mode"))
            return error.UnsupportedQwenQuantization;
    }
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer parsed.deinit();
    const root = try object(parsed.value);
    try string(root, "model_type", "qwen3_5");
    const text = try object(try field(root, "text_config"));
    try string(text, "model_type", "qwen3_5_text");
    // The head's tiedness: the language config decides; a root-level copy must agree.
    const tied = try flag(text, "tie_word_embeddings");
    if (root.get("tie_word_embeddings")) |v| {
        if (v != .bool) return error.BadQwenConfig;
        if (v.bool != tied) return error.BadQwenConfig;
    }
    try boolean(text, "attention_bias", false);
    try boolean(text, "attn_output_gate", true);
    try string(text, "hidden_act", "silu");
    try string(text, "mamba_ssm_dtype", "float32");
    // The checkpoint's own geometry: read every dimension, then admit it against the shapes this engine runs. The 2B
    // pins stay exactly as they were (its acceptance gates depend on them); the 27B is a second admitted arm with its
    // own real layout, so `qwen27`'s forward and the shape-agnostic XPU kernels serve it.
    const g = Geometry{
        .hidden = try integer(text, "hidden_size"),
        .vocab = try integer(text, "vocab_size"),
        .layers = try integer(text, "num_hidden_layers"),
        .intermediate = try integer(text, "intermediate_size"),
        .linear_heads = try integer(text, "linear_num_key_heads"),
        .linear_value_heads = try integer(text, "linear_num_value_heads"),
        .linear_dim = try integer(text, "linear_key_head_dim"),
        .linear_value_dim = try integer(text, "linear_value_head_dim"),
        .conv_taps = try integer(text, "linear_conv_kernel_dim"),
        .query_heads = try integer(text, "num_attention_heads"),
        .kv_heads = try integer(text, "num_key_value_heads"),
        .head_dim = try integer(text, "head_dim"),
        .full_attention_interval = try integer(text, "full_attention_interval"),
    };
    if (!g.admitted()) return error.UnsupportedQwenGeometry;
    if (try number(try field(text, "rms_norm_eps")) != 1e-6) return error.UnsupportedQwenConfig;
    const types = try field(text, "layer_types");
    if (types != .array or types.array.items.len != g.layers) return error.UnsupportedQwenGeometry;
    for (types.array.items, 0..) |kind, i| {
        const want = if (g.fullAttention(i)) "full_attention" else "linear_attention";
        if (kind != .string or !std.mem.eql(u8, kind.string, want)) return error.UnsupportedQwenGeometry;
    }
    const rope = try object(try field(text, "rope_parameters"));
    try string(rope, "rope_type", "default");
    try boolean(rope, "mrope_interleaved", true);
    if (try number(try field(rope, "rope_theta")) != theta or
        try number(try field(rope, "partial_rotary_factor")) != 0.25) return error.UnsupportedQwenRotary;
    const sections = try field(rope, "mrope_section");
    if (sections != .array or sections.array.items.len != 3) return error.UnsupportedQwenRotary;
    for (sections.array.items, [_]i64{ 11, 11, 10 }) |s, expected| {
        if (s != .integer or s.integer != expected) return error.UnsupportedQwenRotary;
    }
    try quantization(root.get("quantization") orelse root.get("quantization_config") orelse return error.UnsupportedQwenQuantization);
    if (root.get("quantization_config")) |q| try quantization(q);
    const context = try integer(text, "max_position_embeddings");
    if (context > 262144) return error.UnsupportedQwenConfig;
    return .{
        .context = context,
        .tied = tied,
        .hidden = g.hidden,
        .vocab = g.vocab,
        .layers = g.layers,
        .intermediate = g.intermediate,
        .linear_heads = g.linear_heads,
        .linear_value_heads = g.linear_value_heads,
        .linear_dim = g.linear_dim,
        .linear_value_dim = g.linear_value_dim,
        .conv_taps = g.conv_taps,
        .query_heads = g.query_heads,
        .kv_heads = g.kv_heads,
        .head_dim = g.head_dim,
        .full_attention_interval = g.full_attention_interval,
    };
}

const fixture =
    \\{
    \\  "model_type": "qwen3_5",
    \\  "text_config": {
    \\    "attention_bias": false,
    \\    "attention_dropout": 0.0,
    \\    "attn_output_gate": true,
    \\    "dtype": "bfloat16",
    \\    "eos_token_id": 248044,
    \\    "full_attention_interval": 4,
    \\    "head_dim": 256,
    \\    "hidden_act": "silu",
    \\    "hidden_size": 2048,
    \\    "initializer_range": 0.02,
    \\    "intermediate_size": 6144,
    \\    "layer_types": [
    \\      "linear_attention",
    \\      "linear_attention",
    \\      "linear_attention",
    \\      "full_attention",
    \\      "linear_attention",
    \\      "linear_attention",
    \\      "linear_attention",
    \\      "full_attention",
    \\      "linear_attention",
    \\      "linear_attention",
    \\      "linear_attention",
    \\      "full_attention",
    \\      "linear_attention",
    \\      "linear_attention",
    \\      "linear_attention",
    \\      "full_attention",
    \\      "linear_attention",
    \\      "linear_attention",
    \\      "linear_attention",
    \\      "full_attention",
    \\      "linear_attention",
    \\      "linear_attention",
    \\      "linear_attention",
    \\      "full_attention"
    \\    ],
    \\    "linear_conv_kernel_dim": 4,
    \\    "linear_key_head_dim": 128,
    \\    "linear_num_key_heads": 16,
    \\    "linear_num_value_heads": 16,
    \\    "linear_value_head_dim": 128,
    \\    "max_position_embeddings": 262144,
    \\    "mlp_only_layers": [],
    \\    "model_type": "qwen3_5_text",
    \\    "mtp_num_hidden_layers": 1,
    \\    "mtp_use_dedicated_embeddings": false,
    \\    "num_attention_heads": 8,
    \\    "num_hidden_layers": 24,
    \\    "num_key_value_heads": 2,
    \\    "rms_norm_eps": 1e-06,
    \\    "tie_word_embeddings": true,
    \\    "use_cache": true,
    \\    "vocab_size": 248320,
    \\    "mamba_ssm_dtype": "float32",
    \\    "rope_parameters": {
    \\      "mrope_interleaved": true,
    \\      "mrope_section": [
    \\        11,
    \\        11,
    \\        10
    \\      ],
    \\      "rope_type": "default",
    \\      "rope_theta": 10000000,
    \\      "partial_rotary_factor": 0.25
    \\    }
    \\  },
    \\  "quantization": {
    \\    "group_size": 64,
    \\    "bits": 4,
    \\    "mode": "affine"
    \\  },
    \\  "quantization_config": {
    \\    "group_size": 64,
    \\    "bits": 4,
    \\    "mode": "affine"
    \\  },
    \\  "tie_word_embeddings": true
    \\}
;

test "admit the affine 2B geometry and text rotary layout" {
    const gpa = std.testing.allocator;
    const config = try parse(gpa, fixture);
    try std.testing.expectEqual(@as(usize, 262144), config.context);
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, fixture, .{});
    defer doc.deinit();
    const compact = try std.json.Stringify.valueAlloc(gpa, doc.value, .{});
    defer gpa.free(compact);
    const changes = .{
        .{ "\"hidden_size\":2048", "\"hidden_size\":1024", error.UnsupportedQwenGeometry },
        .{ "\"bits\":4", "\"bits\":8", error.UnsupportedQwenQuantization },
        .{ "\"group_size\":64", "\"group_size\":32", error.UnsupportedQwenQuantization },
        .{ "\"mode\":\"affine\"", "\"mode\":\"mxfp4\"", error.UnsupportedQwenQuantization },
        .{ "\"rope_theta\":10000000", "\"rope_theta\":100000", error.UnsupportedQwenRotary },
        .{ "\"mrope_section\":[11,11,10]", "\"mrope_section\":[16,8,8]", error.UnsupportedQwenRotary },
        .{ "\"max_position_embeddings\":262144", "\"max_position_embeddings\":524288", error.UnsupportedQwenConfig },
    };
    inline for (changes) |change| {
        try std.testing.expect(std.mem.indexOf(u8, compact, change[0]) != null);
        const bad = try std.mem.replaceOwned(u8, gpa, compact, change[0], change[1]);
        defer gpa.free(bad);
        try std.testing.expectError(change[2], parse(gpa, bad));
    }
}

test "admit untied heads and refuse a root and language tiedness disagreement" {
    const gpa = std.testing.allocator;
    // Both declarations flipped: the language config and the root-level copy agree on untied.
    const untied = try std.mem.replaceOwned(u8, gpa, fixture, "\"tie_word_embeddings\": true", "\"tie_word_embeddings\": false");
    defer gpa.free(untied);
    try std.testing.expectEqual(false, (try parse(gpa, untied)).tied);
    try std.testing.expectEqual(@as(usize, 262144), (try parse(gpa, untied)).context);
    // Only the root flipped: text_config keeps the tie, the root disagrees -> malformed export.
    const mismatch = try std.mem.replaceOwned(u8, gpa, fixture, "\"tie_word_embeddings\": true\n}", "\"tie_word_embeddings\": false\n}");
    defer gpa.free(mismatch);
    try std.testing.expectError(error.BadQwenConfig, parse(gpa, mismatch));
}

/// The real Qwen 3.8 27B MLX affine config: same `model_type` as the 2B, a different geometry. Before this arm
/// existed the validator refused it (`UnsupportedQwenGeometry`) and `serve --backend xpu` exited at load.
const fixture27b =
    \\{
    \\  "model_type": "qwen3_5",
    \\  "text_config": {
    \\    "attention_bias": false,
    \\    "attention_dropout": 0.0,
    \\    "attn_output_gate": true,
    \\    "bos_token_id": 248044,
    \\    "dtype": "bfloat16",
    \\    "eos_token_id": 248044,
    \\    "full_attention_interval": 4,
    \\    "head_dim": 256,
    \\    "hidden_act": "silu",
    \\    "hidden_size": 5120,
    \\    "initializer_range": 0.02,
    \\    "intermediate_size": 17408,
    \\    "layer_types": [
    \\      "linear_attention",
    \\      "linear_attention",
    \\      "linear_attention",
    \\      "full_attention"
    \\    ],
    \\    "linear_conv_kernel_dim": 4,
    \\    "linear_key_head_dim": 128,
    \\    "linear_num_key_heads": 16,
    \\    "linear_num_value_heads": 48,
    \\    "linear_value_head_dim": 128,
    \\    "max_position_embeddings": 262144,
    \\    "model_type": "qwen3_5_text",
    \\    "num_attention_heads": 24,
    \\    "num_hidden_layers": 4,
    \\    "num_key_value_heads": 4,
    \\    "rms_norm_eps": 1e-06,
    \\    "tie_word_embeddings": false,
    \\    "use_cache": true,
    \\    "vocab_size": 248320,
    \\    "mamba_ssm_dtype": "float32",
    \\    "rope_parameters": {
    \\      "mrope_interleaved": true,
    \\      "mrope_section": [11, 11, 10],
    \\      "rope_type": "default",
    \\      "rope_theta": 10000000,
    \\      "partial_rotary_factor": 0.25
    \\    }
    \\  },
    \\  "quantization": { "group_size": 64, "bits": 4, "mode": "affine" },
    \\  "quantization_config": { "group_size": 64, "bits": 4, "mode": "affine" },
    \\  "tie_word_embeddings": false
    \\}
;

test "admit the 27B geometry arm and carry its derived dimensions" {
    const gpa = std.testing.allocator;
    // The fixture states 4 layers so the schedule is readable; the admitted 27B arm is 64. Admit it by hand so the
    // test exercises `Geometry.admitted` on the real 27B numbers rather than on the shortened fixture.
    try std.testing.expect(Geometry.qwen27.admitted());
    try std.testing.expect(Geometry.qwen35.admitted());
    try std.testing.expect(!(Geometry{ .hidden = 5120, .vocab = 248320, .layers = 4, .intermediate = 17408, .linear_heads = 16, .linear_value_heads = 48, .linear_dim = 128, .linear_value_dim = 128, .conv_taps = 4, .query_heads = 24, .kv_heads = 4, .head_dim = 256, .full_attention_interval = 4 }).admitted());
    // The derived fields: the 27B fixture's own numbers, not the 2B constants.
    const long = try std.mem.replaceOwned(u8, gpa, fixture27b, "\"num_hidden_layers\": 4", "\"num_hidden_layers\": 64");
    defer gpa.free(long);
    const wide = try std.mem.replaceOwned(u8, gpa, long, "\"layer_types\": [\n      \"linear_attention\",\n      \"linear_attention\",\n      \"linear_attention\",\n      \"full_attention\"\n    ]", "\"full_attention_interval\": 4");
    defer gpa.free(wide);
    const c = try parse(gpa, wide);
    try std.testing.expectEqual(@as(usize, 5120), c.hidden);
    try std.testing.expectEqual(@as(usize, 64), c.layers);
    try std.testing.expectEqual(@as(usize, 17408), c.intermediate);
    try std.testing.expectEqual(@as(usize, 24), c.query_heads);
    try std.testing.expectEqual(@as(usize, 4), c.kv_heads);
    try std.testing.expectEqual(@as(usize, 4), c.full_attention_interval);
    try std.testing.expectEqual(false, c.tied);
    // A 27B arm with a 2B-shaped dimension swapped in is refused whole, not partially admitted.
    const tampered = try std.mem.replaceOwned(u8, gpa, wide, "\"hidden_size\": 5120", "\"hidden_size\": 2048");
    defer gpa.free(tampered);
    try std.testing.expectError(error.UnsupportedQwenGeometry, parse(gpa, tampered));
}

test "the 2B pins stay exact under the derived reader" {
    const gpa = std.testing.allocator;
    const c = try parse(gpa, fixture);
    try std.testing.expectEqual(@as(usize, 2048), c.hidden);
    try std.testing.expectEqual(@as(usize, 24), c.layers);
    try std.testing.expectEqual(@as(usize, 6144), c.intermediate);
    try std.testing.expectEqual(@as(usize, 8), c.query_heads);
    try std.testing.expectEqual(@as(usize, 2), c.kv_heads);
    try std.testing.expectEqual(@as(usize, 4), c.full_attention_interval);
    try std.testing.expectEqual(true, c.tied);
}
