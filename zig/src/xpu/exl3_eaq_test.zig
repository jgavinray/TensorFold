//! Host tests of the XPU EAQ reader (exl3_eaq.zig) on synthetic packs written to a temporary
//! directory: no GPU, no real weights, and no device code in the module under test.
const std = @import("std");
const core = @import("core");
const eaq = core.eaq_container;
const fixture = core.eaq_fixture;
const Tamper = fixture.Tamper;
const r = @import("exl3_eaq");

fn openIn(t: Tamper, shape: fixture.Shape, why: ?*eaq.Why) !eaq.Pack {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try fixture.write(gpa, io, tmp.dir, t, shape);
    // the container reads the shards during open; after it returns the pack is
    // self-contained, so the directory going away is fine for the mapping tests.
    // Tests that read tensor bytes (PR 2) must keep the tmp dir alive instead.
    var path: [96]u8 = undefined;
    return r.openPack(gpa, io, try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{std.mem.sliceTo(&tmp.sub_path, 0)}), why);
}

test "a good toy pack opens and maps onto the 27B table" {
    var why: eaq.Why = .{};
    var pack = try openIn(.none, .{}, &why);
    defer pack.deinit();
    var table = r.map(std.testing.allocator, &pack, &why) catch |e| {
        std.debug.print("map refused: {s} — why: '{s}'\n", .{ @errorName(e), why.text() });
        return e;
    };
    defer table.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 0), table.projections.len);
    const head = table.head orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("lm_head", head.name);
    try std.testing.expect(!head.dense);
    const q = head.q orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u8, 12), q.halves); // 2 * 6 bits
    try std.testing.expectEqual(@as(usize, 64), q.k);
    try std.testing.expectEqual(@as(usize, 128), q.n);
    try std.testing.expectEqual(core.exl3_format.Codebook.mul1, q.codebook);
    try std.testing.expect(q.suh);
    try std.testing.expect(!q.bias);
    try std.testing.expectEqual(@as(usize, 1), table.others.len);
    try std.testing.expectEqualStrings("model.norm.weight", table.others[0].name);
    try std.testing.expect(table.others[0].dense);
    try std.testing.expect(table.embed == null);
    try std.testing.expect(!table.complete);
}

test "an eaq.fp8/0 entry refuses at open and names it" {
    var why: eaq.Why = .{};
    try std.testing.expectError(r.Error.EaqFp8Entry, openIn(.none, .{ .head_format = "eaq.fp8/0" }, &why));
    try std.testing.expect(std.mem.startsWith(u8, why.text(), "entry lm_head"));
}

test "a required feature refuses through the reader" {
    try std.testing.expectError(eaq.Error.MissingFeature, openIn(.feature, .{}, null));
}

test "an entry name outside the 27B table refuses" {
    var why: eaq.Why = .{};
    var pack = try openIn(.none, .{ .head_name = "model.mystery.weight" }, null);
    defer pack.deinit();
    try std.testing.expectError(r.Error.UnknownProjection, r.map(std.testing.allocator, &pack, &why));
    try std.testing.expect(std.mem.startsWith(u8, why.text(), "entry model.mystery"));
}

test "trellis params that disagree with the parts refuse" {
    var pack = try openIn(.none, .{ .head_params = "{\"bits\":6.0,\"K\":48,\"N\":128,\"dictionary\":\"exl3.mul1.v1\"}" }, null);
    defer pack.deinit();
    try std.testing.expectError(r.Error.BadTrellisPart, r.map(std.testing.allocator, &pack, null));
}

test "missing K refuses" {
    var pack = try openIn(.none, .{ .head_params = "{\"bits\":6.0,\"N\":128,\"dictionary\":\"exl3.mul1.v1\"}" }, null);
    defer pack.deinit();
    try std.testing.expectError(r.Error.BadTrellisParams, r.map(std.testing.allocator, &pack, null));
}

test "an unknown dictionary refuses at the container" {
    try std.testing.expectError(eaq.Error.UnknownDictionary, openIn(.none, .{ .head_params = "{\"bits\":6.0,\"K\":64,\"N\":128,\"dictionary\":\"exl3.muld.v9\"}" }, null));
}

test "layer names classify onto slots and parts, with or without .weight" {
    const attn = r.slotFor("model.layers.3.self_attn.v_proj") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(usize, 3), attn.projection.layer);
    try std.testing.expectEqual(r.Slot.mixer_in, attn.projection.slot);
    try std.testing.expectEqual(@as(u8, 2), attn.projection.part);
    const gdn = r.slotFor("model.layers.2.linear_attn.in_proj_b") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(r.Slot.mixer_in, gdn.projection.slot);
    try std.testing.expectEqual(@as(u8, 3), gdn.projection.part);
    const gu = r.slotFor("model.layers.0.mlp.up_proj.weight") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(r.Slot.gu, gu.projection.slot);
    try std.testing.expectEqual(@as(u8, 1), gu.projection.part);
    const out = r.slotFor("model.layers.4.linear_attn.out_proj") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(r.Slot.mixer_out, out.projection.slot);
    const norm = r.slotFor("model.layers.5.input_layernorm.weight") orelse return error.TestUnexpectedResult;
    try std.testing.expect(norm == .other);
    try std.testing.expectEqual(std.meta.Tag(r.Place).head, @as(std.meta.Tag(r.Place), r.slotFor("lm_head.weight").?));
    try std.testing.expect(r.slotFor("model.layers.+3.mlp.down_proj") == null);
    try std.testing.expect(r.slotFor("model.layers.64.mlp.down_proj") == null);
    try std.testing.expect(r.slotFor("model.layers.3.mlp.gate_proj.weight.safetensors") == null);
}

test "the attention layers are every fourth from the fourth" {
    var n: usize = 0;
    for (0..r.LAYERS) |i| n += @intFromBool(r.isAttention(i));
    try std.testing.expectEqual(@as(usize, 16), n);
}
