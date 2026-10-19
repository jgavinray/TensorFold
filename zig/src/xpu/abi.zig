//! Level Zero entry points and structs, hand-declared from /usr/include/level_zero/ze_api.h (core 1.x); layouts are
//! checked against the C header at compile time below. No SYCL, no OpenCL, no non-Intel anything.
const std = @import("std");

pub const Result = u32;
pub const success: Result = 0;
pub const not_ready: Result = 1;

pub const Driver = ?*opaque {};
pub const Device = ?*opaque {};
pub const Context = ?*opaque {};
pub const CommandList = ?*opaque {};
pub const CommandQueue = ?*opaque {};
pub const EventPool = ?*opaque {};
pub const Event = ?*opaque {};
pub const Module = ?*opaque {};
pub const BuildLog = ?*opaque {};
pub const Kernel = ?*opaque {};

pub const intel_vendor_id: u32 = 0x8086;
pub const device_type_gpu: u32 = 1;

pub const stype = struct {
    pub const device_properties: u32 = 0x3;
    pub const command_queue_group_properties: u32 = 0x6;
    pub const context_desc: u32 = 0xd;
    pub const command_queue_desc: u32 = 0xe;
    pub const event_pool_desc: u32 = 0x10;
    pub const event_desc: u32 = 0x11;
    pub const device_mem_alloc_desc: u32 = 0x15;
    pub const host_mem_alloc_desc: u32 = 0x16;
    pub const module_desc: u32 = 0x1b;
    pub const command_list_desc: u32 = 0x19;
    pub const kernel_desc: u32 = 0x1d;
    pub const device_memory_properties: u32 = 0x7;
    pub const pci_ext_properties: u32 = 0x10008;
};

pub const queue_flag_in_order: u32 = 1 << 1;
/// ze_command_list_flag_t ZE_COMMAND_LIST_FLAG_IN_ORDER: a regular list's commands run one after another.
pub const list_flag_in_order: u32 = 1 << 3;
pub const queue_mode_asynchronous: u32 = 2;
pub const queue_group_compute: u32 = 1 << 0;
pub const event_pool_host_visible: u32 = 1 << 0;
pub const event_pool_kernel_timestamp: u32 = 1 << 2;
pub const event_scope_host: u32 = 1 << 2;
pub const module_format_native: u32 = 1;
pub const wait_forever: u64 = std.math.maxInt(u64);

pub const DeviceProperties = extern struct {
    stype: u32 = stype.device_properties,
    pNext: ?*anyopaque = null,
    type: u32 = 0,
    vendorId: u32 = 0,
    deviceId: u32 = 0,
    flags: u32 = 0,
    subdeviceId: u32 = 0,
    coreClockRate: u32 = 0,
    maxMemAllocSize: u64 = 0,
    maxHardwareContexts: u32 = 0,
    maxCommandQueuePriority: u32 = 0,
    numThreadsPerEU: u32 = 0,
    physicalEUSimdWidth: u32 = 0,
    numEUsPerSubslice: u32 = 0,
    numSubslicesPerSlice: u32 = 0,
    numSlices: u32 = 0,
    timerResolution: u64 = 0,
    timestampValidBits: u32 = 0,
    kernelTimestampValidBits: u32 = 0,
    uuid: [16]u8 = @splat(0),
    name: [256]u8 = @splat(0),
};

/// ze_device_memory_properties_t: one per memory region of a device (a B70 has one, its VRAM).
pub const DeviceMemoryProperties = extern struct {
    stype: u32 = stype.device_memory_properties,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    maxClockRate: u32 = 0,
    maxBusWidth: u32 = 0,
    totalSize: u64 = 0,
    name: [256]u8 = @splat(0),
};

pub const PciAddress = extern struct { domain: u32 = 0, bus: u32 = 0, device: u32 = 0, function: u32 = 0 };
pub const PciExtProperties = extern struct {
    stype: u32 = stype.pci_ext_properties,
    pNext: ?*anyopaque = null,
    address: PciAddress = .{},
    max_speed: [16]u8 = @splat(0), // ze_pci_speed_ext_t: genVersion, width, maxBandwidth (i32 x3) + pad -> 16 bytes
};

pub const QueueGroupProperties = extern struct {
    stype: u32 = stype.command_queue_group_properties,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    maxMemoryFillPatternSize: usize = 0,
    numQueues: u32 = 0,
};

pub const ContextDesc = extern struct { stype: u32 = stype.context_desc, pNext: ?*const anyopaque = null, flags: u32 = 0 };
pub const QueueDesc = extern struct {
    stype: u32 = stype.command_queue_desc,
    pNext: ?*const anyopaque = null,
    ordinal: u32 = 0,
    index: u32 = 0,
    flags: u32 = 0,
    mode: u32 = 0,
    priority: u32 = 0,
};
pub const DeviceMemDesc = extern struct { stype: u32 = stype.device_mem_alloc_desc, pNext: ?*const anyopaque = null, flags: u32 = 0, ordinal: u32 = 0 };
pub const HostMemDesc = extern struct { stype: u32 = stype.host_mem_alloc_desc, pNext: ?*const anyopaque = null, flags: u32 = 0 };
pub const EventPoolDesc = extern struct { stype: u32 = stype.event_pool_desc, pNext: ?*const anyopaque = null, flags: u32 = 0, count: u32 = 0 };
pub const EventDesc = extern struct { stype: u32 = stype.event_desc, pNext: ?*const anyopaque = null, index: u32 = 0, signal: u32 = 0, wait: u32 = 0 };
pub const ModuleDesc = extern struct {
    stype: u32 = stype.module_desc,
    pNext: ?*const anyopaque = null,
    format: u32 = module_format_native,
    inputSize: usize = 0,
    pInputModule: ?[*]const u8 = null,
    pBuildFlags: ?[*:0]const u8 = null,
    pConstants: ?*const anyopaque = null,
};
pub const KernelDesc = extern struct { stype: u32 = stype.kernel_desc, pNext: ?*const anyopaque = null, flags: u32 = 0, pKernelName: ?[*:0]const u8 = null };
pub const GroupCount = extern struct { x: u32, y: u32, z: u32 };
/// A range within a buffer, for zeCommandListAppendMemoryCopyRegion (dst/src size + offset, then row/slice pitch).
pub const CopyRegion = extern struct {
    origin: GroupCount = .{ .x = 0, .y = 0, .z = 0 },
    region: GroupCount = .{ .x = 0, .y = 0, .z = 0 },
};
/// ze_command_list_desc_t: the ordinal comes before the flags. (They were swapped here; harmless while both were 0.)
pub const CommandListDesc = extern struct {
    stype: u32 = stype.command_list_desc,
    pNext: ?*const anyopaque = null,
    commandQueueGroupOrdinal: u32 = 0,
    flags: u32 = 0,
};
pub const CommandQueueDesc = extern struct {
    stype: u32 = stype.command_queue_desc,
    pNext: ?*const anyopaque = null,
    ordinal: u32 = 0,
    index: u32 = 0,
    flags: u32 = 0,
    mode: u32 = 0,
    priority: u32 = 0,
};
pub const TimestampData = extern struct { start: u64 = 0, end: u64 = 0 };
pub const TimestampResult = extern struct { global: TimestampData = .{}, context: TimestampData = .{} };

comptime {
    std.debug.assert(@sizeOf(DeviceProperties) == 368 and @offsetOf(DeviceProperties, "name") == 112 and
        @offsetOf(DeviceProperties, "timerResolution") == 80 and @offsetOf(DeviceProperties, "kernelTimestampValidBits") == 92);
    std.debug.assert(@sizeOf(PciExtProperties) == 48 and @offsetOf(PciExtProperties, "address") == 16);
    std.debug.assert(@sizeOf(QueueGroupProperties) == 40);
    std.debug.assert(@sizeOf(QueueDesc) == 40 and @sizeOf(EventDesc) == 32 and @sizeOf(ModuleDesc) == 56);
    std.debug.assert(@sizeOf(EventPoolDesc) == 24 and @sizeOf(DeviceMemDesc) == 24 and @sizeOf(TimestampResult) == 32);
    std.debug.assert(@sizeOf(DeviceMemoryProperties) == 296 and @offsetOf(DeviceMemoryProperties, "totalSize") == 32);
}

const C = std.builtin.CallingConvention.c;

/// Every Level Zero entry point the runtime uses; a library missing any one is refused whole.
pub const Api = struct {
    zeInit: *const fn (u32) callconv(C) Result,
    zeDriverGet: *const fn (*u32, ?[*]Driver) callconv(C) Result,
    zeDeviceGet: *const fn (Driver, *u32, ?[*]Device) callconv(C) Result,
    zeDeviceGetProperties: *const fn (Device, *DeviceProperties) callconv(C) Result,
    zeDevicePciGetPropertiesExt: *const fn (Device, *PciExtProperties) callconv(C) Result,
    zeDeviceGetMemoryProperties: *const fn (Device, *u32, ?[*]DeviceMemoryProperties) callconv(C) Result,
    zeDeviceGetCommandQueueGroupProperties: *const fn (Device, *u32, ?[*]QueueGroupProperties) callconv(C) Result,
    zeContextCreate: *const fn (Driver, *const ContextDesc, *Context) callconv(C) Result,
    zeContextDestroy: *const fn (Context) callconv(C) Result,
    zeMemAllocDevice: *const fn (Context, *const DeviceMemDesc, usize, usize, Device, *?*anyopaque) callconv(C) Result,
    zeMemAllocShared: *const fn (Context, *const DeviceMemDesc, *const HostMemDesc, usize, usize, Device, *?*anyopaque) callconv(C) Result,
    zeMemAllocHost: *const fn (Context, *const HostMemDesc, usize, usize, *?*anyopaque) callconv(C) Result,
    zeMemFree: *const fn (Context, ?*anyopaque) callconv(C) Result,
    zeCommandListCreateImmediate: *const fn (Context, Device, *const QueueDesc, *CommandList) callconv(C) Result,
    zeCommandListCreate: *const fn (Context, Device, *const CommandListDesc, *CommandList) callconv(C) Result,
    zeCommandListClose: *const fn (CommandList) callconv(C) Result,
    zeCommandListReset: *const fn (CommandList) callconv(C) Result,
    zeCommandQueueCreate: *const fn (Context, Device, *const CommandQueueDesc, *CommandQueue) callconv(C) Result,
    zeCommandQueueDestroy: *const fn (CommandQueue) callconv(C) Result,
    zeCommandQueueExecuteCommandLists: *const fn (CommandQueue, u32, [*]const CommandList, Event) callconv(C) Result,
    zeCommandQueueSynchronize: *const fn (CommandQueue, u64) callconv(C) Result,
    zeCommandListDestroy: *const fn (CommandList) callconv(C) Result,
    zeCommandListAppendMemoryCopy: *const fn (CommandList, ?*anyopaque, ?*const anyopaque, usize, Event, u32, ?[*]Event) callconv(C) Result,
    zeCommandListAppendMemoryFill: *const fn (CommandList, ?*anyopaque, *const anyopaque, usize, usize, Event, u32, ?[*]Event) callconv(C) Result,
    zeCommandListAppendLaunchKernel: *const fn (CommandList, Kernel, *const GroupCount, Event, u32, ?[*]Event) callconv(C) Result,
    zeCommandListAppendBarrier: *const fn (CommandList, Event, u32, ?[*]Event) callconv(C) Result,
    zeCommandListAppendSignalEvent: *const fn (CommandList, Event) callconv(C) Result,
    zeCommandListHostSynchronize: *const fn (CommandList, u64) callconv(C) Result,
    zeEventPoolCreate: *const fn (Context, *const EventPoolDesc, u32, ?[*]Device, *EventPool) callconv(C) Result,
    zeEventPoolDestroy: *const fn (EventPool) callconv(C) Result,
    zeEventCreate: *const fn (EventPool, *const EventDesc, *Event) callconv(C) Result,
    zeEventDestroy: *const fn (Event) callconv(C) Result,
    zeEventHostSynchronize: *const fn (Event, u64) callconv(C) Result,
    zeEventHostReset: *const fn (Event) callconv(C) Result,
    zeEventQueryKernelTimestamp: *const fn (Event, *TimestampResult) callconv(C) Result,
    zeModuleCreate: *const fn (Context, Device, *const ModuleDesc, *Module, *BuildLog) callconv(C) Result,
    zeModuleDestroy: *const fn (Module) callconv(C) Result,
    zeModuleBuildLogGetString: *const fn (BuildLog, *usize, ?[*]u8) callconv(C) Result,
    zeModuleBuildLogDestroy: *const fn (BuildLog) callconv(C) Result,
    zeKernelCreate: *const fn (Module, *const KernelDesc, *Kernel) callconv(C) Result,
    zeKernelDestroy: *const fn (Kernel) callconv(C) Result,
    zeKernelSetArgumentValue: *const fn (Kernel, u32, usize, ?*const anyopaque) callconv(C) Result,
    zeKernelSetGroupSize: *const fn (Kernel, u32, u32, u32) callconv(C) Result,
};
