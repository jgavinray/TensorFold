// Mock Level Zero loader for the admission tests: two Intel GPUs at 0000:08:00.0 and 0000:44:00.0 plus a foreign-vendor
// GPU at 0000:c1:00.0 that must never be opened or have its PCI address read. Layouts are those of ze_api.h (core 1.x).
#include <stdint.h>
#include <string.h>

typedef uint32_t R;
static int foreign_pci_queries = 0;
static uint32_t last_queue_flags = 0xdeadbeef;
static int queue_creates = 0;
static int drivers_asked = 0;
static int list_creates = 0;
static int executes = 0;
static uint32_t last_list_group = 0xdeadbeef;
static uint32_t last_queue_group = 0xdeadbeef;
static uint32_t last_execute_lists = 0;

#ifndef INIT_RESULT
#define INIT_RESULT 0
#endif
static uint32_t init_flags = 0xffffffff;
#ifndef OMIT_INIT
R zeInit(uint32_t f) { init_flags = f; return INIT_RESULT; }
#endif
#ifndef OMIT_DRIVERGET
R zeDriverGet(uint32_t *n, void **d) { drivers_asked++; if (d) d[0] = (void *)0x100; *n = 1; return 0; }
#endif
R zeDeviceGet(void *drv, uint32_t *n, void **d) { if (d) { d[0] = (void *)1; d[1] = (void *)2; d[2] = (void *)3; } *n = 3; return 0; }
R zeDeviceGetProperties(void *dev, void *p) {
  uint8_t *b = p;
  uint32_t type = 1, vendor = (uintptr_t)dev == 3 ? 0x10de : 0x8086;
  memcpy(b + 16, &type, 4); memcpy(b + 20, &vendor, 4);
  uint64_t max_alloc = 1ull << 32, res = 10; uint32_t bits = 32;
  memcpy(b + 40, &max_alloc, 8); memcpy(b + 80, &res, 8); memcpy(b + 92, &bits, 4);
  strcpy((char *)b + 112, "Mock GPU");
  return 0;
}
R zeDevicePciGetPropertiesExt(void *dev, void *p) {
  uint32_t a[4] = {0, 0, 0, 0};
  switch ((uintptr_t)dev) { case 1: a[1] = 0x08; break; case 2: a[1] = 0x44; break; default: a[1] = 0xc1; foreign_pci_queries++; }
  memcpy((uint8_t *)p + 16, a, 16);
  return 0;
}
R zeDeviceGetCommandQueueGroupProperties(void *dev, uint32_t *n, void *g) {
  if (g) { memset(g, 0, 80); *(uint32_t *)((uint8_t *)g + 40 + 16) = 1; *(uint32_t *)((uint8_t *)g + 16) = 2; } // group 0: copy-only, group 1: compute
  *n = 2; return 0;
}
R zeContextCreate(void *d, const void *desc, void **c) { *c = (void *)0x200; return 0; }
R zeContextDestroy(void *c) { return 0; }
R zeMemAllocDevice(void *c, const void *d, size_t s, size_t a, void *dev, void **p) { return 0x78000001; }
R zeMemAllocShared(void *c, const void *d, const void *h, size_t s, size_t a, void *dev, void **p) { return 0x78000001; }
R zeMemAllocHost(void *c, const void *h, size_t s, size_t a, void **p) { return 0x78000001; }
R zeMemFree(void *c, void *p) { return 0; }
R zeCommandListCreateImmediate(void *c, void *dev, const void *desc, void **l) {
  memcpy(&last_queue_flags, (const uint8_t *)desc + 24, 4); queue_creates++; *l = (void *)0x300; return 0;
}
R zeCommandListCreate(void *c, void *dev, const void *desc, void **l) {
  memcpy(&last_list_group, (const uint8_t *)desc + 20, 4); list_creates++; *l = (void *)0x301; return 0;
}
R zeCommandListClose(void *l) { return 0; }
R zeCommandListReset(void *l) { return 0; }
R zeCommandQueueCreate(void *c, void *dev, const void *desc, void **q) {
  memcpy(&last_queue_group, (const uint8_t *)desc + 16, 4); queue_creates++; *q = (void *)0x302; return 0;
}
R zeCommandQueueDestroy(void *q) { return 0; }
R zeCommandQueueExecuteCommandLists(void *q, uint32_t n, void **l, void *e) { last_execute_lists = n; executes++; return 0; }
R zeCommandQueueSynchronize(void *q, uint64_t t) { return 0; }
R zeCommandListDestroy(void *l) { return 0; }
R zeCommandListAppendMemoryCopy(void *l, void *d, const void *s, size_t n, void *e, uint32_t nw, void **w) { return 0; }
R zeCommandListAppendMemoryFill(void *l, void *d, const void *p, size_t ps, size_t n, void *e, uint32_t nw, void **w) { return 0; }
R zeCommandListAppendLaunchKernel(void *l, void *k, const void *g, void *e, uint32_t nw, void **w) { return 0; }
R zeCommandListAppendBarrier(void *l, void *e, uint32_t nw, void **w) { return 0; }
R zeCommandListAppendSignalEvent(void *l, void *e) { return 0; }
R zeCommandListHostSynchronize(void *l, uint64_t t) { return 0; }
R zeEventPoolCreate(void *c, const void *d, uint32_t n, void **dev, void **p) { *p = (void *)0x400; return 0; }
R zeEventPoolDestroy(void *p) { return 0; }
R zeEventCreate(void *p, const void *d, void **e) { *e = (void *)0x500; return 0; }
R zeEventDestroy(void *e) { return 0; }
R zeEventHostSynchronize(void *e, uint64_t t) { return 0; }
R zeEventHostReset(void *e) { return 0; }
R zeEventQueryKernelTimestamp(void *e, void *r) { return 0; }
R zeModuleCreate(void *c, void *d, const void *desc, void **m, void **log) { return 0x78000001; }
R zeModuleDestroy(void *m) { return 0; }
R zeModuleBuildLogGetString(void *l, size_t *n, char *s) { return 0; }
R zeModuleBuildLogDestroy(void *l) { return 0; }
R zeKernelCreate(void *m, const void *d, void **k) { return 0x78000001; }
R zeKernelDestroy(void *k) { return 0; }
R zeKernelSetArgumentValue(void *k, uint32_t i, size_t s, const void *v) { return 0; }
R zeKernelSetGroupSize(void *k, uint32_t x, uint32_t y, uint32_t z) { return 0; }

int tf_mock_foreign_pci_queries(void) { return foreign_pci_queries; }
uint32_t tf_mock_last_queue_flags(void) { return last_queue_flags; }
int tf_mock_queue_creates(void) { return queue_creates; }
int tf_mock_list_creates(void) { return list_creates; }
int tf_mock_executes(void) { return executes; }
uint32_t tf_mock_last_list_group(void) { return last_list_group; }
uint32_t tf_mock_last_queue_group(void) { return last_queue_group; }
uint32_t tf_mock_last_execute_lists(void) { return last_execute_lists; }
uint32_t tf_mock_init_flags(void) { return init_flags; }
