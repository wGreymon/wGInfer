#ifndef __WGINFER_H__
#define __WGINFER_H__

#if defined(_WIN32)
#define __export __declspec(dllexport)
#elif defined(__GNUC__) && ((__GNUC__ >= 4) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 3))
#define __export __attribute__((visibility("default")))
#else
#define __export
#endif

#ifdef __cplusplus
#define __C extern "C"
#include <cstddef>
#include <cstdint>
#else
#define __C
#include <stddef.h>
#include <stdint.h>
#endif

// Device Types
typedef enum {
    WGINFER_DEVICE_CPU = 0,
    //// TODO: Add more device types here. Numbers need to be consecutive.
    WGINFER_DEVICE_NVIDIA = 1,
    WGINFER_DEVICE_METAX = 2,
    WGINFER_DEVICE_TYPE_COUNT
} wginferDeviceType_t;

// Data Types
typedef enum {
    WGINFER_DTYPE_INVALID = 0,
    WGINFER_DTYPE_BYTE = 1,
    WGINFER_DTYPE_BOOL = 2,
    WGINFER_DTYPE_I8 = 3,
    WGINFER_DTYPE_I16 = 4,
    WGINFER_DTYPE_I32 = 5,
    WGINFER_DTYPE_I64 = 6,
    WGINFER_DTYPE_U8 = 7,
    WGINFER_DTYPE_U16 = 8,
    WGINFER_DTYPE_U32 = 9,
    WGINFER_DTYPE_U64 = 10,
    WGINFER_DTYPE_F8 = 11,
    WGINFER_DTYPE_F16 = 12,
    WGINFER_DTYPE_F32 = 13,
    WGINFER_DTYPE_F64 = 14,
    WGINFER_DTYPE_C16 = 15,
    WGINFER_DTYPE_C32 = 16,
    WGINFER_DTYPE_C64 = 17,
    WGINFER_DTYPE_C128 = 18,
    WGINFER_DTYPE_BF16 = 19,
} wginferDataType_t;

// Runtime Types
// Stream
typedef void *wginferStream_t;

// Memory Copy Directions
typedef enum {
    WGINFER_MEMCPY_H2H = 0,
    WGINFER_MEMCPY_H2D = 1,
    WGINFER_MEMCPY_D2H = 2,
    WGINFER_MEMCPY_D2D = 3,
} wginferMemcpyKind_t;

#endif // __WGINFER_H__
