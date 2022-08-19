#pragma once
#include <limits>

#define OP_ASSIGN 0
#define OP_MAX 1
#define OP_MIN 2
#define OP_ADD 3
#define OP_SUB 4

#define DEFAULT_MAX 10
#define DEFAULT_MIN -10
#define DEFAULT_ZERO 0
#define DEFAULT_ONE 1

#define OGS_GPU_HASH_TABLE_OP(val, VType, OpType, ...) do {                \
    if((val) == OP_ASSIGN) {                                               \
        typedef Assign<VType> OpType;                                  \
        {__VA_ARGS__}                                                     \
    } else if((val) == OP_MAX) {                                           \
        typedef Max<VType> OpType;                                     \
        {__VA_ARGS__}                                                     \
    } else if((val) == OP_MIN) {                                           \
        typedef Min<VType> OpType;                                     \
        {__VA_ARGS__}                                                     \
    } else if((val) == OP_ADD) {                                           \
        typedef Add<VType> OpType;                                     \
        {__VA_ARGS__}                                                     \
    } else if((val) == OP_SUB) {                                           \
        typedef Sub<VType> OpType;                                     \
        {__VA_ARGS__}                                                     \
    } else {                                                               \
        LOG(FATAL) << "GPU hash table only support Assign, Max, Min, Add, Sub operations";  \
    }                                                                      \
} while (0)


constexpr int32_t kInt32EmptyKey = 0xffffffff;
constexpr int64_t kInt64EmptyKey = 0xffffffffffffffff;
constexpr int8_t kInt8EmptyKey = 0xff;


template <typename T>
struct EmptyKey {};

template <>
struct EmptyKey<int32_t> {
    static __device__ __host__ inline int32_t getValue() {
        return kInt32EmptyKey;
    }
};

template <>
struct EmptyKey<int64_t> {
    static __device__ __host__ inline int64_t getValue() {
        return kInt64EmptyKey;
    }
};


template <>
struct EmptyKey<int8_t> {
    static __device__ __host__ inline int64_t getValue() {
        return kInt8EmptyKey;
    }
};


template <typename T>
struct Limits {};

constexpr float kFloatMax = std::numeric_limits<float>::max();
constexpr float kFloatMin = std::numeric_limits<float>::lowest();

template <>
struct Limits<float> {
    static __device__ __host__ inline float getMin() {
        return kFloatMin;
    }
    static __device__ __host__ inline float getMax() {
        return kFloatMax;
    }
};


constexpr int32_t kInt32Max = std::numeric_limits<int32_t>::max();
constexpr int32_t kInt32Min = std::numeric_limits<int32_t>::lowest();

template <>
struct Limits<int32_t> {
    static __device__ __host__ inline int32_t getMin() {
        return kInt32Min;
    }
    static __device__ __host__ inline int32_t getMax() {
        return kInt32Max;
    }
};


constexpr int64_t kInt64Max = std::numeric_limits<int64_t>::max();
constexpr int64_t kInt64Min = std::numeric_limits<int64_t>::lowest();

template <>
struct Limits<int64_t> {
    static __device__ __host__ inline int64_t getMin() {
        return kInt64Min;
    }
    static __device__ __host__ inline int64_t getMax() {
        return kInt64Max;
    }
};


template<typename VType>
inline VType get_default_value(int _default_value){
    VType default_value;

    if(_default_value == DEFAULT_MAX){
        default_value = Limits<VType>::getMax();

    } else if (_default_value == DEFAULT_MIN) {
        default_value = Limits<VType>::getMin();

    } else if (_default_value == DEFAULT_ONE) {
        default_value = static_cast<VType>(1);

    } else {
        default_value = static_cast<VType>(0);
    }
    
    return default_value;
}


template <typename Dtype>
struct Max {
    typedef Dtype type;
    __device__ static inline void Call(Dtype* address, Dtype value){
        AtomicMax(address, value);
    }
};


//only support integer
template <typename Dtype>
struct Min {
    typedef Dtype type;
    __device__ static inline void Call(Dtype* address, Dtype value){
        AtomicMin(address, value);
    }
};


template <typename Dtype>
struct Add {
    typedef Dtype type;
    __device__ static inline void Call(Dtype* address, Dtype value){
        AtomicAdd(address, value);
    }
};


template <typename Dtype>
struct Sub {
    typedef Dtype type;
    __device__ static inline void Call(Dtype* address, Dtype value){
        AtomicSub(address, value);
    }
};


template <typename Dtype>
struct Assign {
    typedef Dtype type;
    __device__ static inline void Call(Dtype* address, Dtype value){
        *address = value;
    }
};