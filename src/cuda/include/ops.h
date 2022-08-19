#pragma once

#define OGS_ID_TYPE_SWITCH(val, IdType, ...) do {             \
  if ((val) == torch::kInt32) {                               \
    typedef int32_t IdType;                                   \
    {__VA_ARGS__}                                             \
  } else if ((val) == torch::kInt64) {                        \
    typedef int64_t IdType;                                   \
    {__VA_ARGS__}                                             \
  } else {                                                    \
    LOG(FATAL) << "ID can only be int32 or int64";            \
  }                                                           \
} while (0);

#define OGS_VALUE_TYPE_SWITCH(val, VType, ...) do {             \
  if ((val) == torch::kInt32) {                                 \
    typedef int32_t VType;                                      \
    {__VA_ARGS__}                                               \
  } else if ((val) == torch::kInt64) {                          \
    typedef int64_t VType;                                      \
    {__VA_ARGS__}                                               \
  } else if ((val) == torch::kFloat32) {                        \
    typedef float VType;                                        \
    {__VA_ARGS__}                                               \
  } else {                                                      \
    LOG(FATAL) << "Value can only be int32 or int64 or float32";  \
  }                                                             \
} while (0);


#define OGS_BOOL_SWITCH(val, SWITCH, ...) do {                \
  if ((val)) {                                                \
    constexpr bool SWITCH = true;                             \
    {__VA_ARGS__}                                             \
  } else {                                                    \
    constexpr bool SWITCH = false;                            \
    {__VA_ARGS__}                                             \
  }                                                           \
} while (0);
