#include "graph_ops.h"

#include <curand_kernel.h>
#include "atomic.h"
#include "cuda_common.h"
#include "macro.h"
#include "utils.h"

namespace gs {
namespace impl {
//////////////////////// CSCSumCUDA //////////////////////////
/**
 * @brief SpMV for CSCSum
 */
template <typename IdType, typename DType, bool UseEMap, bool UseNMap>
__global__ void _SegmentSumKernel(IdType* indptr, IdType* EMap, IdType* NMap,
                                  DType* data, int num_rows, int powk,
                                  int out_len, DType* out) {
  // SPMM with CSR.
  int ty = blockIdx.x * blockDim.y + threadIdx.y;
  const IdType stride_y = blockDim.y * gridDim.x;
  const int stride_x = blockDim.x * gridDim.y;
  while (ty < num_rows) {
    int tx = blockIdx.y * blockDim.x + threadIdx.x;
    while (tx < out_len) {
      DType local_accum = 0;
      for (IdType i = indptr[ty]; i < indptr[ty + 1]; ++i) {
        const IdType data_idx = UseEMap ? EMap[i] : i;
        const DType* dataoff = data + data_idx * out_len;
        DType tmp = powk == 1 ? dataoff[tx] : __powf(dataoff[tx], powk);
        local_accum += tmp;
      }
      int out_pos = UseNMap ? NMap[ty * out_len + tx] : ty * out_len + tx;
      out[out_pos] = local_accum;
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/**
 * @brief SpMMCOO for graphSum
 */
template <typename IdType, typename DType, bool UseEMap>
__global__ void _SegmentSumCOOKernel(IdType* target, IdType* EMap, DType* data,
                                     int64_t E, int powk, int out_len,
                                     DType* out) {
  // SPMM with COO.
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
  const IdType stride_y = blockDim.y * gridDim.y;
  const int64_t stride_x = blockDim.x * gridDim.x;
  while (ty < E) {
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const IdType data_idx = UseEMap ? EMap[ty] : ty;
    const DType* dataoff = data + data_idx * out_len;
    DType* outoff = out + target[ty] * out_len;
    while (tx < out_len) {
      DType val = powk == 1 ? dataoff[tx] : __powf(dataoff[tx], powk);
      AtomicAdd(outoff + tx, val);
      tx += stride_x;
    }
    ty += stride_y;
  }
}

template <typename IdType, typename DType>
void CSCSum(torch::Tensor indptr, torch::optional<torch::Tensor> e_ids,
            torch::optional<torch::Tensor> n_ids, torch::Tensor data,
            torch::Tensor out_data, int64_t powk) {
  auto num_element = indptr.numel() - 1;
  auto use_n_map = n_ids.has_value(), use_e_map = e_ids.has_value();
  auto n_ids_map = use_n_map ? n_ids.value().data_ptr<IdType>() : nullptr;
  auto e_ids_map = use_e_map ? e_ids.value().data_ptr<IdType>() : nullptr;

  // Aligning DGL
  const int out_len = 1;

  const int ntx = 1;
  const int nty = 256;
  const int nby = (out_len + ntx - 1) / ntx;
  const int nbx = (num_element + nty - 1) / nty;
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  SWITCH_IDX(use_e_map, use_n_map, {
    CUDA_KERNEL_CALL((_SegmentSumKernel<IdType, DType, UseEMap, UseNMap>),
                     nblks, nthrs, indptr.data_ptr<IdType>(), e_ids_map,
                     n_ids_map, data.data_ptr<DType>(), num_element, powk,
                     out_len, out_data.data_ptr<DType>());
  });
}

void CSCSumCUDA(torch::Tensor indptr, torch::optional<torch::Tensor> e_ids,
                torch::optional<torch::Tensor> n_ids, torch::Tensor data,
                torch::Tensor out_data, int64_t powk) {
  CSCSum<int64_t, float>(indptr, e_ids, n_ids, data, out_data, powk);
}

template <typename IdType, typename DType>
void COOSum(torch::Tensor target, torch::optional<torch::Tensor> e_ids,
            torch::Tensor data, torch::Tensor out_data, int64_t powk) {
  int64_t E = target.numel();
  auto use_e_map = e_ids.has_value();
  auto e_ids_map = use_e_map ? e_ids.value().data_ptr<IdType>() : nullptr;

  // Aligning DGL
  const int out_len = 1;

  const int ntx = FindNumThreads(out_len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (out_len + ntx - 1) / ntx;
  const int nby = FindNumBlocks<'y'>((E + nty - 1) / nty);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  SWITCH_IDX(use_e_map, false, {
    CUDA_KERNEL_CALL((_SegmentSumCOOKernel<IdType, DType, UseEMap>), nblks,
                     nthrs, target.data_ptr<IdType>(), e_ids_map,
                     data.data_ptr<DType>(), E, powk, out_len,
                     out_data.data_ptr<DType>());
  });
}

void COOSumCUDA(torch::Tensor target, torch::optional<torch::Tensor> e_ids,
                torch::Tensor data, torch::Tensor out_data, int64_t powk) {
  COOSum<int64_t, float>(target, e_ids, data, out_data, powk);
}

#define SWITCH_NID(NODE_MAP, ...)    \
  do {                               \
    if ((NODE_MAP)) {                \
      constexpr bool UseNid = true;  \
      { __VA_ARGS__ }                \
    } else {                         \
      constexpr bool UseNid = false; \
      { __VA_ARGS__ }                \
    }                                \
  } while (0)

template <typename IdType>
__global__ void _CSCSplitKernel(IdType* indptr, IdType* out, IdType* out_sizes,
                                int64_t num_split, int64_t split_size,
                                int64_t last_size) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
  while (row < num_split) {
    int64_t inoff = row * split_size;
    int64_t outoff = row * (split_size + 1);
    int64_t size = (row == num_split - 1) ? last_size : split_size;
    IdType prefix = indptr[inoff];
    out[outoff] = 0;
    out_sizes[row] = indptr[inoff + size] - prefix;
    for (int idx = threadIdx.x; idx < size; idx += blockDim.x) {
      out[outoff + idx + 1] = indptr[inoff + idx + 1] - prefix;
    }
    row += gridDim.x * blockDim.y;
  }
}

template <typename IdType, bool UseNid>
__global__ void _CSCSplitIndptrKernel(IdType* indptr, IdType* nid, IdType** out,
                                      IdType** out_nid, IdType* out_range,
                                      int64_t num_split, int64_t split_size,
                                      int64_t last_size) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
  while (row < num_split) {
    int64_t inoff = row * split_size;
    int64_t size = (row == num_split - 1) ? last_size : split_size;
    IdType prefix = indptr[inoff];
    out[row][0] = 0;
    out_range[row + 1] = indptr[inoff + size];
    for (int idx = threadIdx.x; idx < size; idx += blockDim.x) {
      out[row][idx + 1] = indptr[inoff + idx + 1] - prefix;
      if (UseNid) out_nid[row][idx] = nid[inoff + idx];
    }
    row += gridDim.x * blockDim.y;
  }
}

template <typename IdType>
__global__ void _CSCSplitIndicesKernel(IdType* indices, IdType* split_index,
                                       IdType** out, int64_t num_split) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
  while (row < num_split) {
    int64_t size = split_index[row + 1] - split_index[row];
    for (int idx = threadIdx.x; idx < size; idx += blockDim.x) {
      out[row][idx] = indices[split_index[row] + idx];
    }
    row += gridDim.x * blockDim.y;
  }
}

// template <typename IdType>
// std::vector<std::vector<torch::Tensor>> _CSCSplit(
//     torch::Tensor indptr, torch::Tensor indices,
//     torch::optional<torch::Tensor> nid, int64_t split_size) {
//   std::vector<torch::Tensor> sub_indptrs;
//   std::vector<torch::Tensor> sub_indices;
//   std::vector<torch::Tensor> sub_nids;
//   torch::Tensor sub_ind;

//   auto use_nid = nid.has_value();
//   auto total_element = indptr.numel() - 1;
//   auto num_split = (total_element + split_size - 1) / split_size;
//   auto last_size = total_element - (num_split - 1) * split_size;

//   torch::Tensor indices_range = torch::zeros(num_split + 1,
//   indices.options()); IdType *indptr_data = indptr.data_ptr<IdType>(),
//          *range_data = indices_range.data_ptr<IdType>();
//   IdType* nid_data = use_nid ? nid.value().data_ptr<IdType>() : nullptr;
//   IdType* temp[num_split];
//   IdType* nid_temp[num_split];
//   IdType **d_ptr, **d_ptr_nid;
//   d_ptr = (IdType**)c10::cuda::CUDACachingAllocator::raw_alloc(
//       sizeof(IdType**) * num_split);
//   if (use_nid)
//     d_ptr_nid = (IdType**)c10::cuda::CUDACachingAllocator::raw_alloc(
//         sizeof(IdType**) * num_split);

//   for (int i = 0; i < num_split; ++i) {
//     auto size = (i == num_split - 1) ? last_size + 1 : split_size + 1;
//     sub_ind = torch::empty(size, indptr.options());
//     sub_indptrs.push_back(sub_ind);
//     temp[i] = sub_ind.data_ptr<IdType>();
//     if (use_nid) {
//       sub_ind = torch::empty(size - 1, indptr.options());
//       sub_nids.push_back(sub_ind);
//       nid_temp[i] = sub_ind.data_ptr<IdType>();
//     }
//   }
//   cudaMemcpy(d_ptr, temp, sizeof(IdType**) * num_split,
//   cudaMemcpyHostToDevice); if (use_nid)
//     cudaMemcpy(d_ptr_nid, nid_temp, sizeof(IdType**) * num_split,
//                cudaMemcpyHostToDevice);

//   dim3 block(64, 8);
//   dim3 grid((num_split + block.y - 1) / block.y);
//   SWITCH_NID(use_nid, {
//     CUDA_KERNEL_CALL((_CSCSplitIndptrKernel<IdType, UseNid>), grid, block,
//                      indptr_data, nid_data, d_ptr, d_ptr_nid, range_data,
//                      num_split, split_size, last_size);
//   });
//   auto range_host = indices_range.to(torch::kCPU, false, true);
//   range_data = range_host.data_ptr<IdType>();
//   for (int i = 0; i < num_split; ++i) {
//     sub_ind =
//         torch::empty(range_data[i + 1] - range_data[i], indices.options());
//     sub_indices.push_back(sub_ind);
//     temp[i] = sub_ind.data_ptr<IdType>();
//   }
//   cudaMemcpy(d_ptr, temp, sizeof(IdType**) * num_split,
//   cudaMemcpyHostToDevice);

//   dim3 nthrs(64, 8);
//   dim3 nblks((num_split + block.y - 1) / block.y);
//   CUDA_KERNEL_CALL((_CSCSplitIndicesKernel<IdType>), nblks, nthrs,
//                    indices.data_ptr<IdType>(),
//                    indices_range.data_ptr<IdType>(), d_ptr, num_split);

//   c10::cuda::CUDACachingAllocator::raw_delete(d_ptr);
//   if (use_nid) c10::cuda::CUDACachingAllocator::raw_delete(d_ptr_nid);

//   return {sub_indptrs, sub_indices, sub_nids};
// }

template <typename IdType>
std::vector<std::vector<torch::Tensor>> _CSCSplit(
    torch::Tensor indptr, torch::Tensor indices,
    torch::optional<torch::Tensor> nid, int64_t split_size) {
  std::vector<torch::Tensor> sub_indptrs;
  std::vector<torch::Tensor> sub_indices;
  std::vector<torch::Tensor> sub_nids;

  if (nid.has_value()) sub_nids = torch::split(nid.value(), split_size);

  auto total_element = indptr.numel() - 1;
  auto num_split = total_element / split_size;
  auto redundant = total_element - num_split * split_size;
  int64_t total_indptrs_len;
  if (redundant != 0) {
    total_indptrs_len = num_split * (split_size + 1) + redundant + 1;
    num_split += 1;
  } else {
    total_indptrs_len = num_split * (split_size + 1);
    redundant = split_size;
  }
  auto total_indptrs = torch::empty(total_indptrs_len, indptr.options());
  auto indices_split_sizes = torch::empty(num_split, indices.options());

  dim3 block(64, 8);
  dim3 grid((num_split + block.y - 1) / block.y);
  CUDA_KERNEL_CALL((_CSCSplitKernel<IdType>), grid, block,
                   indptr.data_ptr<IdType>(), total_indptrs.data_ptr<IdType>(),
                   indices_split_sizes.data_ptr<IdType>(), num_split,
                   split_size, redundant);

  sub_indptrs = torch::split(total_indptrs, split_size + 1);

  indices_split_sizes = indices_split_sizes.to(torch::kCPU);
  auto data_ptr = indices_split_sizes.data_ptr<IdType>();
  std::vector<IdType> subindices_sizes(data_ptr, data_ptr + num_split);
  sub_indices = torch::split_with_sizes(indices, subindices_sizes);

  return {sub_indptrs, sub_indices, sub_nids};
}

std::vector<std::vector<torch::Tensor>> CSCSplitCUDA(
    torch::Tensor indptr, torch::Tensor indices,
    torch::optional<torch::Tensor> nid, int64_t split_size) {
  return _CSCSplit<int64_t>(indptr, indices, nid, split_size);
}
}  // namespace impl
}  // namespace gs