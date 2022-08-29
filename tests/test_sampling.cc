#include <gtest/gtest.h>
#include <torch/torch.h>
#include <vector>
#include "graph.h"

using namespace gs;

template <typename Idx>
void _TestCSCSamplingUniform(bool replace) {
  Graph A(false);
  auto option = torch::TensorOptions().dtype(torch::kInt64);
  auto indptr =
      torch::from_blob(std::vector<Idx>({0, 1, 1, 3, 4}).data(), {5}, option)
          .to(torch::kCUDA);
  auto indices =
      torch::from_blob(std::vector<Idx>({4, 0, 1, 2}).data(), {4}, option)
          .to(torch::kCUDA);
  auto res_indptr =
      torch::from_blob(std::vector<Idx>({0, 2, 3}).data(), {3}, option)
          .to(torch::kCUDA);
  auto res_indices =
      torch::from_blob(std::vector<Idx>({0, 1, 2}).data(), {3}, option)
          .to(torch::kCUDA);
  auto column_ids =
      torch::from_blob(std::vector<Idx>({2, 3}).data(), {2}, option)
          .to(torch::kCUDA);
  int64_t fanout = 2;
  A.LoadCSC(indptr, indices);
  auto subA = A.ColumnwiseFusedSlicingAndSampling(column_ids, fanout, replace);
  auto csc_ptr = subA->GetCSC();
  EXPECT_TRUE(csc_ptr->indptr.equal(res_indptr));
  EXPECT_TRUE(csc_ptr->indices.equal(res_indices));
}

TEST(SamplingTest, TestCSCSampling) {
  _TestCSCSamplingUniform<int64_t>(true);
  _TestCSCSamplingUniform<int64_t>(false);
}
