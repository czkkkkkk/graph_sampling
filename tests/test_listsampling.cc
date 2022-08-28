#include <gtest/gtest.h>
#include "tensor_ops.h"
#include <torch/torch.h>

using namespace gs;

TEST(ListSampling, test1)
{
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    int64_t _data[] = {0, 1, 2, 3, 4, 5};
    torch::Tensor data = torch::from_blob(_data, {6}, options).to(torch::kCUDA);
    torch::Tensor select;
    torch::Tensor index;

    std::tie(select, index) = ListSampling(data, 10, false);
    EXPECT_TRUE(select.equal(data));
}

TEST(ListSampling, test2)
{
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    int64_t _data[] = {0, 1, 2, 3, 4, 5};
    torch::Tensor data = torch::from_blob(_data, {6}, options).to(torch::kCUDA);
    torch::Tensor select;
    torch::Tensor index;

    std::tie(select, index) = ListSampling(data, 10, true);
    ASSERT_TRUE(select.numel() == 10);
    EXPECT_FALSE(select.equal(data));
    ASSERT_TRUE(std::get<0>(torch::_unique(select)).numel() <= 6);
}

TEST(ListSampling, test3)
{
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    int64_t _data[] = {0, 1, 2, 3, 4, 5};
    torch::Tensor data = torch::from_blob(_data, {6}, options).to(torch::kCUDA);
    torch::Tensor select;
    torch::Tensor index;

    std::tie(select, index) = ListSampling(data, 5, false);
    ASSERT_TRUE(select.numel() == 5);
    ASSERT_TRUE(std::get<0>(torch::_unique(select)).numel() == 5);
}