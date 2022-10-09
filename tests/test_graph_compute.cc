#include <gtest/gtest.h>
#include "graph.h"
#include <torch/torch.h>
#include <chrono>

using namespace gs;

TEST(GraphSum, test1)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto data_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor indices = torch::arange(0, 100, options);
    torch::Tensor expected = torch::ones(20, data_options) * 5;
    A.LoadCSC(indptr, indices);

    auto result = A.Sum(0, 1);

    EXPECT_EQ(result.numel(), expected.numel());
    EXPECT_TRUE(result.equal(expected));
}

TEST(GraphSum, test2)
{
    Graph A(false);
    auto options = torch::dtype(torch::kInt64).device(torch::kCUDA);
    auto data_options = torch::dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor indices = torch::arange(0, 100, options);
    torch::Tensor data = torch::arange(1, 6, data_options);
    torch::Tensor expected = (data.sum()).repeat({20});
    A.LoadCSC(indptr, indices);
    A.SetData(data.repeat({20}));

    auto result = A.Sum(0, 1);

    EXPECT_EQ(result.numel(), expected.numel());
    EXPECT_TRUE(result.equal(expected));
}

TEST(GraphSum, test3)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto data_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor indices = torch::arange(0, 100, options);
    torch::Tensor expected = torch::ones(20, data_options) * 5;
    A.LoadCSC(indptr, indices);

    auto result = A.Sum(0, 2);

    EXPECT_EQ(result.numel(), expected.numel());
    EXPECT_TRUE(result.equal(expected));
}

TEST(GraphSum, test4)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto data_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor indices = torch::arange(0, 100, options);
    torch::Tensor data = torch::arange(1, 6, data_options);
    torch::Tensor expected = (data.pow(2).sum()).repeat({20});
    A.LoadCSC(indptr, indices);
    A.SetData(data.repeat({20}));

    auto result = A.Sum(0, 2);

    EXPECT_EQ(result.numel(), expected.numel());
    EXPECT_TRUE(result.equal(expected));
}

TEST(GraphDiv, test1)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto data_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor indices = torch::arange(0, 100, options);
    torch::Tensor divisor = torch::ones(20, data_options) * 10;
    torch::Tensor expected = torch::ones(100, data_options) / 10;
    A.LoadCSC(indptr, indices);

    auto graph_ptr = A.Divide(divisor, 0);
    auto result = graph_ptr->GetData().value();

    EXPECT_EQ(result.numel(), expected.numel());
    EXPECT_TRUE(result.isclose(expected).all().item<bool>());
}

TEST(GraphDiv, test2)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto data_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor indices = torch::arange(0, 100, options);
    torch::Tensor divisor = torch::arange(1, 21, data_options);
    torch::Tensor expected = torch::repeat_interleave(torch::ones(20, data_options) / divisor, 5);
    A.LoadCSC(indptr, indices);

    auto graph_ptr = A.Divide(divisor, 0);
    auto result = graph_ptr->GetData().value();

    EXPECT_EQ(result.numel(), expected.numel());
    EXPECT_TRUE(result.isclose(expected).all().item<bool>());
}

TEST(GraphNormalize, test1)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto data_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor indices = torch::arange(0, 100, options);
    torch::Tensor expected = torch::ones(100, data_options) / 5;
    A.LoadCSC(indptr, indices);

    auto graph_ptr = A.Normalize(0);
    auto result = graph_ptr->GetData().value();

    EXPECT_EQ(result.numel(), expected.numel());
    EXPECT_TRUE(result.isclose(expected).all().item<bool>());
}

TEST(GraphNormalize, test2)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto data_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor indices = torch::arange(0, 100, options);
    torch::Tensor data = torch::arange(1, 6, data_options);
    torch::Tensor expected = (data / data.sum()).repeat({20});
    A.LoadCSC(indptr, indices);
    A.SetData(data.repeat({20}));

    auto graph_ptr = A.Normalize(0);
    auto result = graph_ptr->GetData().value();

    EXPECT_EQ(result.numel(), expected.numel());
    EXPECT_TRUE(result.isclose(expected).all().item<bool>());
}
