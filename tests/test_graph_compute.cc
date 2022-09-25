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

    auto result = A.Sum(0);

    EXPECT_EQ(result.numel(), expected.numel());
    EXPECT_TRUE(result.equal(expected));
}

TEST(GraphSum, test2)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto data_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor indices = torch::arange(0, 100, options);
    torch::Tensor data = torch::arange(1, 6, data_options);
    torch::Tensor expected = (data.sum()).repeat({20});
    A.LoadCSC(indptr, indices);
    A.SetData(data.repeat({20}));

    auto result = A.Sum(0);

    EXPECT_EQ(result.numel(), expected.numel());
    EXPECT_TRUE(result.equal(expected));
}

TEST(GraphL2Norm, test1)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto data_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor indices = torch::arange(0, 100, options);
    torch::Tensor expected = torch::ones(20, data_options) * 5;
    A.LoadCSC(indptr, indices);

    auto result = A.L2Norm(0);

    EXPECT_EQ(result.numel(), expected.numel());
    EXPECT_TRUE(result.equal(expected));
}

TEST(GraphL2Norm, test2)
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

    auto result = A.L2Norm(0);

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
    auto result = graph_ptr->GetData();

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
    auto result = graph_ptr->GetData();

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
    auto result = graph_ptr->GetData();

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
    auto result = graph_ptr->GetData();

    EXPECT_EQ(result.numel(), expected.numel());
    EXPECT_TRUE(result.isclose(expected).all().item<bool>());
}

void Data1Index(int num_cols, int num_rows)
{
    Graph A(true);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto data_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto indptr = torch::arange(0, num_cols + 1, options) * num_rows;
    auto indices = torch::arange(0, num_cols * num_rows, options);
    auto e_ids = indices.flip({0});
    auto divisor = torch::arange(1, num_cols + 1, data_options);
    auto expected = torch::repeat_interleave(torch::ones(num_cols, data_options) / divisor, num_rows);
    auto csc_ptr = std::make_shared<CSC>(CSC{indptr, indices, e_ids});
    A.SetCSC(csc_ptr);

    torch::cuda::synchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    auto graph_ptr = A.Divide(divisor, 0);
    torch::cuda::synchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Data1Index with " << num_cols << " columns and " << num_rows << " rows takes "
              << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0
              << " milliseconds";

    auto result = graph_ptr->GetData();

    EXPECT_EQ(result.numel(), expected.numel());
    EXPECT_TRUE(result.isclose(expected).all().item<bool>());
}

void Data2Index(int num_cols, int num_rows)
{
    Graph A(true);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto data_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto indptr = torch::arange(0, num_cols + 1, options) * num_rows;
    auto indices = torch::arange(0, num_cols * num_rows, options);
    auto e_ids = indices.flip({0});
    auto divisor = torch::arange(1, num_cols + 1, data_options);
    auto expected = torch::repeat_interleave(torch::ones(num_cols, data_options) / divisor, num_rows).flip({0});
    auto csc_ptr = std::make_shared<CSC>(CSC{indptr, indices, e_ids});
    A.SetCSC(csc_ptr);

    torch::cuda::synchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    auto graph_ptr = A.Divide_2index(divisor, 0);
    torch::cuda::synchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Data2Index with " << num_cols << " columns and " << num_rows << " rows takes "
              << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0
              << " milliseconds";

    auto result = graph_ptr->GetData();

    EXPECT_EQ(result.numel(), expected.numel());
    EXPECT_TRUE(result.isclose(expected).all().item<bool>());
}

TEST(DataIndexing, speed_test)
{
    int num_cols[] = {100, 100, 500, 1000, 1000, 1000, 1000, 2000, 5000, 10000};
    int num_rows[] = {100, 1000, 1000, 1000, 2000, 5000, 10000, 10000, 10000, 10000};
    for (int i = 0; i < 10; i++)
    {
        Data1Index(num_cols[i], num_rows[i]);
        Data2Index(num_cols[i], num_rows[i]);
    }
    Data1Index(10, 10000000);
    Data2Index(10, 10000000);
    Data1Index(10000000, 10);
    Data2Index(10000000, 10);
}
