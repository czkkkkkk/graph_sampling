#include <gtest/gtest.h>
#include <torch/torch.h>
#include "graph.h"

using namespace gs;

TEST(SpMM, u_mul_e_sum1)
{
    Graph A(false);
    auto Ioptions = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto Doptions = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor coo_row = torch::arange(0, 100, Ioptions);
    torch::Tensor coo_col = torch::arange(0, 100, Ioptions);
    torch::Tensor lhs = torch::arange(0, 100, Doptions);
    torch::Tensor rhs = torch::arange(0, 100, Doptions);
    torch::Tensor out = torch::zeros(100, Doptions);
    torch::Tensor res = torch::pow(torch::arange(0, 100, Doptions), 2);
    A.LoadCOO(coo_row, coo_col);

    A.SpMM("mul", "sum", lhs, rhs, out, torch::Tensor(), torch::Tensor(), _COO);
    EXPECT_TRUE(res.equal(out));
}

TEST(SpMM, u_mul_e_sum2)
{
    Graph A(false);
    auto Ioptions = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto Doptions = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor nid = torch::arange(0, 20, Ioptions);
    torch::Tensor indptr = torch::arange(0, 21, Ioptions) * 5;
    torch::Tensor indices = torch::arange(0, 100, Ioptions);
    torch::Tensor lhs = torch::arange(0, 100, Doptions);
    torch::Tensor rhs = torch::arange(0, 100, Doptions);
    torch::Tensor out = torch::zeros(20, Doptions);
    std::vector<float> seg_sum;
    for (int i = 0; i < 20; i++)
    {
        float temp = 0;
        for (int j = i * 5; j < (i + 1) * 5; j++)
        {
            temp += j * j;
        }
        seg_sum.push_back(temp);
    }
    torch::Tensor res = torch::from_blob(seg_sum.data(), 20, torch::dtype(torch::kFloat32)).to(torch::kCUDA);
    A.LoadCSCWithColIds(nid, indptr, indices);

    A.SpMM("mul", "sum", lhs, rhs, out, torch::Tensor(), torch::Tensor(), _COO);
    EXPECT_TRUE(res.equal(out));
}

TEST(SpMM, u_mul_e_sum3)
{
    Graph A(false);
    auto Ioptions = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto Doptions = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor nid = torch::arange(0, 20, Ioptions);
    torch::Tensor indptr = torch::arange(0, 21, Ioptions) * 5;
    torch::Tensor indices = torch::arange(0, 100, Ioptions);
    torch::Tensor lhs = torch::ones({100, 64}, Doptions) * -2;
    torch::Tensor rhs = torch::ones({100, 64}, Doptions) * 4;
    torch::Tensor out = torch::zeros({20, 64}, Doptions);
    torch::Tensor res = torch::ones({20, 64}, Doptions) * -40;
    A.LoadCSCWithColIds(nid, indptr, indices);

    A.SpMM("mul", "sum", lhs, rhs, out, torch::Tensor(), torch::Tensor(), _COO);
    EXPECT_TRUE(res.equal(out));
}
