#include <gtest/gtest.h>
#include <torch/torch.h>
#include "graph.h"
#include "kernel.h"

using namespace gs;

TEST(SDDMM, add_test1)
{
    auto A = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(new Graph(false)));
    auto Ioptions = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto Doptions = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor coo_row = torch::arange(0, 100, Ioptions);
    torch::Tensor coo_col = torch::arange(0, 100, Ioptions);
    torch::Tensor lhs = torch::arange(0, 100, Doptions);
    torch::Tensor rhs = torch::arange(0, 100, Doptions);
    torch::Tensor out = torch::zeros(100, Doptions);
    torch::Tensor res = torch::arange(0, 100, Doptions) * 2;
    auto coo_ptr = std::make_shared<COO>(COO{coo_row, coo_col, torch::nullopt});
    A->SetCOO(coo_ptr);

    SDDMM("add", A, lhs, rhs, out, 0, 2);
    EXPECT_TRUE(res.equal(out));
}