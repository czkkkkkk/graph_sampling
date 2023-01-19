#include <gtest/gtest.h>
#include "graph.h"
#include <torch/torch.h>

using namespace gs;

TEST(ColwiseSlicingRelabel, test1)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    torch::Tensor col_ids = torch::arange(0, 20, options);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor indices = torch::arange(0, 100, options);
    A.LoadCSCWithColIds(col_ids, indptr, indices);

    torch::Tensor ids = torch::arange(1, 4, options);
    auto subA = A.Slicing(ids, 0, _CSC, _CSC, true);
    EXPECT_EQ(A.GetNumCols(), 20);
    EXPECT_EQ(A.GetNumRows(), 100);
    EXPECT_EQ(subA->GetNumCols(), 3);
    EXPECT_EQ(subA->GetNumRows(), 15);

    auto csc_ptr = subA->GetCSC();
    EXPECT_TRUE(csc_ptr->indptr.equal(torch::arange(0, 4, options) * 5));
    EXPECT_TRUE(csc_ptr->indices.equal(torch::arange(0, 15, options)));
}