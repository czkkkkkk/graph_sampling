#include <gtest/gtest.h>
#include "graph.h"
#include <torch/torch.h>

using namespace gs;

TEST(ColwiseSlicing, test1)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor indices = torch::arange(0, 100, options);
    A.LoadCSC(indptr, indices);

    torch::Tensor col_ids = torch::arange(0, 3, options);
    auto subA = A.ColumnwiseSlicing(col_ids);
    auto csc_ptr = subA->GetCSC();
    EXPECT_TRUE(csc_ptr->indptr.equal(torch::arange(0, 4, options) * 5));
    EXPECT_TRUE(csc_ptr->indices.equal(torch::arange(0, 15, options)));
}

TEST(FusedRowColSlicing, test1)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor indices = torch::arange(0, 100, options);
    A.LoadCSC(indptr, indices);

    torch::Tensor seeds = torch::arange(0, 5, options);

    auto subA1 = A.ColumnwiseSlicing(seeds)->RowwiseSlicing(seeds);
    auto subA2 = A.FusionSlicing(seeds);

    EXPECT_TRUE(subA1->GetCSC()->indptr.equal(subA2->GetCSC()->indptr));
    EXPECT_TRUE(subA1->GetCSC()->indptr.equal(subA2->GetCSC()->indptr));
}