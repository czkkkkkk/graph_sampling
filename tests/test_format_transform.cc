#include <gtest/gtest.h>
#include "graph.h"
#include <torch/torch.h>

using namespace gs;

TEST(CSC2CSR, test1)
{
    Graph A(true);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    torch::Tensor col_ids = torch::arange(0, 20, options);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor col = torch::repeat_interleave(col_ids, 5);
    torch::Tensor indices = torch::arange(0, 100, options);
    A.LoadCSCWithColIds(col_ids, indptr, indices);
    A.CSC2CSR();

    auto coo_ptr = A.GetCOO();
    EXPECT_TRUE(coo_ptr->row.equal(indices));
    EXPECT_TRUE(coo_ptr->col.equal(col));

    auto csr_ptr = A.GetCSR();
    EXPECT_TRUE(csr_ptr->row_ids.equal(indices));
    EXPECT_TRUE(csr_ptr->indptr.equal(torch::arange(0, 101, options)));
    EXPECT_TRUE(csr_ptr->indices.equal(col));
}
