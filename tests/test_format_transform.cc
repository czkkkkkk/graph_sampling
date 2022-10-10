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
    EXPECT_EQ(A.GetNumRows(), 100);
    EXPECT_TRUE(A.GetCOORows(true).equal(indices));
    EXPECT_TRUE(A.AllValidNode().equal(indices));
    EXPECT_TRUE(csr_ptr->indptr.equal(torch::arange(0, 101, options)));
    EXPECT_TRUE(csr_ptr->indices.equal(col));
}

TEST(CSC2CSR, test2)
{
    Graph A(true);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    torch::Tensor col_ids = torch::arange(0, 21, options);
    torch::Tensor indptr = torch::cat({torch::zeros(1, options), torch::arange(0, 21, options)}) * 5;
    torch::Tensor col = torch::repeat_interleave(torch::arange(1, 21, options), 5);
    torch::Tensor indices = torch::arange(2, 102, options);
    torch::Tensor valid_nodes = torch::arange(0, 102, options);
    A.LoadCSCWithColIds(col_ids, indptr, indices);
    A.CSC2CSR();

    auto coo_ptr = A.GetCOO();
    EXPECT_TRUE(coo_ptr->row.equal(indices));
    EXPECT_TRUE(coo_ptr->col.equal(col));

    auto csr_ptr = A.GetCSR();
    EXPECT_EQ(A.GetNumRows(), 102);
    EXPECT_TRUE(A.GetCOORows(true).equal(indices));
    EXPECT_TRUE(A.AllValidNode().equal(valid_nodes));
    EXPECT_TRUE(csr_ptr->indptr.equal(torch::cat({torch::zeros(2, options), torch::arange(0, 101, options)})));
    EXPECT_TRUE(csr_ptr->indices.equal(col));
}

TEST(CSR2CSC, test1)
{
    Graph A(true);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    torch::Tensor col_ids = torch::arange(0, 20, options);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor col = torch::repeat_interleave(col_ids, 5);
    torch::Tensor indices = torch::arange(0, 100, options);
    A.LoadCSCWithColIds(col_ids, indptr, indices);
    A.CSC2CSR();
    A.CSR2CSC();

    auto coo_ptr = A.GetCOO();
    EXPECT_TRUE(coo_ptr->row.equal(indices));
    EXPECT_TRUE(coo_ptr->col.equal(col));

    auto csc_ptr = A.GetCSC();
    EXPECT_EQ(A.GetNumRows(), 100);
    EXPECT_TRUE(A.GetCOORows(true).equal(indices));
    EXPECT_TRUE(A.AllValidNode().equal(indices));
    EXPECT_TRUE(csc_ptr->indptr.equal(indptr));
    EXPECT_TRUE(csc_ptr->indices.equal(indices));
}
