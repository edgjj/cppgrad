#include <cppgrad/cppgrad.hpp>
#include <cppgrad/exceptions/index_error.hpp>
#include <cppgrad/exceptions/out_of_memory.hpp>
#include <cppgrad/exceptions/type_error.hpp>
#include <gtest/gtest.h>

using namespace cppgrad;

TEST(TensorOpsTests, SumTest)
{
    Tensor t1 = { 2, 3, 4, 5, 6, 7 };
    Tensor t2 = { 9, 9, 9, 9, 9, 9 };

    auto t3 = t1 + t2;

    ASSERT_EQ(t3[0].item<i32>(), 11);
    ASSERT_EQ(t3[1].item<i32>(), 12);
    ASSERT_EQ(t3[2].item<i32>(), 13);
    ASSERT_EQ(t3[3].item<i32>(), 14);
    ASSERT_EQ(t3[4].item<i32>(), 15);
    ASSERT_EQ(t3[5].item<i32>(), 16);
}

TEST(TensorOpsTests, SubTest)
{
    Tensor t1 = { 2, 3, 4, 5, 6, 7 };
    Tensor t2 = { 9, 9, 9, 9, 9, 9 };

    auto t3 = t1 - t2;

    ASSERT_EQ(t3[0].item<i32>(), -7);
    ASSERT_EQ(t3[1].item<i32>(), -6);
    ASSERT_EQ(t3[2].item<i32>(), -5);
    ASSERT_EQ(t3[3].item<i32>(), -4);
    ASSERT_EQ(t3[4].item<i32>(), -3);
    ASSERT_EQ(t3[5].item<i32>(), -2);
}

TEST(TensorOpsTests, MulTest)
{
    Tensor t1 = { 2, 3, 4, 5, 6, 7 };
    Tensor t2 = { 10, 10, 10, 10, 10, 10 };

    auto t3 = t1 * t2;

    ASSERT_EQ(t3[0].item<i32>(), 20);
    ASSERT_EQ(t3[1].item<i32>(), 30);
    ASSERT_EQ(t3[2].item<i32>(), 40);
    ASSERT_EQ(t3[3].item<i32>(), 50);
    ASSERT_EQ(t3[4].item<i32>(), 60);
    ASSERT_EQ(t3[5].item<i32>(), 70);
}

TEST(TensorOpsTests, DotTest)
{
    Tensor t1 = { 2, 3, 4, 5, 6, 7 };
    Tensor t2 = { 10, 20, 12, 55, 23, 44 };

    auto t3 = cppgrad::mm(t1, t2);

    ASSERT_EQ(t3.item<i32>(), 849);
}

TEST(TensorOpsTests, MatmulTestEqShape)
{
    Tensor t1 = {
        { 1, 2, 3 },
        { 9, 4, 5 },
        { 6, 4, 2 }
    };

    Tensor t2 = {
        { 4, 3, 2 },
        { 1, 0, 1 },
        { 9, 3, 3 }
    };

    auto t3 = cppgrad::mm(t1, t2);
    std::vector<size_t> shape { 3, 3 };

    ASSERT_EQ(t3.shape(), shape);

    ASSERT_EQ(t3(0, 0).item<i32>(), 33);
    ASSERT_EQ(t3(0, 1).item<i32>(), 12);
    ASSERT_EQ(t3(0, 2).item<i32>(), 13);
    ASSERT_EQ(t3(1, 0).item<i32>(), 85);
    ASSERT_EQ(t3(1, 1).item<i32>(), 42);
    ASSERT_EQ(t3(1, 2).item<i32>(), 37);
    ASSERT_EQ(t3(2, 0).item<i32>(), 46);
    ASSERT_EQ(t3(2, 1).item<i32>(), 24);
    ASSERT_EQ(t3(2, 2).item<i32>(), 22);
}

TEST(TensorOpsTests, MatmulTestNonEqShape)
{
    Tensor t1 = {
        { 1, 2, 3 },
        { 9, 4, 5 },
    };

    Tensor t2 = {
        { 4, 3 },
        { 1, 0 },
        { 9, 3 },
    };

    auto t3 = cppgrad::mm(t1, t2);
    std::vector<size_t> shape { 2, 2 };

    ASSERT_EQ(t3.shape(), shape);

    ASSERT_EQ(t3(0, 0).item<i32>(), 33);
    ASSERT_EQ(t3(0, 1).item<i32>(), 12);
    ASSERT_EQ(t3(1, 0).item<i32>(), 85);
    ASSERT_EQ(t3(1, 1).item<i32>(), 42);
}