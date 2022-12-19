#include <cppgrad/cppgrad.hpp>
#include <cppgrad/exceptions/index_error.hpp>
#include <cppgrad/exceptions/out_of_memory.hpp>
#include <cppgrad/exceptions/type_error.hpp>
#include <gtest/gtest.h>

#include <random>

using namespace cppgrad;

#ifdef CPPGRAD_HAS_CUDA

TEST(TensorCUDAOpsTests, SumTest)
{
    Tensor t1 = Tensor { 2, 3, 4, 5, 6, 7 }.cuda();
    Tensor t2 = Tensor { 9, 9, 9, 9, 9, 9 }.cuda();

    auto t3 = t1 + t2;

    ASSERT_EQ(t3[0].item<i32>(), 11);
    ASSERT_EQ(t3[1].item<i32>(), 12);
    ASSERT_EQ(t3[2].item<i32>(), 13);
    ASSERT_EQ(t3[3].item<i32>(), 14);
    ASSERT_EQ(t3[4].item<i32>(), 15);
    ASSERT_EQ(t3[5].item<i32>(), 16);
}

TEST(TensorCUDAOpsTests, SubTest)
{
    Tensor t1 = Tensor { 2, 3, 4, 5, 6, 7 }.cuda();
    Tensor t2 = Tensor { 9, 9, 9, 9, 9, 9 }.cuda();

    auto t3 = t1 - t2;

    ASSERT_EQ(t3[0].item<i32>(), -7);
    ASSERT_EQ(t3[1].item<i32>(), -6);
    ASSERT_EQ(t3[2].item<i32>(), -5);
    ASSERT_EQ(t3[3].item<i32>(), -4);
    ASSERT_EQ(t3[4].item<i32>(), -3);
    ASSERT_EQ(t3[5].item<i32>(), -2);
}

TEST(TensorCUDAOpsTests, MulTest)
{
    Tensor t1 = Tensor { 2, 3, 4, 5, 6, 7 }.cuda();
    Tensor t2 = Tensor { 10, 10, 10, 10, 10, 10 }.cuda();

    auto t3 = t1 * t2;

    ASSERT_EQ(t3[0].item<i32>(), 20);
    ASSERT_EQ(t3[1].item<i32>(), 30);
    ASSERT_EQ(t3[2].item<i32>(), 40);
    ASSERT_EQ(t3[3].item<i32>(), 50);
    ASSERT_EQ(t3[4].item<i32>(), 60);
    ASSERT_EQ(t3[5].item<i32>(), 70);
}

TEST(TensorCUDAOpsTests, DotTest)
{
    Tensor t1 = Tensor { 2, 3, 4, 5, 6, 7 }.cuda();
    Tensor t2 = Tensor { 10, 20, 12, 55, 23, 44 }.cuda();

    auto t3 = cppgrad::mm(t1, t2);

    ASSERT_EQ(t3.item<i32>(), 849);
}

TEST(TensorCUDAOpsTests, DotMultiBlockTest)
{
    std::mt19937 engine(std::random_device {}());
    std::uniform_int_distribution<int> dist { 1, 10 };

    std::vector<int> v1, v2;
    // occupy 32 blocks
    size_t num_elements = 128 * 32;
    for (size_t i = 0; i < num_elements; i++) {
        v1.push_back(dist(engine));
        v2.push_back(dist(engine));
    }

    auto t1 = Tensor::from_blob<i32, CUDA>(v1.data(), { num_elements });
    auto t2 = Tensor::from_blob<i32, CUDA>(v2.data(), { num_elements });
    ASSERT_EQ(t1[0].item<i32>(), v1[0]);

    auto t3 = cppgrad::mm(t1, t2);

    auto value = std::inner_product(v1.begin(), v1.end(), v2.begin(), 0);

    ASSERT_EQ(t3.item<i32>(), value);
}

TEST(TensorCUDAOpsTests, PowTest)
{
    Tensor t1 = Tensor { 2, 3, 4, 5, 6, 7 }.cuda();
    Tensor t2 = Tensor { 5, 5, 5, 5, 5, 5 }.cuda();

    auto t3 = cppgrad::pow(t1, t2);

    ASSERT_EQ(t3[0].item<i32>(), std::pow(2, 5));
    ASSERT_EQ(t3[1].item<i32>(), std::pow(3, 5));
    ASSERT_EQ(t3[2].item<i32>(), std::pow(4, 5));
    ASSERT_EQ(t3[3].item<i32>(), std::pow(5, 5));
    ASSERT_EQ(t3[4].item<i32>(), std::pow(6, 5));
    ASSERT_EQ(t3[5].item<i32>(), std::pow(7, 5));
}

TEST(TensorCUDAOpsTests, MatmulTestEqShape)
{
    Tensor t1 = Tensor {
        { 1, 2, 3 },
        { 9, 4, 5 },
        { 6, 4, 2 }
    }.cuda();

    Tensor t2 = Tensor {
        { 4, 3, 2 },
        { 1, 0, 1 },
        { 9, 3, 3 }
    }.cuda();

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

TEST(TensorCUDAOpsTests, MatmulTestNonEqShape)
{
    Tensor t1 = Tensor {
        { 1, 2, 3 },
        { 9, 4, 5 }
    }.cuda();

    Tensor t2 = Tensor {
        { 4, 3 },
        { 1, 0 },
        { 9, 3 }
    }.cuda();

    auto t3 = cppgrad::mm(t1, t2);
    std::vector<size_t> shape { 2, 2 };

    ASSERT_EQ(t3.shape(), shape);

    ASSERT_EQ(t3(0, 0).item<i32>(), 33);
    ASSERT_EQ(t3(0, 1).item<i32>(), 12);
    ASSERT_EQ(t3(1, 0).item<i32>(), 85);
    ASSERT_EQ(t3(1, 1).item<i32>(), 42);
}

#endif