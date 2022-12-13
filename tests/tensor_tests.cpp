#include <cppgrad/cppgrad.hpp>
#include <cppgrad/exceptions/out_of_memory.hpp>
#include <gtest/gtest.h>

using namespace cppgrad;

TEST(TensorBasicTests, InitTensor)
{
    auto tensor = Tensor::create<f32>({ 128, 8 });

    auto shape = std::vector<size_t> { 128, 8 };
    auto strides = std::vector<size_t> { 32, 4 };

    ASSERT_EQ(tensor.shape(), shape);
    ASSERT_EQ(tensor.strides(), strides);
}

TEST(TensorBasicTests, ViewTensor)
{
    auto tensor = Tensor::create<f32>({ 128, 8 });

    auto new_tensor = tensor[0];

    auto shape = std::vector<size_t> { 8 };
    auto strides = std::vector<size_t> { 4 };

    ASSERT_EQ(new_tensor.shape(), shape);
    ASSERT_EQ(new_tensor.strides(), strides);
}

TEST(TensorBasicTests, AssignTensorScalar)
{
    Tensor t = 10;
    ASSERT_EQ(t.item<i32>(), 10);
}

TEST(TensorBasicTests, AssignTensorVector)
{
    Tensor t = { 123, 443, 551, 999, 32, 66 };

    ASSERT_EQ(t[0].item<i32>(), 123);
    ASSERT_EQ(t[1].item<i32>(), 443);
    ASSERT_EQ(t[2].item<i32>(), 551);
    ASSERT_EQ(t[3].item<i32>(), 999);
    ASSERT_EQ(t[4].item<i32>(), 32);
    ASSERT_EQ(t[5].item<i32>(), 66);
}

TEST(TensorBasicTests, OOMTest)
{
    // try to allocate 2 ^ 48 elements of i64
    // ~= 281.47 TBytes
    size_t size = (size_t)1 << 48;
    ASSERT_THROW(Tensor::create<i64>({ size }), exceptions::OutOfMemoryError);
}

TEST(TensorBasicTests, TypeErrorTest)
{
    auto tensor = Tensor::create<f32>({ 128, 8 });
    ASSERT_THROW(tensor[0][0].item<i32>(), exceptions::TypeError);
}

TEST(TensorBasicTests, IndexErrorTest)
{
    auto tensor = Tensor::create<f32>({ 128, 8 });
    ASSERT_THROW(tensor[129], exceptions::IndexError);

    auto empty_tensor = Tensor::create<i32>({ 0 });
    ASSERT_THROW(empty_tensor[0], exceptions::IndexError);
}

TEST(TensorBasicTests, AssignTensorMultidimensional)
{
    Tensor t = {
        { { 1, 2, 3 },
            { 4, 5, 6 } },
        { { 7, 8, 9 },
            { 10, 11, 12 } }
    };

    auto shape = std::vector<size_t> { 2, 2, 3 };

    ASSERT_EQ(t.shape(), shape);
    ASSERT_EQ(t.numel(), 12);

    ASSERT_EQ(t(0, 0, 1).item<i32>(), 2);
}