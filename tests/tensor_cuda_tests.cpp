#include <cppgrad/cppgrad.hpp>
#include <cppgrad/exceptions/out_of_memory.hpp>
#include <gtest/gtest.h>

using namespace cppgrad;

// simply build to nothing if there's no CUDA support

#ifdef CPPGRAD_HAS_CUDA

TEST(TensorCUDATests, InitTensor)
{
    auto tensor = Tensor::create<f32, CUDA>({ 128, 8 });

    auto shape = std::vector<size_t> { 128, 8 };
    auto strides = std::vector<size_t> { 32, 4 };

    ASSERT_EQ(tensor.shape(), shape);
    ASSERT_EQ(tensor.strides(), strides);
}

TEST(TensorCUDATests, ViewTensor)
{
    auto tensor = Tensor::create<f32, CUDA>({ 128, 8 });

    auto new_tensor = tensor[0];

    auto shape = std::vector<size_t> { 8 };
    auto strides = std::vector<size_t> { 4 };

    ASSERT_EQ(new_tensor.shape(), shape);
    ASSERT_EQ(new_tensor.strides(), strides);
}

TEST(TensorCUDATests, AssignTensorScalar)
{
    Tensor t = 10;
    t = t.cuda();

    ASSERT_EQ(t.item<i32>(), 10);
}

TEST(TensorCUDATests, AssignTensorVector)
{
    Tensor t = { 123, 443, 551, 999, 32, 66 };
    t = t.cuda();

    ASSERT_EQ(t[0].item<i32>(), 123);
    ASSERT_EQ(t[1].item<i32>(), 443);
    ASSERT_EQ(t[2].item<i32>(), 551);
    ASSERT_EQ(t[3].item<i32>(), 999);
    ASSERT_EQ(t[4].item<i32>(), 32);
    ASSERT_EQ(t[5].item<i32>(), 66);
}

TEST(TensorCUDATests, OOMTest)
{
    size_t size = (size_t)1 << 48;
    ASSERT_THROW(Tensor::create<i64>({ size }), exceptions::OutOfMemoryError);
}
#endif