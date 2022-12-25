#include <cppgrad/cppgrad.hpp>
#include <gtest/gtest.h>

#include <cmath>

using namespace cppgrad;

TEST(AutogradTests, MicrogradTest1)
{
    auto x = Tensor(-2.0);
    x.set_requires_grad(true);

    auto z = Tensor(2.0) * x + Tensor(2.0) + x;

    auto h = relu(z * z);
    auto q = relu(z) + z * x;

    auto y = h + q + q * x;
    y.backward();

    EXPECT_DOUBLE_EQ(x.grad().item<f64>(), -6.0);
    EXPECT_DOUBLE_EQ(y.item<f64>(), 8.0);
}

TEST(AutogradTests, MicrogradTest2)
{
    auto a = Tensor(-4.0);
    auto b = Tensor(2.0);
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    auto c = a + b;
    auto d = a * b + pow(b, 3.0); 
    c += c + 1.0;

    c += 1.0 + c + (-a);
    d += d * 2.0 + relu(b + a);
    d += 3.0 * d + relu(b - a);
    auto e = c - d;
    auto f = pow(e, 2.0);
    auto g = f / 2.0;
    g += 10.0 / f;

    g.backward();

    EXPECT_NEAR(a.grad().item<f64>(), 138.8338192419825, 1e-8);
    EXPECT_NEAR(b.grad().item<f64>(), 645.5772594752186, 1e-8);
    EXPECT_NEAR(c.grad().item<f64>(), -6.941690962099126, 1e-8);

    EXPECT_NEAR(c.item<f64>(), -1.0, 1e-8);
    EXPECT_NEAR(d.item<f64>(), 6.0, 1e-8);
    EXPECT_NEAR(e.item<f64>(), -7.0, 1e-8);
    EXPECT_NEAR(f.item<f64>(), 49.0, 1e-8);
    EXPECT_NEAR(g.item<f64>(), 24.70408163265306, 1e-8);
}