#include <cppgrad/cppgrad.hpp>
#include <gtest/gtest.h>

#include "nn_module_tests.hpp"

using namespace cppgrad;

TEST(MicrogradInherited, SanityTest)
{
    auto x = Value(-2.0);
    auto z = Value(2.0) * x + Value(2.0) + x;

    auto h = (z * z).relu();
    auto q = z.relu() + z * x;

    auto y = h + q + q * x;
   // auto y = h + q;
    y.backward();

    std::cout << "current x grad value: " << x.grad() << '\n';
    std::cout << "current y value: " << y.data() << '\n';

    EXPECT_DOUBLE_EQ(x.grad(), -6.0);
    EXPECT_DOUBLE_EQ(y.data(), 8.0);
}

TEST(MicrogradInherited, MoreOpsTest)
{
    auto a = Value(-4.0);
    auto b = Value(2.0);
    auto c = a + b;
    auto d = a * b + b.pow(3.0);
    c += c + 1.0;
    c += 1.0 + c + (-a);
    d += d * 2 + (b + a).relu();
    d += 3.0 * d + (b - a).relu();
    auto e = c - d;
    auto f = e.pow(2);
    auto g = f / 2.0;
    g += 10.0 / f;
    
    g.backward();

    EXPECT_NEAR(a.grad(), 138.8338192419825, 1e-8);
    EXPECT_NEAR(b.grad(), 645.5772594752186, 1e-8);
    EXPECT_NEAR(c.grad(), -6.941690962099126, 1e-8);

    EXPECT_NEAR(c.data(), -1.0, 1e-8);
    EXPECT_NEAR(d.data(), 6.0, 1e-8);
    EXPECT_NEAR(e.data(), -7.0, 1e-8);
    EXPECT_NEAR(f.data(), 49.0, 1e-8);
    EXPECT_NEAR(g.data(), 24.70408163265306, 1e-8);

}