#include <cppgrad/cppgrad.hpp>
#include <gtest/gtest.h>

using namespace cppgrad;

TEST(SanityTest, MicrogradInheritedSuite)
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

TEST(MoreOpsTest, MicrogradInheritedSuite)
{
    auto a = Value(-4.0);
    auto b = Value(2.0);
    auto c = a + b;
    auto d = a * b + b.pow(3.0);
    c += c - 1.0;
    c += 1.0 + c + (-a);
    d += d * 2 + (b + a).relu();
    d += 3.0 * d + (b - a).relu();
    auto e = c - d;
    auto f = e.pow(2);
    auto g = f / 2.0;
    g -= 10.0 / f;
    g += 0.2;
    g.backward();

    std::cout << "a grad: " << a.grad() << '\n';
    std::cout << "b grad: " << b.grad() << '\n';
    std::cout << "c grad: " << c.grad() << '\n';

    std::cout << "a value: " << a.data() << '\n';
    std::cout << "b value: " << b.data() << '\n';
    std::cout << "c value: " << c.data() << '\n';
    std::cout << "d value: " << d.data() << '\n';
    std::cout << "e value: " << e.data() << '\n';
    std::cout << "f value: " << f.data() << '\n';
    std::cout << "g value: " << g.data() << '\n';


}