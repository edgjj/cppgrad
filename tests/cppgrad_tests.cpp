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

//TEST(MoreOpsTest, MicrogradInheritedSuite)
//{
//	//boost::beast::
//
//	EXPECT_EQ(7 * 6, 42);
//}