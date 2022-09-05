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

    std::cout << "current x grad value: " << x.get_grad() << '\n';
    std::cout << "current y value: " << y.get_data() << '\n';
}

//TEST(MoreOpsTest, MicrogradInheritedSuite)
//{
//	//boost::beast::
//
//	EXPECT_EQ(7 * 6, 42);
//}