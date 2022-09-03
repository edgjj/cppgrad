#include <cppgrad/cppgrad.hpp>
#include <gtest/gtest.h>

using namespace cppgrad;

TEST(SanityTest, MicrogradInheritedSuite)
{
    Value x = -4.0;
    auto z = Value{ 2.0 } * x + Value{ 2.0 } + x;
    auto q = z.relu() + z * x;
    auto h = (z * z).relu();
    auto y = h + q + q * x;
    y.backward();
}

TEST(MoreOpsTest, MicrogradInheritedSuite)
{
	//boost::beast::

	EXPECT_EQ(7 * 6, 42);
}