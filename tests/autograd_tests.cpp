// Copyright (c) 2023 Yegor Suslin
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include <cppgrad/cppgrad.hpp>
#include <gtest/gtest.h>

#include <cmath>

using namespace cppgrad;

/**
 *  Ensure that Sum, Mul, Relu backwards are ok.
 */
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

/**
 *  Ensure that Sum, Div, Mul, Relu, Pow, Neg backwards are ok.
 *  Also checks += operator.
 */
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

/**
 *  Ensure that Matmul, Sum backwards are ok.
 */
TEST(AutogradTests, MatmulSumTest)
{
    Tensor t1 = Tensor {
        { 1, 2, 3 },
        { 9, 4, 5 }
    };

    Tensor t2 = Tensor {
        { 4, 3 },
        { 1, 0 },
        { 9, 3 }
    };

    t1.set_requires_grad(true);
    t2.set_requires_grad(true);

    Tensor t3 = Tensor {
        { 1, 4 },
        { 3, 1 }
    };

    auto v1 = mm(t1, t2);
    v1 += t3;

    v1.backward();

    ASSERT_EQ(t1.grad()(0, 0).item<i32>(), 7);
    ASSERT_EQ(t1.grad()(0, 1).item<i32>(), 1);
    ASSERT_EQ(t1.grad()(0, 2).item<i32>(), 12);
    ASSERT_EQ(t1.grad()(1, 0).item<i32>(), 7);
    ASSERT_EQ(t1.grad()(1, 1).item<i32>(), 1);
    ASSERT_EQ(t1.grad()(1, 2).item<i32>(), 12);

    ASSERT_EQ(t2.grad()(0, 0).item<i32>(), 10);
    ASSERT_EQ(t2.grad()(0, 1).item<i32>(), 10);
    ASSERT_EQ(t2.grad()(1, 0).item<i32>(), 6);
    ASSERT_EQ(t2.grad()(1, 1).item<i32>(), 6);
    ASSERT_EQ(t2.grad()(2, 0).item<i32>(), 8);
    ASSERT_EQ(t2.grad()(2, 1).item<i32>(), 8);
}

/**
 *  Ensure that DotProduct, Mul backwards are ok.
 */
TEST(AutogradTests, DotProductMulTest)
{
    Tensor t1 = Tensor {
        1, 2, 3, 9, 4, 5
    };

    Tensor t2 = Tensor {
        4, 3, 1, 0, 9, 3
    };

    t1.set_requires_grad(true);
    t2.set_requires_grad(true);

    Tensor t3 = Tensor { 123 };

    auto v1 = mm(t1, t2);
    v1 *= t3;

    v1.backward();

    ASSERT_EQ(t1.grad()[0].item<i32>(), 492);
    ASSERT_EQ(t1.grad()[1].item<i32>(), 369);
    ASSERT_EQ(t1.grad()[2].item<i32>(), 123);
    ASSERT_EQ(t1.grad()[3].item<i32>(), 0);
    ASSERT_EQ(t1.grad()[4].item<i32>(), 1107);
    ASSERT_EQ(t1.grad()[5].item<i32>(), 369);

    ASSERT_EQ(t2.grad()[0].item<i32>(), 123);
    ASSERT_EQ(t2.grad()[1].item<i32>(), 246);
    ASSERT_EQ(t2.grad()[2].item<i32>(), 369);
    ASSERT_EQ(t2.grad()[3].item<i32>(), 1107);
    ASSERT_EQ(t2.grad()[4].item<i32>(), 492);
    ASSERT_EQ(t2.grad()[5].item<i32>(), 615);
}

/**
 * Ensures that Mul, Relu, Tanh, Exp, Log, Neg, Sign backwards are ok.
 * Also tests grad resetting.
 */
TEST(AutogradTests, MathOpsNoSignTests)
{
    Tensor t1 { 0.4 };
    Tensor t2 { 0.2 };

    t1.set_requires_grad(true);
    t2.set_requires_grad(true);

    auto v1 = t1 * t2;
    v1 = neg(log(exp(tanh(relu(v1)))));
    v1.backward();

    EXPECT_NEAR(t1.grad().item<f64>(), -0.1987, 1e-4);
    EXPECT_NEAR(t2.grad().item<f64>(), -0.3975, 1e-4);
    EXPECT_NEAR(v1.item<f64>(), -0.0798, 1e-4);

    t1 = 0.1;
    t2 = 0.4;
    ASSERT_FALSE(t1.requires_grad());
    ASSERT_FALSE(t2.requires_grad());

    t1.set_requires_grad(true);
    t2.set_requires_grad(true);

    v1 = t1 * t2;
    v1 = sign(neg(log(exp(tanh(relu(v1))))));
    v1.backward();

    ASSERT_EQ(t1.grad().item<f64>(), 0.0);
    ASSERT_EQ(t2.grad().item<f64>(), 0.0);
    ASSERT_EQ(v1.item<f64>(), -1.0);
}

/**
 *  Typical Dense/Linear layer backward behaivor test.
 *
 *  pseudo:
 *
 *  y_hat = mm(weights, x) + bias
 *  y_hat = relu(y_hat)
 *  loss = MSE(y_hat, y)
 */
TEST(AutogradTest, MatmulSumMSETest)
{
    auto x = Tensor { { 4.2, 6.3, 1.0, 4.0, 2.0, 0.03, 4.3, 0.32 } };

    // let it be 2 output perceptron
    Tensor w {
        { 0.023, 2.11, 0.023, 2.11, 0.023, 2.11, 0.023, 2.11 },
        { 32.023, 2.11, 2.023, 2.11, 0.023, 2.11, 0.723, 2.11 },
    };

    Tensor b { 0.4, 0.2 };
    w.set_requires_grad(true);
    b.set_requires_grad(true);

    auto y_hat = mm(x, w.T()) + b;
    EXPECT_NEAR(y_hat(0, 0).item<f64>(), 23.135999275189, 1e-4);
    EXPECT_NEAR(y_hat(0, 1).item<f64>(), 162.345988261853, 1e-4);

    y_hat = relu(y_hat);

    Tensor y { 0.92, 0.1 };
    auto loss = y_hat - y; // leverage need of unsqueezing things
    loss *= loss;

    loss = sum(loss) / (double)y.numel();
    loss.backward();

    EXPECT_NEAR(loss.item<f64>(), 13408.655664817667, 1e-2);

    EXPECT_NEAR(w.grad()(0, 0).item<f64>(), 9.330719264833e+01, 1e-4);
    EXPECT_NEAR(w.grad()(0, 1).item<f64>(), 1.399607995659e+02, 1e-4);
    EXPECT_NEAR(w.grad()(0, 2).item<f64>(), 2.221599925850e+01, 1e-4);
    EXPECT_NEAR(w.grad()(0, 3).item<f64>(), 8.886399703400e+01, 1e-4);
    EXPECT_NEAR(w.grad()(0, 4).item<f64>(), 4.443199851700e+01, 1e-4);
    EXPECT_NEAR(w.grad()(0, 5).item<f64>(), 6.664799628580e-01, 1e-4);
    EXPECT_NEAR(w.grad()(0, 6).item<f64>(), 9.552880104891e+01, 1e-4);
    EXPECT_NEAR(w.grad()(0, 7).item<f64>(), 7.109119603819e+00, 1e-4);

    EXPECT_NEAR(w.grad()(1, 0).item<f64>(), 6.814331197476e+02, 1e-4);
    EXPECT_NEAR(w.grad()(1, 1).item<f64>(), 1.022149756986e+03, 1e-4);
    EXPECT_NEAR(w.grad()(1, 2).item<f64>(), 1.622459882604e+02, 1e-4);
    EXPECT_NEAR(w.grad()(1, 3).item<f64>(), 6.489839530414e+02, 1e-4);
    EXPECT_NEAR(w.grad()(1, 4).item<f64>(), 3.244919765207e+02, 1e-4);
    EXPECT_NEAR(w.grad()(1, 5).item<f64>(), 4.867379539016e+00, 1e-4);
    EXPECT_NEAR(w.grad()(1, 6).item<f64>(), 6.976577804655e+02, 1e-4);
    EXPECT_NEAR(w.grad()(1, 7).item<f64>(), 5.191871508284e+01, 1e-4);

    EXPECT_NEAR(b.grad()(0, 0).item<f64>(), 22.215999258500, 1e-4);
    EXPECT_NEAR(b.grad()(0, 1).item<f64>(), 162.245988260362, 1e-4);
}

TEST(AutogradTests, SigmoidTest)
{
    auto a = Tensor { 0.4f, 0.6f };
    auto b = Tensor { 0.21f, 0.01f };
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    auto c = cppgrad::mm(a, b);
    c = cppgrad::sigmoid(c);

    c.backward();

    EXPECT_NEAR(a.grad()(0).item<f32>(), 0.0523938276f, 1e-6);
    EXPECT_NEAR(a.grad()(1).item<f32>(), 0.0024949443f, 1e-6);
    EXPECT_NEAR(b.grad()(0).item<f32>(), 0.0997977778f, 1e-6);
    EXPECT_NEAR(b.grad()(1).item<f32>(), 0.1496966630f, 1e-6);
}