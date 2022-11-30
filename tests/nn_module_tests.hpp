#pragma once

#include <cppgrad/cppgrad.hpp>
#include <gtest/gtest.h>

using namespace cppgrad;

struct ReLU : public nn::Module {
    ReLU(nn::Module* parent)
        : Module(parent)
    {
    }

    template <typename T>
    Value<T> forward(const Value<T>& value)
    {
        return value.relu();
    }
};

struct Linear : public nn::Module {
    Linear(nn::Module* parent)
        : Module(parent)
    {
    }

    template <typename T>
    Value<T> forward(const Value<T>& value)
    {
        return value * 2;
    }
};

struct JustNet : public nn::Module {
    JustNet()
        : linear1(this)
        , relu(this)
    {
        // relu = ReLU(nullptr);
    }

    template <typename T>
    Value<T> forward(const Value<T>& value)
    {
        return value;
    }

private:
    Linear linear1;
    ReLU relu;
};

// TEST(ModulesSuite, BasicTest)
// {
// }