#ifndef CPPGRAD_CPU_DEFS_HPP
#define CPPGRAD_CPU_DEFS_HPP

#include "cppgrad/device/cpu/static_thread_pool.hpp"
#include <cstddef>

namespace cppgrad {

namespace impl {

    constexpr unsigned int CPPGRAD_CPU_MIN_PARTITION_1D = 256;
    constexpr unsigned int CPPGRAD_CPU_MIN_PARTITION_2D = 16;

    unsigned int get_blocks_N(unsigned int N)
    {
        return std::max(
            std::min((N + CPPGRAD_CPU_MIN_PARTITION_1D - 1u) / CPPGRAD_CPU_MIN_PARTITION_1D,
                std::thread::hardware_concurrency()),
            1u);
    }

}

}

#endif