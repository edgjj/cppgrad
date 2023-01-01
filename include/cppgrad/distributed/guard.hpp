// Copyright (c) 2023 Yegor Suslin
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef CPPGRAD_DISTRIBUTED_GUARD_HPP
#define CPPGRAD_DISTRIBUTED_GUARD_HPP

#ifdef CPPGRAD_HAS_MPI

namespace cppgrad::distributed {

struct Environment {
    Environment(int argc, char* argv[]);
    ~Environment();

private:
    static bool _inited;
};

}

#endif

#endif