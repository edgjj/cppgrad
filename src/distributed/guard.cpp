#include "cppgrad/distributed/guard.hpp"
#include "cppgrad/distributed/error_string.hpp"
#include "cppgrad/exceptions/generic_error.hpp"
#include "cppgrad/exceptions/mpi_error.hpp"
#include <mpi.h>

namespace cppgrad::distributed {

// funny
bool Environment::_inited = false;

Environment::Environment(int argc, char* argv[])
{
    if (_inited) {
        throw exceptions::GenericError("MPI context has been already initialized.");
    }

    int is_finalized = 0;
    MPI_Finalized(&is_finalized);

    if (is_finalized != 0) {
        throw exceptions::GenericError("MPI context has been finalized, reinitialization is not possible.");
    }

    _inited = true;

    CPPGRAD_MPI_CHECK(MPI_Init, &argc, &argv);
}

Environment::~Environment()
{
    MPI_Finalize();
    _inited = false;
}

}