# CPU device lib
file (GLOB 
    CPU_SOURCES 
    src/device/cpu/*.cpp
)

file (GLOB
    CPU_HDR
    include/cppgrad/device/cpu/*.hpp
)

include (cmake/setup_asyncpp.cmake)
cppgrad_setup_asyncpp()

add_library(cppgrad_cpu_backend 
    STATIC
    ${CPU_SOURCES}
    ${CPU_HDR}
)

target_include_directories(cppgrad_cpu_backend PRIVATE include/)
target_link_libraries(cppgrad_cpu_backend PRIVATE Async++)

list(APPEND 
    CPPGRAD_INTERNAL_LIBS 
    cppgrad_cpu_backend)

if (CPPGRAD_CUDA)
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)

    add_compile_definitions(CPPGRAD_HAS_CUDA)

    file (GLOB 
        CUDA_SOURCES 
        src/device/cuda/*.cu
    )

    file (GLOB
        CUDA_HDR
        include/cppgrad/device/cuda/*.hpp
        include/cppgrad/device/cuda/kernels/*.cuh
    )

    add_library(cppgrad_cuda_backend 
        STATIC
        ${CUDA_SOURCES}
        ${CUDA_HDR}
    )

    target_include_directories(cppgrad_cuda_backend PRIVATE include/ ${CUDAToolkit_INCLUDE_DIRS})
	target_link_libraries(cppgrad_cuda_backend PRIVATE CUDA::toolkit)

    # little remark on cuda compute capatibilities
    # 
    # 61 - Pascal (Nvidia TITAN Xp, Titan X, GeForce GTX 1080 Ti, GTX 1080, GTX 1070, GTX 1060, GTX 1050 Ti, GTX 1050, GT 1030, MX150)
    # 70 - Volta (Nvidia TITAN V, Quadro GV100, Tesla V100, Tesla V100S)
    # 75 - Turing (NVIDIA TITAN RTX, GeForce RTX 20xx, GeForce GTX 16xx, Quadro RTX/T, Tesla T4)
    # 80 - Ampere (A100)
    # 86 - Ampere (GeForce RTX 3090, RTX 3080, RTX 3070, RTX 3060 Ti, RTX 3060, RTX 3050 Ti	RTX A6000, A40)
    set_property(TARGET cppgrad_cuda_backend PROPERTY CUDA_ARCHITECTURES 61 70 75 80 86) # and possibly 86

    list(APPEND 
        CPPGRAD_INTERNAL_LIBS 
        cppgrad_cuda_backend)
endif()


if (CPPGRAD_MPI)
    if(MSVC)
        set(MPI_MSMPI_LIB_PATH_ENV_NATIVE "$ENV{MSMPI_LIB64}")
        file(TO_CMAKE_PATH "${MPI_MSMPI_LIB_PATH_ENV_NATIVE}" MPI_MSMPI_LIB_PATH)

        # Microsoft MPI might use backslashes in the environment variables,
        # so it's important to convert to CMake-standard forward slashes
        # before appending a subdirectory with a forward slash.
        set(MPI_MSMPI_INC_PATH_ENV_NATIVE "$ENV{MSMPI_INC}")
        file(TO_CMAKE_PATH "${MPI_MSMPI_INC_PATH_ENV_NATIVE}" MPI_MSMPI_INC_PATH_ENV)
        set(MPI_MSMPI_INC_PATH_EXTRA "${MPI_MSMPI_INC_PATH_ENV}/x64")
    endif()
    
    find_package(MPI REQUIRED)

    add_compile_definitions(CPPGRAD_HAS_MPI)
    file (GLOB 
        DISTRIBUTED_SOURCES 
        src/distributed/*.cpp
        
    )

    file (GLOB
        DISTRIBUTED_HDR
        include/cppgrad/distributed/*.hpp
    )

    add_library(cppgrad_distrubuted
        STATIC
        ${DISTRIBUTED_SOURCES}
        ${DISTRIBUTED_HDR}
        include/cppgrad/exceptions/mpi_error.hpp
    )

    target_include_directories(cppgrad_distrubuted PRIVATE include/ ${MPI_INCLUDE_PATH})
	target_link_libraries(cppgrad_distrubuted PRIVATE ${MPI_LIBRARIES})

    list(APPEND 
        CPPGRAD_INTERNAL_LIBS 
        cppgrad_distrubuted)
    
endif()