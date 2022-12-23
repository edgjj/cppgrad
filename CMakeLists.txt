cmake_minimum_required(VERSION 3.20)

project(cppgrad VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED)

if (MSVC)
	add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/MP;/fp:fast;>")
else()
	add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-ffast-math>")
endif()

string(
    COMPARE EQUAL
    "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}"
    IS_TOP_PROJECT
)

set(CPPGRAD_CUDA ON CACHE BOOL "Build with CUDA backend enabled")
set(CPPGRAD_MPI OFF CACHE BOOL "Build with distributed MPI compute capatibilities")

# to be used
set(CPPGRAD_INTERNAL_LIBS "") 

include (cmake/build_extras.cmake)

file (GLOB 
	CPPGRAD_HDR 
	include/cppgrad/*.hpp
	include/cppgrad/autograd/*.hpp
	include/cppgrad/device/*.hpp
	include/cppgrad/tensor/*.hpp
	include/cppgrad/tensor/util/*.hpp
)

set(CPPGRAD_DEVICE_INCLUDES 
	include/cppgrad/device/cpu/cpu.hpp
	include/cppgrad/device/cuda/cuda.hpp
)

list(APPEND CPPGRAD_HDR ${CPPGRAD_DEVICE_INCLUDES})

file (GLOB
	CPPGRAD_SRC
	src/*.cpp
	src/tensor/*.cpp
	src/autograd/*.cpp
)

add_library (cppgrad 
	STATIC 
	${CPPGRAD_SRC}
	${CPPGRAD_HDR}
)

target_include_directories(cppgrad PUBLIC include/)
target_link_libraries(cppgrad PRIVATE ${CPPGRAD_INTERNAL_LIBS})

set_target_properties(cppgrad
    PROPERTIES LINKER_LANGUAGE CXX
)


if (IS_TOP_PROJECT)
	if (BUILD_TESTS)
		include(CTest)
		enable_testing()
		add_subdirectory(tests)
	endif()

	if (BUILD_EXAMPLES)
		add_subdirectory(examples)
	endif()
endif()