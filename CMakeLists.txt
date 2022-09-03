cmake_minimum_required(VERSION 3.20)

project(cppgrad VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED)

file (GLOB CPPGRAD_HDR include/cppgrad/*.hpp)

set (CPPGRAD_SRC
	src/stub.cpp)

add_library (cppgrad 
	STATIC 
	${CPPGRAD_SRC}
	${CPPGRAD_HDR}
)

target_include_directories(cppgrad PRIVATE include/)
target_link_libraries(cppgrad PRIVATE)

set_target_properties(cppgrad
    PROPERTIES LINKER_LANGUAGE CXX
)

if (BUILD_TESTS)
	include(CTest)
	enable_testing()
	add_subdirectory(tests)
endif()

if (BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()
