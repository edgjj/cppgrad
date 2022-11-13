cmake_minimum_required(VERSION 3.20)

project(cppgrad VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED)

string(
    COMPARE EQUAL
    "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}"
    IS_TOP_PROJECT
)

file (GLOB 
	CPPGRAD_HDR 
	include/cppgrad/*.hpp
	include/cppgrad/device/*.hpp)

file (GLOB
	CPPGRAD_SRC
	src/*.cpp
	src/device/*.cpp
)

add_library (cppgrad 
	STATIC 
	${CPPGRAD_SRC}
	${CPPGRAD_HDR}
)

target_include_directories(cppgrad PUBLIC include/)
target_link_libraries(cppgrad PRIVATE)

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