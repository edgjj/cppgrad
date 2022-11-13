include (${PROJECT_SOURCE_DIR}/cmake/setup_gtest.cmake)
cppgrad_setup_gtest()

include (GoogleTest)

set (CPPGRAD_TESTS_SRC
	cppgrad_tests.cpp
	nn_module_tests.hpp
)

function (add_cppgrad_test name source_path) 
    add_executable( ${name} ${source_path} )
	target_link_libraries(${name} PRIVATE cppgrad gtest_main)
    gtest_discover_tests(${name} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests)
endfunction()

file (GLOB TESTS_SOURCES *.cpp)

foreach(src_path ${TESTS_SOURCES} )
    get_filename_component(name ${src_path} NAME_WE)
    add_cppgrad_test(${name} ${src_path})
endforeach()