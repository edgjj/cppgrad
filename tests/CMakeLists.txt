include (${PROJECT_SOURCE_DIR}/cmake/setup_gtest.cmake)
cppgrad_setup_gtest()

include (GoogleTest)

set (CPPGRAD_TESTS_SRC
	cppgrad_tests.cpp
)

add_executable(cppgrad_tests ${CPPGRAD_TESTS_SRC})
target_link_libraries(cppgrad_tests cppgrad gtest_main)

target_include_directories(cppgrad_tests 
	PRIVATE 
	${PROJECT_SOURCE_DIR}/include 
	${GOOGLETEST_PATH}
)

gtest_discover_tests(cppgrad_tests
	WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests
)