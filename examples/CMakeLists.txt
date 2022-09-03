set (CPPGRAD_EXAMPLES_SRC
	example1.cpp
)

add_executable(cppgrad_examples ${CPPGRAD_EXAMPLES_SRC})
target_link_libraries(cppgrad_examples cppgrad)
target_include_directories(cppgrad_examples 
	PRIVATE 
	${PROJECT_SOURCE_DIR}/include
)