# Copyright (c) 2023 Yegor Suslin

# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

set (CPPGRAD_EXAMPLES_SRC
	example1.cpp
)

function (add_cppgrad_example name source_path) 
    add_executable( ${name} ${source_path})
	target_link_libraries(${name} PRIVATE cppgrad)
endfunction()

file (GLOB EXAMPLE_SOURCES *.cpp)

foreach(src_path ${EXAMPLE_SOURCES} )
    get_filename_component(name ${src_path} NAME_WE)
    add_cppgrad_example("${name}" ${src_path})
endforeach()
