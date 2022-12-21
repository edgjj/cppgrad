function(cppgrad_setup_asyncpp)
	include(FetchContent)
 
	FetchContent_Declare(
        asyncpp
        GIT_REPOSITORY https://github.com/Amanieu/asyncplusplus.git
        GIT_TAG master
    )

    set (BUILD_SHARED_LIBS OFF CACHE BOOL "")
    FetchContent_MakeAvailable(asyncpp)

endfunction()