find_package(Catch2 3 REQUIRED)
include(Catch) # catch_discover_tests

add_subdirectory(internal)

add_subdirectory(Core)

add_subdirectory(Basic)

option(AUTODIFF_BUILD_EIGEN_TESTS "Build tests requiring Eigen 3" TRUE)
if (AUTODIFF_BUILD_EIGEN_TESTS)
    add_subdirectory(Eigen)
endif ()

# add_subdirectory(Python)
