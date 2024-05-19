find_package(Eigen3 3.4 REQUIRED)

add_subdirectory(Array)
add_subdirectory(Basic)
add_subdirectory(CWise)
add_subdirectory(Products)
add_subdirectory(Reductions)

add_executable(EigenModuleTest testModule.cpp)
target_compile_features(EigenModuleTest PRIVATE cxx_std_11)
target_link_libraries(EigenModuleTest PRIVATE
    Catch2::Catch2WithMain
    Eigen3::Eigen
    AutoDiff::AutoDiff
)
catch_discover_tests(EigenModuleTest)

