find_package(Catch2 3 REQUIRED)
find_package(Eigen3 3.4 REQUIRED)

add_executable(BasicBenchmarks basic-sigmoid.cpp)
target_compile_features(BasicBenchmarks PRIVATE cxx_std_11)
target_link_libraries(BasicBenchmarks PRIVATE Catch2::Catch2WithMain AutoDiff::AutoDiff)

add_executable(EigenBenchmarks eigen-sigmoid.cpp)
target_compile_features(EigenBenchmarks PRIVATE cxx_std_11)
target_link_libraries(EigenBenchmarks PRIVATE
    Catch2::Catch2WithMain
    AutoDiff::AutoDiff
    Eigen3::Eigen
)
