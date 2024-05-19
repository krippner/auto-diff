add_executable(EigenReductionsTests
    testMean.cpp
    testNorm.cpp
    testSquaredNorm.cpp
    testTotal.cpp
)
target_compile_features(EigenReductionsTests PRIVATE cxx_std_11)
target_link_libraries(EigenReductionsTests PRIVATE
    Catch2::Catch2WithMain
    Eigen3::Eigen
    AutoDiff::AutoDiff
)
catch_discover_tests(EigenReductionsTests TEST_PREFIX EigenReductions)
