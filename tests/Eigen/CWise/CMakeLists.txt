add_executable(EigenCWiseTests
    testCos.cpp
    testDifference.cpp
    testExp.cpp
    testLog.cpp
    testMax.cpp
    testMin.cpp
    testNegation.cpp
    testPow.cpp
    testProduct.cpp
    testQuotient.cpp
    testSin.cpp
    testSqrt.cpp
    testSquare.cpp
    testSum.cpp
)
target_compile_features(EigenCWiseTests PRIVATE cxx_std_11)
target_link_libraries(EigenCWiseTests PRIVATE
    Catch2::Catch2WithMain
    Eigen3::Eigen
    AutoDiff::AutoDiff
)  
catch_discover_tests(EigenCWiseTests TEST_PREFIX EigenCWise)
