add_executable(BasicTests
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
target_compile_features(BasicTests PRIVATE cxx_std_11)
target_link_libraries(BasicTests PRIVATE Catch2::Catch2WithMain AutoDiff::AutoDiff)  
catch_discover_tests(BasicTests TEST_PREFIX Basic)

add_executable(BasicModuleTest testModule.cpp)
target_compile_features(BasicModuleTest PRIVATE cxx_std_11)
target_link_libraries(BasicModuleTest PRIVATE Catch2::Catch2WithMain AutoDiff::AutoDiff)  
catch_discover_tests(BasicModuleTest)
