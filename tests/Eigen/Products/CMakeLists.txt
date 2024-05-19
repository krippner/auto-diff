add_executable(EigenProductsTests
    testDotProduct.cpp
    testMatrixProduct.cpp
    testMatrixVectorProduct.cpp
    testTensorProduct.cpp
)
target_compile_features(EigenProductsTests PRIVATE cxx_std_11)
target_link_libraries(EigenProductsTests PRIVATE
    Catch2::Catch2WithMain
    Eigen3::Eigen
    AutoDiff::AutoDiff
)
catch_discover_tests(EigenProductsTests TEST_PREFIX EigenProducts)
