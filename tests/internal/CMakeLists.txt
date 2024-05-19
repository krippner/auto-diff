add_executable(InternalTests
    testComputation.cpp
    testEvaluator.cpp
    testNode.cpp
    testShape.cpp
    testTopoView.cpp
    testTraits.cpp
    testTypeImpl.cpp
)
target_compile_features(InternalTests PRIVATE cxx_std_11)
target_link_libraries(InternalTests PRIVATE Catch2::Catch2WithMain AutoDiff::AutoDiff)  
catch_discover_tests(InternalTests TEST_PREFIX Internal)
