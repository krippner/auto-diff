add_executable(CoreTests
    testFunction.cpp
    testFunctionNoFwd.cpp
    testFunctionNoRev.cpp
    testVariable.cpp
)
target_compile_features(CoreTests PRIVATE cxx_std_11)
target_link_libraries(CoreTests PRIVATE Catch2::Catch2WithMain AutoDiff::AutoDiff)  
catch_discover_tests(CoreTests TEST_PREFIX Core)
