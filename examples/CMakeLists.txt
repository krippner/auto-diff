find_package(Eigen3 3.4 REQUIRED)

add_executable(BasicExamples basic.cpp)
target_link_libraries(BasicExamples PRIVATE AutoDiff::AutoDiff)

add_executable(EigenExamples eigen.cpp)
target_link_libraries(EigenExamples PRIVATE AutoDiff::AutoDiff Eigen3::Eigen)

add_executable(Backpropagation backpropagation.cpp)
target_link_libraries(Backpropagation PRIVATE AutoDiff::AutoDiff Eigen3::Eigen)
