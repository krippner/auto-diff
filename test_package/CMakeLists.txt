cmake_minimum_required(VERSION 3.15)

project(PackageTest CXX)

find_package(AutoDiff CONFIG REQUIRED)

add_executable(test test_package.cpp)
target_link_libraries(test AutoDiff::AutoDiff)

