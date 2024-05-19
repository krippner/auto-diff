cmake_minimum_required(VERSION 3.15)

project(AutoDiff
    VERSION 0.4.0
    LANGUAGES CXX
    DESCRIPTION ""
)

# PROJECT_IS_TOP_LEVEL is introduced in v3.21
if (CMAKE_VERSION VERSION_LESS 3.21)
    if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        set(PROJECT_IS_TOP_LEVEL TRUE)
    else ()
        set(PROJECT_IS_TOP_LEVEL FALSE)
    endif ()
endif ()

add_library(AutoDiff INTERFACE)
add_library(AutoDiff::AutoDiff ALIAS AutoDiff)
target_include_directories(
    AutoDiff INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    $<INSTALL_INTERFACE:include>
)
target_compile_features(AutoDiff INTERFACE cxx_std_17)

if (${PROJECT_IS_TOP_LEVEL})

    include(CTest)
    if (BUILD_TESTING)
        add_subdirectory(tests)
    endif ()

    option(AUTODIFF_BUILD_EXAMPLES "Build example projects" OFF)
    if (AUTODIFF_BUILD_EXAMPLES)
        add_subdirectory(examples)
    endif ()

    option(AUTODIFF_BUILD_BENCHMARKS "Build benchmarks" OFF)
    if (AUTODIFF_BUILD_BENCHMARKS)
        add_subdirectory(benchmarks)
    endif ()

    include(GNUInstallDirs)

    set(AUTODIFF_CMAKE_CONFIG_DESTINATION lib/cmake/AutoDiff)

    include(CMakePackageConfigHelpers)
    configure_package_config_file(
        "${CMAKE_CURRENT_LIST_DIR}/CMake/AutoDiffConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/AutoDiffConfig.cmake"
        INSTALL_DESTINATION "${AUTODIFF_CMAKE_CONFIG_DESTINATION}"
    )
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/AutoDiffConfigVersion.cmake"
        VERSION       ${CMAKE_PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    install(
        FILES
            "${CMAKE_CURRENT_BINARY_DIR}/AutoDiffConfig.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/AutoDiffConfigVersion.cmake"
        DESTINATION
            "${AUTODIFF_CMAKE_CONFIG_DESTINATION}"
    )

    install(
        TARGETS AutoDiff EXPORT AutoDiffTargets
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    )
    export(EXPORT AutoDiffTargets
        FILE
            "${CMAKE_CURRENT_BINARY_DIR}/AutoDiffTargets.cmake"
        NAMESPACE
            AutoDiff::
    )
    install(EXPORT AutoDiffTargets
        FILE
            AutoDiffTargets.cmake
        NAMESPACE
            AutoDiff::
        DESTINATION
            "${AUTODIFF_CMAKE_CONFIG_DESTINATION}"
    )

    install(DIRECTORY include/AutoDiff
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    )

endif ()
