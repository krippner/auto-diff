import os

from conan import ConanFile
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import rmdir


class AutoDiffRecipe(ConanFile):
    name = "autodiff"
    version = "0.4.0"
    package_type = "header-library"

    license = "MIT"
    author = "Matthias Krippner"
    url = "https://github.com/krippner/auto-diff"
    homepage = url
    description = ""
    topics = ""

    settings = "os", "arch", "compiler", "build_type"

    exports_sources = "CMakeLists.txt", "CMake/*", "include/*", "tests/*"
    no_copy_source = True

    def validate(self):
        check_min_cppstd(self, 17)

    def requirements(self):
        self.tool_requires("cmake/[>=3.15]")
        self.test_requires("catch2/[~3]")
        self.test_requires("eigen/[~3.4]")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        tc.cache_variables["AUTODIFF_BUILD_EXAMPLES"] = False
        tc.cache_variables["AUTODIFF_BUILD_BENCHMARKS"] = False
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
        cmake.test()

    def package(self):
        # copy(self, "include/*", self.source_folder, self.package_folder)
        cmake = CMake(self)
        cmake.install()
        rmdir(self, os.path.join(self.package_folder, "lib", "cmake"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "AutoDiff")
        self.cpp_info.set_property("cmake_target_name", "AutoDiff::AutoDiff")
        self.cpp_info.bindirs = []
        self.cpp_info.libdirs = []

    def package_id(self):
        self.info.clear()
