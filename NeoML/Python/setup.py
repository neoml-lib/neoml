# -*- coding: utf-8 -*-
import os
import sys
import subprocess

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

# Convert distutils Windows platform specifiers to CMake -A arguments
WIN_ARCH = {
    "win32": "Win32",
    "win-amd64": "x64"
}

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = "Debug" if self.debug else "Release"

        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        cmake_args = [
            "-DCMAKE_INSTALL_PREFIX={}".format(extdir),
            "-DCMAKE_BUILD_TYPE={}".format(cfg)
        ]
        build_args = []

        if self.compiler.compiler_type != "msvc":
            if not cmake_generator:
                cmake_args += ["-GNinja"]

        else:
            cmake_args += ["-A", WIN_ARCH[self.plat_name]]
            build_args += ["--config", cfg]

        if hasattr(self, "parallel") and self.parallel:
            build_args += ["--parallel {}".format(self.parallel)]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "install"] + build_args, cwd=self.build_temp
        )


setup(
    name='neoml',
    version='1.0.1',
    description='NeoML python bindings',
    url='http://github.com/neoml-lib/neoml',
    install_requires=['numpy>=1.19.1', 'scipy>=1.5.2'],
    ext_modules=[CMakeExtension("neoml.PythonWrapper")],
    cmdclass={"build_ext": CMakeBuild},
    include_package_data=True,
    packages=find_packages(),
    zip_safe=False,
    test_suite='tests'
)
