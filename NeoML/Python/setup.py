# -*- coding: utf-8 -*-

""" Copyright (c) 2017-2024 ABBYY

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
--------------------------------------------------------------------------------------------------------------
"""

import os
import sys
import subprocess
import re

is_readthedocs = (os.getenv('READTHEDOCS') == 'True')
launch_dir = os.getcwd()
this_directory = os.path.abspath(os.path.dirname(__file__))

if is_readthedocs:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    this_directory = os.getcwd()

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

        cfg = "RelWithDebInfo" if self.debug else "Release"

        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        cmake_args = [
            "-DCMAKE_INSTALL_PREFIX={}".format(extdir),
            "-DCMAKE_BUILD_TYPE={}".format(cfg),
            "-DPython3_FIND_VERSION={}.{}".format(sys.version_info.major, sys.version_info.minor)
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


# Get version from Build/Inc/ProductBuildNumber.h file
def get_version():
    file_path = os.path.join(this_directory, "../../Build/Inc/ProductBuildNumber.h")

    pattern = r"#define VERINFO_MAJOR_VERSION ([0-9]+)\n#define VERINFO_MINOR_VERSION ([0-9]+)\n#define VERINFO_MODIFICATION_NUMBER ([0-9]+)"
    with open(file_path, 'r', encoding='utf-8') as f:
        result = re.search(pattern, f.read())
        if result:
            return "{}.{}.{}".format(result.group(1), result.group(2), result.group(3))
        raise Exception("Failed to parse {}".format(file_path))


# Get the content of README.txt file
def get_long_description():
    with open(os.path.join(this_directory, 'README.txt'), encoding='utf-8') as f:
        return f.read()


setup(
    name='neoml',
    version=get_version(),
    description='NeoML python bindings',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='http://github.com/neoml-lib/neoml',
    install_requires=['numpy>=2.0.2', 'scipy>=1.5.2', 'onnx==1.16.0', 'protobuf==5.28.*'],
    ext_modules=[CMakeExtension("neoml.PythonWrapper")],
    cmdclass={"build_ext": CMakeBuild},
    include_package_data=True,
    packages=find_packages(),
    zip_safe=False,
    test_suite='tests'
)


if is_readthedocs:
    os.chdir(launch_dir)
