# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import re

is_readthedocs = (os.getenv('READTHEDOCS') == 'True')
launch_dir = os.getcwd()

if is_readthedocs:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

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

def get_version():
    if is_readthedocs:
        file_path = os.path.join(os.getcwd(), "../../Build/Inc/ProductBuildNumber.h")
    else:
        file_path = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), "../../Build/Inc/ProductBuildNumber.h")
    pattern = r"#define VERINFO_MAJOR_VERSION ([0-9]+)\n#define VERINFO_MINOR_VERSION ([0-9]+)\n#define VERINFO_MODIFICATION_NUMBER ([0-9]+)"
    with open(file_path, 'r', encoding='utf-8') as f:
        result = re.search(pattern, f.read())
        if result:
            return "{}.{}.{}".format(result.group(1), result.group(2), result.group(3))
        raise Exception("Failed to parse {}".format(file_path))
    return ""

setup(
    name='neoml',
    version=get_version(),
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

if is_readthedocs:
    os.chdir(launch_dir)
