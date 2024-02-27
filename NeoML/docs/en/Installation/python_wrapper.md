# Build the Python Wrapper

<!-- TOC -->

- [Before build](#before-build)
- [Windows](#windows)

<!-- /TOC -->

## Before build

Download the library sources from the repository:

``` sh
git clone https://github.com/neoml-lib/neoml <path_to_src>
```

Create a working directory next to the sources:

``` sh
mkdir PythonBuild
cd PythonBuild
```

The sample command lines below use [the standard CMake commands and options](https://cmake.org/cmake/help/latest/index.html) that specify build architecture, configuration, compiler to use, etc. Change the options as needed. 

## Windows
The installation procces is similar to the C++ instruction, except that you have to specify a path to the Python Wrapper, instead of NeoML C++ version. This is because Python Wrapper has it's own CMakeLists file stored in **NeoML/Python**.
You need Microsoft Visual Studio 2015 or later to create a project. Here is a sample command line that does it:

``` console 
cmake -G "Visual Studio 14 2015" -A <arch> <path_to_src>/NeoML/Python -DCMAKE_INSTALL_PREFIX=<install_path>
```

* \<arch> can be win32 or x64.
* \<install_path> is the **PythonBuild** directory created at the previous step.

Now you can build the project using Visual Studio or this command line:

``` console
cmake --build . --target install --config <cfg>
```

* \<cfg> can take the values: Debug, Release, RelWithDebInfo, MinSizeRel.

