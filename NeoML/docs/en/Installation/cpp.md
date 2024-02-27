# Build the C++ fully functional version

<!-- TOC -->

- [Before build](#before-build)
- [Windows](#windows)
- [Linux/macOS](#linux/macos)
- [Android](#android)
- [iOS](#ios)
- [Troubleshooting](#troubleshooting)

<!-- /TOC -->

## Before build

Download the library sources from the repository:

``` sh
git clone https://github.com/neoml-lib/neoml <path_to_src>
```

Create a working directory next to the sources:

``` sh
mkdir Build
cd Build
```

The sample command lines below use [the standard CMake commands and options](https://cmake.org/cmake/help/latest/index.html) that specify build architecture, configuration, compiler to use, etc. Change the options as needed. 

## Windows

You need Microsoft Visual Studio 2015 or later to create a project. Here is a sample command line that does it:

``` console 
cmake -G "Visual Studio 14 2015" -A <arch> <path_to_src>/NeoML -DCMAKE_INSTALL_PREFIX=<install_path>
```

* \<arch> can be win32 or x64.
* \<install_path> is the **Build** directory created at the previous step.

Now you can build the project using Visual Studio or this command line:

``` console
cmake --build . --target install --config <cfg>
```

* \<cfg> can take the values: Debug, Release, RelWithDebInfo, MinSizeRel.

You will get three dynamic link libraries as a result: **NeoML.dll**, **NeoMathEngine.dll**, and **NeoOnnx.dll**.

## Linux/macOS

Generate Ninja build rules:

``` console
cmake -G Ninja <path_to_src>/NeoML -DCMAKE_BUILD_TYPE=<cfg> -DCMAKE_INSTALL_PREFIX=<install_path>
```

* \<cfg> can take the values: Debug, Release, RelWithDebInfo, MinSizeRel.

Now you can build the project and install the results at the target path. Use this command line:

``` console
cmake --build . --target install
```
You will get three dynamic link libraries as a result: **libNeoML.so**, **libNeoMathEngine.so**, **libNeoOnnx.so** in case of Linux and **libNeoML.dylib**, **libNeoMathEngine.dylib**, **libNeoOnnx.dylib** in case of macOS.

## Android

You may build the Android version of the library on a host machine with Windows, Linux, or macOS. First install [Android NDK](https://developer.android.com/ndk/downloads).

Here is a sample command line for generating Ninja build rules:

``` console
cmake -G Ninja <path_to_src>/NeoML -DCMAKE_TOOLCHAIN_FILE=<path_to_ndk>/build/cmake/android.toolchain.cmake -DANDROID_ABI=<abi> -DCMAKE_INSTALL_PREFIX=<install_path> -DCMAKE_BUILD_TYPE=<cfg>
```

* \<abi> can be one of the following: armeabi-v7a, arm64-v8a, x86, x86_64.
* \<cfg> can take the values: Debug, Release, RelWithDebInfo, MinSizeRel.

Now you can build the project and install the results at the target path. Use this command line:

``` console
cmake --build . --target install
```

You will get three dynamic link libraries as a result: **libNeoML.so** , **libNeoMathEngine.so**, and **libNeoOnnx.so**.

## iOS

You will need Apple Xcode to create a project. A toolchain file that contains the build settings for arm64 and x86_64 architectures is provided in the library sources (see the cmake directory). Use this file to create a project:

``` console
cmake -G Xcode <path_to_src>/NeoML -DCMAKE_TOOLCHAIN_FILE=<path_to_src>/NeoML/cmake/ios.toolchain.cmake -DIOS_ARCH=<arch> -DCMAKE_INSTALL_PREFIX=<install_path>
```

* \<arch> can be arm64 or x86_64.

Now you can build the project using Xcode or this command line:

``` console
cmake --build . --target install --config <cfg>
```

* \<cfg> can take the values: Debug, Release, RelWithDebInfo, MinSizeRel.

You will get three frameworks as a result: **NeoML.framework**, **NeoMathEngine.framework**, and **NeoOnnx.framework**.

## Troubleshooting

**Protobuf**

On Windows, CMake sometimes can't see the path to the Protobuf library. To handle this, you can specify it yourself by adding `-DCMAKE_PREFIX_PATH=<path_to_Protobuf>` to the CMake command, that creates a project.

In this case, you will get the following CMake command:
``` console 
cmake -G "Visual Studio 14 2015" -A <arch> <path_to_src>/NeoML -DCMAKE_INSTALL_PREFIX=<install_path> -DCMAKE_PREFIX_PATH=<path_to_Protobuf>
``` 