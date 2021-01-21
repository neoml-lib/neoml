# Build the inference version

<!-- TOC -->

- [Before build](#before-build)
- [Java](#java)
- [Objective-C](#objective-c)

<!-- /TOC -->

## Before build

Download the library sources from the repository:

``` sh
git clone https://github.com/neoml-lib/neoml <path_to_src>
```

### Java

The <path_to_src>/NeoML/Java directory contains a project for building the Android library with Android Studio. You can also use gradle console. Switch to the project directory:

``` console
cd <path_to_src>/NeoML/Java
```

Make sure you have the JAVA_HOME environment variable pointing to the directory where Java is installed and the ANDROID_HOME environment variable pointing to Android SDK.

If you are on Windows, run the following command to build the project:

``` console
gradlew.bat "inference:assembleRelease"
```

If you are on macOS/Linux:

``` console
./gradlew "inference:assembleRelease"
```

Now the <path_to_src>/NeoML/Java/inference/build/outputs/aar directory contains the **NeoInference.aar** file.

### Objective-C

The Objective-C library is built using [the standard CMake commands and options](https://cmake.org/cmake/help/latest/index.html).

Create a working directory:

``` sh
mkdir Build
cd Build
```

You will need Apple Xcode to create a project. A toolchain file that contains the build settings for arm64 and x86_64 architectures is provided in the library sources (see the cmake directory). Use this file to create a project:

``` console
cmake -G Xcode <path_to_src>/NeoML/objc -DCMAKE_TOOLCHAIN_FILE=<path_to_src>/cmake/ios.toolchain.cmake -DIOS_ARCH=<arch> -DCMAKE_INSTALL_PREFIX=<install_path>
```

* \<arch> can be arm64 or x86_64.

Now you can build the project using Xcode or this command line:

``` console
cmake --build . --target install --config <cfg>
```

* \<cfg> can take the values: Debug, Release, RelWithDebInfo, MinSizeRel.

The build result is the **NeoInference.framework**.