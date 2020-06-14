set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)

set(CMAKE_SYSTEM_NAME IOS)
set(UNIX True)
set(APPLE True)
set(IOS True)

# Get the Xcode version being used.
execute_process(COMMAND xcodebuild -version
    OUTPUT_VARIABLE XCODE_VERSION
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REGEX MATCH "Xcode [0-9\\.]+" XCODE_VERSION "${XCODE_VERSION}")
string(REGEX REPLACE "Xcode ([0-9\\.]+)" "\\1" XCODE_VERSION "${XCODE_VERSION}")
message(STATUS "Building with Xcode version: ${XCODE_VERSION}")

# Determine the cmake host system version so we know where to find the iOS SDKs
find_program(CMAKE_UNAME uname /bin /usr/bin /usr/local/bin)
if(CMAKE_UNAME)
    exec_program(uname ARGS -r OUTPUT_VARIABLE CMAKE_HOST_SYSTEM_VERSION)
    string (REGEX REPLACE "^([0-9]+)\\.([0-9]+).*$" "\\1" DARWIN_MAJOR_VERSION "${CMAKE_HOST_SYSTEM_VERSION}")
endif(CMAKE_UNAME)

set(CMAKE_VISIBILITY_INLINES_HIDDEN FALSE)

# Skip the platform compiler checks for cross compiling
set(CMAKE_CXX_COMPILER_WORKS TRUE)
set(CMAKE_C_COMPILER_WORKS TRUE)

# All iOS/Darwin specific settings - some may be redundant
set(CMAKE_MACOSX_BUNDLE YES)
set(CMAKE_SHARED_LIBRARY_PREFIX "lib")
set(CMAKE_SHARED_LIBRARY_SUFFIX ".dylib")
set(CMAKE_SHARED_MODULE_PREFIX "lib")
set(CMAKE_SHARED_MODULE_SUFFIX ".so")
set(CMAKE_MODULE_EXISTS 1)
set(CMAKE_DL_LIBS "")

set(CMAKE_C_OSX_COMPATIBILITY_VERSION_FLAG "-compatibility_version ")
set(CMAKE_C_OSX_CURRENT_VERSION_FLAG "-current_version ")
set(CMAKE_CXX_OSX_COMPATIBILITY_VERSION_FLAG "${CMAKE_C_OSX_COMPATIBILITY_VERSION_FLAG}")
set(CMAKE_CXX_OSX_CURRENT_VERSION_FLAG "${CMAKE_C_OSX_CURRENT_VERSION_FLAG}")

set(CMAKE_C_FLAGS_INIT "-fmodules")
set(CMAKE_CXX_FLAGS_INIT "-Qunused-arguments -fcxx-modules")

set(CMAKE_C_LINK_FLAGS "-Wl,-search_paths_first ${CMAKE_C_LINK_FLAGS}")
set(CMAKE_CXX_LINK_FLAGS "-Wl,-search_paths_first ${CMAKE_CXX_LINK_FLAGS}")

set(CMAKE_PLATFORM_HAS_INSTALLNAME 1)
#set(CMAKE_SHARED_LINKER_FLAGS "-rpath @executable_path/Frameworks ${CMAKE_SHARED_LINKER_FLAGS}")
set(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "-dynamiclib -headerpad_max_install_names")
set(CMAKE_SHARED_MODULE_CREATE_C_FLAGS "-bundle -headerpad_max_install_names")
set(CMAKE_SHARED_MODULE_LOADER_C_FLAG "-Wl,-bundle_loader,")
set(CMAKE_SHARED_MODULE_LOADER_CXX_FLAG "-Wl,-bundle_loader,")
set(CMAKE_FIND_LIBRARY_SUFFIXES ".dylib" ".so" ".a")

# hack: if a new cmake (which uses CMAKE_INSTALL_NAME_TOOL) runs on an old build tree
# (where install_name_tool was hardcoded) and where CMAKE_INSTALL_NAME_TOOL isn't in the cache
# and still cmake didn't fail in CMakeFindBinUtils.cmake (because it isn't rerun)
# hardcode CMAKE_INSTALL_NAME_TOOL here to install_name_tool, so it behaves as it did before, Alex
if(NOT DEFINED CMAKE_INSTALL_NAME_TOOL)
    find_program(CMAKE_INSTALL_NAME_TOOL install_name_tool)
endif()

set(IOS_ARCH ${IOS_ARCH} CACHE STRING "Type of iOS Platform")

set(CMAKE_TRY_COMPILE_PLATFORM_VARIABLES IOS_ARCH)

# Check the platform selection and setup for developer root
if (IOS_ARCH STREQUAL "arm64")
    set(IOS_PLATFORM_LOCATION "iPhoneOS.platform")

    # This causes the installers to properly locate the output libraries
    set(CMAKE_XCODE_EFFECTIVE_PLATFORMS "-iphoneos")
elseif(IOS_ARCH STREQUAL "x86_64")
    set(SIMULATOR true)
    set(IOS_PLATFORM_LOCATION "iPhoneSimulator.platform")

    # This causes the installers to properly locate the output libraries
    set(CMAKE_XCODE_EFFECTIVE_PLATFORMS "-iphonesimulator")
else()
    message(FATAL_ERROR "Unsupported IOS_ARCH='${IOS_ARCH}' value selected. Please choose arm64 or x86_64")
endif()

# Setup iOS developer location unless specified manually with CMAKE_IOS_DEVELOPER_ROOT
set(XCODE_ROOT "/Applications/Xcode.app/Contents/Developer/Platforms/${IOS_PLATFORM_LOCATION}/Developer")
if(NOT DEFINED CMAKE_IOS_DEVELOPER_ROOT)
    if(EXISTS ${XCODE_ROOT})
        set(CMAKE_IOS_DEVELOPER_ROOT ${XCODE_ROOT})
    endif()
endif()
set (CMAKE_IOS_DEVELOPER_ROOT ${CMAKE_IOS_DEVELOPER_ROOT} CACHE PATH "Location of iOS Platform")

# Find and use the most recent iOS sdk unless specified manually with CMAKE_IOS_SDK_ROOT
if(NOT DEFINED CMAKE_IOS_SDK_ROOT)
    file(GLOB _CMAKE_IOS_SDKS "${CMAKE_IOS_DEVELOPER_ROOT}/SDKs/*")
    if(_CMAKE_IOS_SDKS) 
        list(SORT _CMAKE_IOS_SDKS)
        list(REVERSE _CMAKE_IOS_SDKS)
        list(GET _CMAKE_IOS_SDKS 0 CMAKE_IOS_SDK_ROOT)
    else(_CMAKE_IOS_SDKS)
        message(FATAL_ERROR "No iOS SDK's found in default search path ${CMAKE_IOS_DEVELOPER_ROOT}. Manually set CMAKE_IOS_SDK_ROOT or install the iOS SDK.")
    endif()
    message (STATUS "Toolchain using default iOS SDK: ${CMAKE_IOS_SDK_ROOT}")
endif(NOT DEFINED CMAKE_IOS_SDK_ROOT)
set(CMAKE_IOS_SDK_ROOT ${CMAKE_IOS_SDK_ROOT} CACHE PATH "Location of the selected iOS SDK")

# Set the sysroot default to the most recent SDK
set(CMAKE_OSX_SYSROOT ${CMAKE_IOS_SDK_ROOT} CACHE PATH "Sysroot used for iOS support")

set(CMAKE_OSX_ARCHITECTURES ${IOS_ARCH} CACHE STRING "Build architecture for iOS")

# Find the C & C++ compilers for the specified SDK.
if(NOT CMAKE_C_COMPILER)
    execute_process(COMMAND xcrun -sdk ${CMAKE_OSX_SYSROOT} -find clang
        OUTPUT_VARIABLE CMAKE_C_COMPILER
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "Using C compiler: ${CMAKE_C_COMPILER}")
endif()

if(NOT CMAKE_CXX_COMPILER)
    execute_process(COMMAND xcrun -sdk ${CMAKE_OSX_SYSROOT} -find clang++
        OUTPUT_VARIABLE CMAKE_CXX_COMPILER
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "Using CXX compiler: ${CMAKE_CXX_COMPILER}")
endif()

set(CMAKE_AR ar CACHE FILEPATH "" FORCE)

# Get the SDK version information.
execute_process(COMMAND xcodebuild -sdk ${CMAKE_OSX_SYSROOT} -version SDKVersion
    OUTPUT_VARIABLE SDK_VERSION
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(CMAKE_SYSTEM_VERSION ${SDK_VERSION})

# Set the find root to the iOS developer roots and to user defined paths
set(CMAKE_FIND_ROOT_PATH ${CMAKE_IOS_DEVELOPER_ROOT} ${CMAKE_IOS_SDK_ROOT} ${CMAKE_PREFIX_PATH} CACHE STRING "iOS find search path root")

# default to searching for frameworks first
set(CMAKE_FIND_FRAMEWORK FIRST)

# set up the default search directories for frameworks
set(CMAKE_SYSTEM_FRAMEWORK_PATH
    ${CMAKE_IOS_SDK_ROOT}/System/Library/Frameworks
    ${CMAKE_IOS_SDK_ROOT}/System/Library/PrivateFrameworks
    ${CMAKE_IOS_SDK_ROOT}/Developer/Library/Frameworks
)

# only search the iOS sdks, not the remainder of the host filesystem
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)


# This little macro lets you set any XCode specific property
macro(set_xcode_property TARGET XCODE_PROPERTY XCODE_VALUE XCODE_RELVERSION)
    set(XCODE_RELVERSION_I "${XCODE_RELVERSION}")
    if(XCODE_RELVERSION_I STREQUAL "All")
        set_property(TARGET ${TARGET} PROPERTY XCODE_ATTRIBUTE_${XCODE_PROPERTY} "${XCODE_VALUE}")
    else()
        set_property(TARGET ${TARGET} PROPERTY XCODE_ATTRIBUTE_${XCODE_PROPERTY}[variant=${XCODE_RELVERSION_I}] "${XCODE_VALUE}")
    endif()
endmacro()


# This macro lets you find executable programs on the host system
macro (find_host_package)
    set (CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
    set (CMAKE_FIND_ROOT_PATH_MODE_LIBRARY NEVER)
    set (CMAKE_FIND_ROOT_PATH_MODE_INCLUDE NEVER)
    set (IOS FALSE)

    find_package(${ARGN})

    set (IOS TRUE)
    set (CMAKE_FIND_ROOT_PATH_MODE_PROGRAM ONLY)
    set (CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
    set (CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
endmacro (find_host_package)

set(CMAKE_INSTALL_NAME_DIR "@rpath/")
set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR TRUE)
