# CMake simple script to detect Intel(R) Math Kernel Library (MKL)
# Note, MKLROOT environment variable is not set by installer, it should be set manually.
#
# Options: 
#   MKL_USE_STATIC_LIBS        Try to find static mkl libraries
# 
# The module provides imported interface target: MKL::Libs
# Variables are defined by module:
#   MKL_FOUND                  True/False
#   MKL_INCLUDE_DIR            MKL include folder
#   MKL_CORE_LIB
#   MKL_SEQUENTIAL_LIB
#   MKL_INTEL_LIB
#
# Usage:
#
#  find_package(MKL)
#  add_executable(app main.cpp)
#  target_link_libraries(app PRIVATE MKL::Libs)

find_path(MKL_INCLUDE_DIR
    NAMES
        mkl.h
        mkl_blas.h
        mkl_cblas.h
    PATHS
        $ENV{MKLROOT}
        /opt/intel
        /opt/intel/mkl
    PATH_SUFFIXES
        include
        IntelSWTools/compilers_and_libraries/windows/mkl/include
)

if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(MKL_LIBRARY_DIR_SUFFIX "ia32")
    set(MKL_LIB_SUFFIX "c")
else()
    set(MKL_LIBRARY_DIR_SUFFIX "intel64")
    set(MKL_LIB_SUFFIX "lp64")
endif()

if(MKL_USE_STATIC_LIBS)
    set(_MKL_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

find_library(MKL_CORE_LIB
    NAMES mkl_core
    PATHS
        $ENV{MKLROOT}/lib
        /opt/intel/lib
        /opt/intel/mkl/lib
    PATH_SUFFIXES 
        ${MKL_LIBRARY_DIR_SUFFIX}
        IntelSWTools/compilers_and_libraries/windows/mkl/lib/${MKL_LIBRARY_DIR_SUFFIX}
)

find_library(MKL_SEQUENTIAL_LIB
    NAMES mkl_sequential
    PATHS
        $ENV{MKLROOT}/lib
        /opt/intel/lib
        /opt/intel/mkl/lib
    PATH_SUFFIXES 
        ${MKL_LIBRARY_DIR_SUFFIX}
        IntelSWTools/compilers_and_libraries/windows/mkl/lib/${MKL_LIBRARY_DIR_SUFFIX}
)

find_library(MKL_INTEL_LIB
    NAMES mkl_intel_${MKL_LIB_SUFFIX}
    PATHS
        $ENV{MKLROOT}/lib
        /opt/intel/lib
        /opt/intel/mkl/lib
    PATH_SUFFIXES 
        ${MKL_LIBRARY_DIR_SUFFIX}
        IntelSWTools/compilers_and_libraries/windows/mkl/lib/${MKL_LIBRARY_DIR_SUFFIX}
)

if(MKL_USE_STATIC_LIBS)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${_MKL_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

set(MKL_FOUND TRUE)
if(NOT MKL_INCLUDE_DIR)
    set(MKL_FOUND FALSE)
    message(STATUS "MKL_INCLUDE_DIR was not found!")
endif()

if(NOT MKL_CORE_LIB)
    set(MKL_FOUND FALSE)
    message(STATUS "MKL_CORE_LIB was not found!")
endif()

if(NOT MKL_SEQUENTIAL_LIB)
    set(MKL_FOUND FALSE)
    message(STATUS "MKL_SEQUENTIAL_LIB was not found!")
endif()

if(NOT MKL_INTEL_LIB)
    set(MKL_FOUND FALSE)
    message(STATUS "MKL_INTEL_LIB was not found!")
endif()

if(MKL_FOUND)
    add_library(MKL::Libs IMPORTED INTERFACE)
    target_include_directories(MKL::Libs INTERFACE ${MKL_INCLUDE_DIR})
    if(UNIX)
        set(MKL_LIBS ${MKL_INTEL_LIB} ${MKL_SEQUENTIAL_LIB} ${MKL_CORE_LIB})
        if(NOT APPLE)
            set(THREADS_PREFER_PTHREAD_FLAG ON)
            find_package(Threads REQUIRED)
            set(MKL_LIBS -Wl,--start-group ${MKL_LIBS} -Wl,--end-group Threads::Threads)
        endif()
        target_link_libraries(MKL::Libs INTERFACE ${MKL_LIBS} ${CMAKE_DL_LIBS})
    else()
        target_link_libraries(MKL::Libs INTERFACE ${MKL_INTEL_LIB} ${MKL_SEQUENTIAL_LIB} ${MKL_CORE_LIB})
    endif()
    
    if(NOT MKL_FIND_QUIETLY)
        message(STATUS "Found Intel(R) MKL: TRUE")
    endif()
else()
    if(MKL_FIND_REQUIRED)
        message(SEND_ERROR "Found Intel(R) MKL could: FALSE")
    else()
        message(STATUS "Found Intel(R) MKL: FALSE")
    endif()
endif()