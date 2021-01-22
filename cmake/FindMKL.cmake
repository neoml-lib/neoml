# CMake simple script to detect Intel(R) Math Kernel Library (MKL)
# Note, MKLROOT environment variable is not set by installer, it should be set manually.
#
# Options: 
#   MKL_USE_STATIC_LIBS        Find static mkl libraries if this variable is true, otherwise try to find shared libraries.
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

if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(MKL_LIBRARY_DIR_SUFFIX "ia32")
    set(MKL_INTERFACE_TYPE "")
    if(WIN32)
        set(MKL_INTERFACE_TYPE "_c")
    endif()
else()
    set(MKL_LIBRARY_DIR_SUFFIX "intel64")
    set(MKL_INTERFACE_TYPE "_lp64")
endif()

if(WIN32)
    set(MKL_LIBRARY_PREFIX "")
    if(MKL_USE_STATIC_LIBS)
        set(MKL_LIBRARY_SUFFIX ".lib")
    else()
        set(MKL_LIBRARY_SUFFIX "_dll.lib")
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(MKL_LIBRARY_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
    if(MKL_USE_STATIC_LIBS)
        set(MKL_LIBRARY_SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX})
    else()
        set(MKL_LIBRARY_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})
    endif()
else()
    if(NOT MKL_FIND_QUIETLY)
        message(WARNING "MKL: OS '${CMAKE_SYSTEM_NAME}' is not supported.")
    endif()
endif()

if(USE_FINE_OBJECTS)

    # The only MKL we can use is the one located in $ENV{ROOT}/ThirdParty

    function(set_if_exist VAR_NAME VAR_VAL)
        if(EXISTS ${VAR_VAL})
            set(${VAR_NAME} ${VAR_VAL} PARENT_SCOPE)
        endif()
    endfunction()

    # additional helper variable (not exported by this script)
    set(MKL_LIB_DIR_Private $ENV{ROOT}/ThirdParty/MKL/${CMAKE_SYSTEM_NAME}/lib/${MKL_LIBRARY_DIR_SUFFIX}/)
    
    set_if_exist(MKL_INCLUDE_DIR $ENV{ROOT}/ThirdParty/MKL/${CMAKE_SYSTEM_NAME}/include)
    set_if_exist(MKL_CORE_LIB ${MKL_LIB_DIR_Private}/${MKL_LIBRARY_PREFIX}mkl_core${MKL_LIBRARY_SUFFIX})
    set_if_exist(MKL_SEQUENTIAL_LIB ${MKL_LIB_DIR_Private}/${MKL_LIBRARY_PREFIX}mkl_sequential${MKL_LIBRARY_SUFFIX})
    set_if_exist(MKL_INTEL_LIB ${MKL_LIB_DIR_Private}/${MKL_LIBRARY_PREFIX}mkl_intel${MKL_INTERFACE_TYPE}${MKL_LIBRARY_SUFFIX})

else() # NOT USE_FINE_OBJECTS

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
            mkl
            include
            IntelSWTools/compilers_and_libraries/windows/mkl/include
    )

    find_library(MKL_CORE_LIB
        NAMES ${MKL_LIBRARY_PREFIX}mkl_core${MKL_LIBRARY_SUFFIX}
        PATHS
            $ENV{MKLROOT}/lib
            /opt/intel/lib
            /opt/intel/mkl/lib
        PATH_SUFFIXES 
            ${MKL_LIBRARY_DIR_SUFFIX}
            IntelSWTools/compilers_and_libraries/windows/mkl/lib/${MKL_LIBRARY_DIR_SUFFIX}
    )

    find_library(MKL_SEQUENTIAL_LIB
        NAMES ${MKL_LIBRARY_PREFIX}mkl_sequential${MKL_LIBRARY_SUFFIX}
        PATHS
            $ENV{MKLROOT}/lib
            /opt/intel/lib
            /opt/intel/mkl/lib
        PATH_SUFFIXES 
            ${MKL_LIBRARY_DIR_SUFFIX}
            IntelSWTools/compilers_and_libraries/windows/mkl/lib/${MKL_LIBRARY_DIR_SUFFIX}
    )

    find_library(MKL_INTEL_LIB
        NAMES ${MKL_LIBRARY_PREFIX}mkl_intel${MKL_INTERFACE_TYPE}${MKL_LIBRARY_SUFFIX}
        PATHS
            $ENV{MKLROOT}/lib
            /opt/intel/lib
            /opt/intel/mkl/lib
        PATH_SUFFIXES 
            ${MKL_LIBRARY_DIR_SUFFIX}
            IntelSWTools/compilers_and_libraries/windows/mkl/lib/${MKL_LIBRARY_DIR_SUFFIX}
    )

endif() # USE_FINE_OBJECTS

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKL DEFAULT_MSG MKL_INCLUDE_DIR MKL_CORE_LIB MKL_SEQUENTIAL_LIB MKL_INTEL_LIB)

mark_as_advanced(MKL_INCLUDE_DIR MKL_CORE_LIB MKL_SEQUENTIAL_LIB MKL_INTEL_LIB)

if(MKL_FOUND)
    add_library(MKL::Libs IMPORTED INTERFACE)
    target_include_directories(MKL::Libs INTERFACE ${MKL_INCLUDE_DIR})
    if(UNIX)
        set(MKL_LIBS ${MKL_INTEL_LIB} ${MKL_SEQUENTIAL_LIB} ${MKL_CORE_LIB})
        if(NOT APPLE)
            set(THREADS_PREFER_PTHREAD_FLAG ON)
            find_package(Threads REQUIRED)
            if(MKL_USE_STATIC_LIBS)
                set(MKL_LIBS -Wl,--start-group ${MKL_LIBS} -Wl,--end-group)
            endif()
            set(MKL_LIBS ${MKL_LIBS} Threads::Threads)
        endif()
        target_link_libraries(MKL::Libs INTERFACE ${MKL_LIBS} ${CMAKE_DL_LIBS})
    else()
        target_link_libraries(MKL::Libs INTERFACE ${MKL_INTEL_LIB} ${MKL_SEQUENTIAL_LIB} ${MKL_CORE_LIB})
    endif()
endif()
