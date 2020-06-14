if(NOT DEFINED ENV{ROOT})
    message(FATAL_ERROR "Environment variable ROOT is not set!")
endif()

file(TO_CMAKE_PATH "$ENV{ROOT}" FINE_ROOT)

# Cmake build type to fine build type
if(MSVC OR XCODE)
    set(FINE_BUILD_TYPE $<$<OR:$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:Release>$<$<CONFIG:Debug>:Debug>$<$<CONFIG:Release>:Final>)
else()
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        set(FINE_BUILD_TYPE Final)
    elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
        set(FINE_BUILD_TYPE Release)
    else()
        set(FINE_BUILD_TYPE Debug)
    endif()
endif()

if(WIN32 AND MSVC)
    set(FINE_LIBRARIES_DIR_DEFAULT ${FINE_ROOT}/lib)
elseif(IOS AND XCODE)
    set(FINE_LIBRARIES_DIR_DEFAULT ${FINE_ROOT}/X.IOS.${FINE_BUILD_TYPE}/${IOS_ARCH})
else(LINUX OR DARWIN OR ANDROID)
    set(FINE_LIBRARIES_DIR_DEFAULT ${FINE_ROOT}/X.${CMAKE_SYSTEM_NAME}.${FINE_BUILD_TYPE}/)
    if(LINUX)
        if(CMAKE_SIZEOF_VOID_P STREQUAL "4")
            string(APPEND FINE_LIBRARIES_DIR_DEFAULT "x86")
        else()
            string(APPEND FINE_LIBRARIES_DIR_DEFAULT "x86_64")
        endif()
    elseif(DARWIN)
        string(APPEND FINE_LIBRARIES_DIR_DEFAULT "x86_64")
    elseif(ANDROID)
        string(APPEND FINE_LIBRARIES_DIR_DEFAULT obj/local/${ANDROID_ABI})
    endif()
endif()

set(FINE_LIBRARIES_DIR "${FINE_LIBRARIES_DIR_DEFAULT}" CACHE PATH "")

set(FineObj_FOUND FALSE)

if(IOS OR ANDROID)
    set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM BOTH)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
endif()

# Message Compiler
find_program(FineObj_MESSAGE_COMPILER
    NAMES MsgComCon
    PATHS ${FINE_ROOT}/FineObjects/Utils/MsgCompiler/Bin/${CMAKE_HOST_SYSTEM_NAME}
)

if(NOT WIN32 AND TARGET FineObj AND TARGET FineObjStaticPart AND TARGET PortLayer)
    add_library(FineObjects IMPORTED INTERFACE)

    if(LINUX OR ANDROID)
        target_link_libraries(FineObjects INTERFACE -Wl,--whole-archive FineObjStaticPart -Wl,--no-whole-archive)
    elseif(DARWIN OR IOS)
        target_link_libraries(FineObjects INTERFACE -force_load FineObjStaticPart)
    endif()

    target_link_libraries(FineObjects INTERFACE FineObj)
    
    if(LINUX OR DARWIN)
        target_link_libraries(FineObjects INTERFACE PortLayer dl)
    endif()
    target_compile_definitions(FineObjects INTERFACE NEOML_USE_FINEOBJ)
    target_include_directories(FineObjects INTERFACE ${FINE_ROOT}/FineObjects/Inc)
    set(FineObj_FOUND TRUE)
else()
    # - Try to find FineObjects.
    # Include dir
    find_path(FineObj_INCLUDE_DIR
        NAMES FineObj.h
        PATHS ${FINE_ROOT}/FineObjects/Inc
    )

    set(CONFIGS Final Release Debug)

    if(MSVC)
         if(CMAKE_SIZEOF_VOID_P EQUAL 4)
            set(WIN_ARCH_SUFFIX Win32)
        else()
            set(WIN_ARCH_SUFFIX x64)
        endif()
        
        foreach(CONFIG ${CONFIGS})
            set(NAME_SUFFIX ${WIN_ARCH_SUFFIX}.${CONFIG})

            # Find FineObj
            find_library(FineObj_LIBRARY_${CONFIG}
                NAMES FineObj.${NAME_SUFFIX}
                PATHS ${FINE_LIBRARIES_DIR} ${FINE_ROOT}/FineObjects/lib
                NO_DEFAULT_PATH
            )
            
            # Find FineObjStaticPart
            find_library(FineObjStaticPart_LIBRARY_${CONFIG}
                NAMES FineObjStaticPart.${NAME_SUFFIX}
                PATHS ${FINE_LIBRARIES_DIR} ${FINE_ROOT}/FineObjects/lib
                NO_DEFAULT_PATH
            )
            
            if(FineObj_INCLUDE_DIR AND FineObj_LIBRARY_${CONFIG} AND FineObjStaticPart_LIBRARY_${CONFIG} AND FineObj_MESSAGE_COMPILER)
                message(STATUS "Found FineObjects libraries for build type: ${CONFIG}")
                set(FineObj_FOUND TRUE)
            endif()
        endforeach()

        if(FineObj_FOUND)
            # Import target
            add_library(FineObjects IMPORTED INTERFACE)
            target_include_directories(FineObjects INTERFACE ${FineObj_INCLUDE_DIR})
            target_link_libraries(FineObjects INTERFACE
                $<$<OR:$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:${FineObj_LIBRARY_Release} ${FineObjStaticPart_LIBRARY_Release}>
                $<$<CONFIG:Debug>:${FineObj_LIBRARY_Debug} ${FineObjStaticPart_LIBRARY_Debug}>
                $<$<CONFIG:Release>:${FineObj_LIBRARY_Final} ${FineObjStaticPart_LIBRARY_Final}>
            )
            target_compile_definitions(FineObjects INTERFACE NEOML_USE_FINEOBJ _UNICODE UNICODE)
        endif()
    elseif(XCODE AND IOS)
        foreach(CONFIG ${CONFIGS})
            set(LIBRARIES_DIR ${FINE_ROOT}/X.${CMAKE_SYSTEM_NAME}.${CONFIG}/${IOS_ARCH})
            
            # Find FineObj
            find_library(FineObj_LIBRARY_${CONFIG}
                NAMES FineObj
                PATHS ${LIBRARIES_DIR}
                NO_DEFAULT_PATH
            )
            
            # Find FineObjStaticPart
            find_library(FineObjStaticPart_LIBRARY_${CONFIG}
                NAMES FineObjStaticPart
                PATHS ${LIBRARIES_DIR}
                NO_DEFAULT_PATH
            )

            if(FineObj_INCLUDE_DIR AND FineObj_LIBRARY_${CONFIG} AND FineObjStaticPart_LIBRARY_${CONFIG} AND FineObj_MESSAGE_COMPILER)
                message(STATUS "Found FineObjects libraries in dir: ${LIBRARIES_DIR}")
                set(FineObj_FOUND TRUE)
            endif()
        endforeach()
        
        if(FineObj_FOUND)
            # Import target
            add_library(FineObjects IMPORTED INTERFACE)
            
            target_include_directories(FineObjects INTERFACE ${FineObj_INCLUDE_DIR}
                ${FineObj_INCLUDE_DIR}/../PortLayer/Inc
                ${FineObj_INCLUDE_DIR}/../PortLayer/${CMAKE_SYSTEM_NAME}/Inc
            )
            
            target_link_libraries(FineObjects INTERFACE
                $<$<OR:$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:${FineObj_LIBRARY_Release} -force_load ${FineObjStaticPart_LIBRARY_Release}>
                $<$<CONFIG:Debug>:${FineObj_LIBRARY_Debug} -force_load ${FineObjStaticPart_LIBRARY_Debug}>
                $<$<CONFIG:Release>:${FineObj_LIBRARY_Final} -force_load ${FineObjStaticPart_LIBRARY_Final}>
            )
            
            target_compile_definitions(FineObjects INTERFACE NEOML_USE_FINEOBJ _DLL _UNICODE UNICODE _NATIVE_WCHAR_T_DEFINED _MT __LITTLE_ENDIAN__ _PLATFORM_64_BIT)
        endif()
    else()

        # Find FineObj
        find_library(FineObj_LIBRARY
            NAMES FineObj
            PATHS ${FINE_LIBRARIES_DIR} 
        )

        # Find FineObjStaticPart
        find_library(FineObjStaticPart_LIBRARY
            NAMES FineObjStaticPart
            PATHS ${FINE_LIBRARIES_DIR}
        )
        
        # Find PortLayer
        find_library(PortLayer_LIBRARY
            NAMES PortLayer
            PATHS ${FINE_LIBRARIES_DIR}
            NO_DEFAULT_PATH
        )

        if(FineObj_INCLUDE_DIR AND FineObj_LIBRARY AND FineObjStaticPart_LIBRARY AND PortLayer_LIBRARY AND FineObj_MESSAGE_COMPILER)
            set(FineObj_FOUND TRUE)

            # Import target
            add_library(FineObjects IMPORTED INTERFACE)
            
            target_include_directories(FineObjects INTERFACE ${FineObj_INCLUDE_DIR}
                ${FineObj_INCLUDE_DIR}/../PortLayer/Inc
                ${FineObj_INCLUDE_DIR}/../PortLayer/${CMAKE_SYSTEM_NAME}/Inc
            )

            if(LINUX OR ANDROID)
                target_link_libraries(FineObjects INTERFACE -Wl,--whole-archive ${FineObjStaticPart_LIBRARY} -Wl,--no-whole-archive)
            elseif(DARWIN)
                target_link_libraries(FineObjects INTERFACE -force_load ${FineObjStaticPart_LIBRARY})
            endif()

            target_link_libraries(FineObjects INTERFACE ${FineObj_LIBRARY})
            if(LINUX OR DARWIN)
                target_link_libraries(FineObjects INTERFACE ${PortLayer_LIBRARY} dl)
            endif()

            target_compile_definitions(FineObjects INTERFACE NEOML_USE_FINEOBJ _DLL _UNICODE UNICODE _NATIVE_WCHAR_T_DEFINED _MT __LITTLE_ENDIAN__)
            if(NOT CMAKE_SIZEOF_VOID_P STREQUAL "4")
                target_compile_definitions(FineObjects INTERFACE _PLATFORM_64_BIT)
            endif()

            if(ANDROID)
                # Android NDK version. See FineObjStaticPart\Platform\llvm\ReadMe.md
                include(${CMAKE_TOOLCHAIN_FILE})
                if(NOT ${ANDROID_NDK_MAJOR} STREQUAL "18" OR
                   NOT ${ANDROID_NDK_MINOR} STREQUAL "1" OR
                   NOT ${ANDROID_NDK_BUILD} STREQUAL "5063045" OR
                   NOT ${ANDROID_NDK_BETA} STREQUAL "0")
                    message(FATAL_ERROR "Incorrect NDK Version. Correct is r18b")
                endif()
                target_compile_definitions(FineObjects INTERFACE -DANDROID_NDK_MAJOR=${ANDROID_NDK_MAJOR} ANDROID_NDK_MINOR=${ANDROID_NDK_MINOR} ANDROID_NDK_BUILD=${ANDROID_NDK_BUILD} ANDROID_NDK_BETA=${ANDROID_NDK_BETA})
            endif()
        else()
            set(FineObj_FOUND FALSE)
        endif()

    endif()
endif()

if(NOT FineObj_FOUND)
    unset(FINE_LIBRARIES_DIR_DEFAULT)
    unset(FINE_LIBRARIES_DIR CACHE)
else()
    get_filename_component(FINE_MESSAGE_COMPILER_DIR ${FineObj_MESSAGE_COMPILER} DIRECTORY)
    get_filename_component(FINE_MESSAGE_COMPILER ${FineObj_MESSAGE_COMPILER} NAME)
    if(WIN32)
        set(FINE_MESSAGE_COMPILER ${FineObj_MESSAGE_COMPILER} -I ${FINE_ROOT}/FineObjects/Res)
    else()
        set(FINE_MESSAGE_COMPILER ${FineObj_MESSAGE_COMPILER} -posix -I ${FINE_ROOT}/FineObjects/Res)
    endif()
endif()

if(IOS OR ANDROID)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
endif()

message(STATUS "Found FineObjects: ${FineObj_FOUND}")
