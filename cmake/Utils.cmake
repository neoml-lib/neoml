include(GoogleTest)

function(add_gtest_for_target TARGET_NAME MATH_ENGINE_TYPE WORKING_DIR)

    if(WIN32)
        if(MSVC)
            target_compile_options(${TARGET_NAME} PRIVATE /wd4305 /wd4996)
        endif()

        get_target_property(LIB_TYPE NeoMathEngine TYPE)
        if(LIB_TYPE STREQUAL "SHARED_LIBRARY")
            add_custom_command(TARGET ${TARGET_NAME} POST_BUILD 
                COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:NeoMathEngine> $<TARGET_FILE_DIR:${TARGET_NAME}>
                COMMENT "Copy NeoMathEngine to ${TARGET_NAME} binary dir to discover tests."
            )
        endif()
        
        if(TARGET NeoML)
            get_target_property(LIB_TYPE NeoML TYPE)
            if(LIB_TYPE STREQUAL "SHARED_LIBRARY")
                add_custom_command(TARGET ${TARGET_NAME} POST_BUILD 
                    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:NeoML> $<TARGET_FILE_DIR:${TARGET_NAME}>
                    COMMENT "Copy NeoML to ${TARGET_NAME} binary dir to discover tests."
                )
            endif()
            if(USE_FINE_OBJECTS)
                set(ARCH_SUFFIX "")
                if(CMAKE_SIZEOF_VOID_P EQUAL 8)
                    set(ARCH_SUFFIX ".x64")
                endif()
                add_custom_command(TARGET ${TARGET_NAME} POST_BUILD 
                    COMMAND ${CMAKE_COMMAND} -E copy ${FINE_ROOT}/Win${FINE_BUILD_TYPE}${ARCH_SUFFIX}/FineObj.dll $<TARGET_FILE_DIR:${TARGET_NAME}>
                    COMMENT "Copy FineObjects"
                )
            endif()
        endif()
    endif()

    string(TOLOWER ${MATH_ENGINE_TYPE} TYPE)
    gtest_discover_tests(${TARGET_NAME}
        TEST_SUFFIX .${TYPE}
        TEST_LIST ${ENGINE_TYPE}_TESTS
        EXTRA_ARGS --MathEngine=${TYPE}
        WORKING_DIRECTORY ${WORKING_DIR}
        DISCOVERY_TIMEOUT 60
    )
endfunction()

#Add gtest target
macro(add_gtest_target)
    set(BUILD_GMOCK OFF CACHE BOOL "")
    set(INSTALL_GTEST OFF CACHE BOOL "")
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

    unset(CMAKE_CXX_VISIBILITY_PRESET)

    if(IOS)
        string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-command-line-argument")
    endif()

    if(NOT USE_FINE_OBJECTS)

        include(FetchContent)

        FetchContent_Declare(
            GoogleTest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG release-1.10.0
        )

        FetchContent_GetProperties(GoogleTest)
        if(NOT googletest_POPULATED)
            FetchContent_Populate(GoogleTest)
            add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
        endif()
    else()
        set(CMAKE_CXX_STANDARD 11)
        add_subdirectory(${FINE_ROOT}/FineObjects/FineGTest/gmock-1.7.0/gtest ${CMAKE_BINARY_DIR}/gmock-1.7.0/gtest EXCLUDE_FROM_ALL)
        if(NOT WIN32)
            target_compile_options(gtest PUBLIC -Wno-sign-compare)
        endif()
    endif()
endmacro()
