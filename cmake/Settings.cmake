
# Define global cmake variables and some usefull variables
macro(set_global_variables)
    
    # Usefull variables
    if(UNIX AND APPLE AND NOT IOS)
        set(DARWIN TRUE)
    elseif(UNIX AND NOT APPLE AND NOT ANDROID)
        set(LINUX TRUE)
    endif()

    # Cmake default variables
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    
    set(CMAKE_CXX_VISIBILITY_PRESET hidden)

    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
    
    if(LINUX)
        string(APPEND CMAKE_SHARED_LINKER_FLAGS " -Wl,--no-undefined")
    endif()
    
    if(LINUX OR DARWIN)
        set(CMAKE_SKIP_RPATH FALSE)
        set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
        if(LINUX)
            set(CMAKE_INSTALL_RPATH "\$ORIGIN" "\$ORIGIN/../lib")
        elseif(DARWIN)
            set(CMAKE_INSTALL_NAME_DIR "@rpath")
            set(CMAKE_INSTALL_RPATH "@executable_path" "@executable_path/../lib")
        endif()
    endif()

    if(MSVC AND NOT USE_FINE_OBJECTS)
        # parallel compilation on msvc when build is OSS
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/MP>)
    endif()

    if(WARNINGS_ARE_ERRORS)
        if(MSVC)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/WX>)
        elseif(NOT WIN32)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Werror>)
        endif()
    endif()

endmacro()

# Some settings for target
function(configure_target TARGET_NAME)
    if(NOT WIN32)
        target_compile_options(${TARGET_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall>
            $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
            $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
            $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
            $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-value>
            $<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-pragmas>
            $<$<COMPILE_LANGUAGE:CXX>:-Wno-strict-overflow>
        )
    endif()
    
    # No extensions use
    set_target_properties(${TARGET_NAME} PROPERTIES 
        CXX_STANDARD_REQUIRED YES 
        CXX_EXTENSIONS OFF
    )
    
    if(IOS)
        set_target_properties(${TARGET_NAME} PROPERTIES 
            XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET 11.0
            XCODE_ATTRIBUTE_ENABLE_BITCODE "YES"
            XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH[variant=Debug] "YES"
        )

        get_target_property(TARGET_TYPE ${TARGET_NAME} TYPE)
        if(TARGET_TYPE STREQUAL "SHARED_LIBRARY")
            string(TOLOWER ${TARGET_NAME} LOWERCASE_PROJECT_NAME)
            set_target_properties(${TARGET_NAME} PROPERTIES
                FRAMEWORK TRUE
                BUILD_WITH_INSTALL_RPATH TRUE
                INSTALL_NAME_DIR "@rpath"
                MACOSX_FRAMEWORK_IDENTIFIER "com.abbyy.${LOWERCASE_PROJECT_NAME}"
                MACOSX_FRAMEWORK_SHORT_VERSION_STRING ${FINE_VERSION_MAJOR}.${FINE_VERSION_MINOR}.${FINE_VERSION_PATCH}
                MACOSX_FRAMEWORK_BUNDLE_VERSION ${FINE_VERSION_MAJOR}.${FINE_VERSION_MINOR}.${FINE_VERSION_PATCH}
                XCODE_ATTRIBUTE_LD_RUNPATH_SEARCH_PATHS "$(inherited) @executable_path/Frameworks @loader_path/Frameworks"
            )
            if(USE_FINE_OBJECTS)
                set_target_properties(${TARGET_NAME} PROPERTIES
                    XCODE_ATTRIBUTE_GCC_GENERATE_DEBUGGING_SYMBOLS "YES"
                )
            endif()
        endif()
    elseif(WIN32)
        if(USE_FINE_OBJECTS)
            set_target_properties(${TARGET_NAME} PROPERTIES ARCHIVE_OUTPUT_NAME ${TARGET_NAME}.${WIN_ARCH_SUFFIX}.${FINE_BUILD_TYPE})
            # Generate debug info for Release config
            target_link_options(${TARGET_NAME} PRIVATE "$<$<CONFIG:Release>:/DEBUG>")
        endif()
    endif()
endfunction()

if(USE_FINE_OBJECTS)

    function(fine_unexport_symbols TARGET_NAME)
        get_target_property(TARGET_TYPE ${TARGET_NAME} TYPE)
        if(TARGET_TYPE STREQUAL "SHARED_LIBRARY")
            if(ANDROID OR LINUX)
                if(NOT CMAKE_GENERATOR STREQUAL "Unix Makefiles")
                    message(FATAL_ERROR "Cmake generator must be 'Unix Makefiles'!")
                endif()
                
                if(CMAKE_HOST_SYSTEM_NAME MATCHES "Linux|Darwin")
                    set(LINK_CMD_FILE link.txt)
                elseif(CMAKE_HOST_SYSTEM_NAME MATCHES "Windows")
                    set(LINK_CMD_FILE build.make)
                else()
                    message(FATAL_ERROR "Unknown host system!")
                endif()

                add_custom_command(TARGET ${PROJECT_NAME} 
                    POST_BUILD
                    COMMAND python ${FINE_ROOT}/FineObjects/Cmake/unexport.py
                        lib${PROJECT_NAME}.so ${FINE_ROOT}/FineObjects/Cmake/unexported_symbols.txt 
                        ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${PROJECT_NAME}.dir/${LINK_CMD_FILE}
                        COMMENT "Unexport symbols in ${PROJECT_NAME}"
                )
            elseif(DARWIN)
                target_link_options(${TARGET_NAME} PRIVATE -unexported_symbols_list ${FINE_ROOT}/FineObjects/Cmake/unexported_symbols_ios.txt)
            elseif(IOS)
                set_target_properties(${TARGET_NAME} PROPERTIES
                    XCODE_ATTRIBUTE_UNEXPORTED_SYMBOLS_FILE ${FINE_ROOT}/FineObjects/Cmake/unexported_symbols_ios.txt
                )
            endif()
        endif()
    endfunction()

endif()
