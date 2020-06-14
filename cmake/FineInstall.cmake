include(CMakeParseArguments)

function(fine_install)
    cmake_parse_arguments(ARGS "" "" "TARGETS;DIRECTORY;FILES" ${ARGN})
    if(NOT FineObj_FOUND)
        message(NOTICE "Fine Objects is not found. Do nothing!")
        return()
    endif()

    if(WIN32)
        set(DIR_SUFFIX "")
        if(CMAKE_SIZEOF_VOID_P STREQUAL "8")
            set(DIR_SUFFIX .x64)
        endif()
        
        # Targets
        install(TARGETS ${ARGS_TARGETS}
                ARCHIVE DESTINATION ${FINE_LIBRARIES_DIR}
                LIBRARY DESTINATION ${FINE_LIBRARIES_DIR}
                RUNTIME DESTINATION ${FINE_ROOT}/Win${FINE_BUILD_TYPE}${DIR_SUFFIX}
        )

        # Directories
        install(DIRECTORY ${ARGS_DIRECTORY}
            DESTINATION ${FINE_ROOT}/Win${FINE_BUILD_TYPE}${DIR_SUFFIX}
        )

        # PDB's 
        foreach(TARGET_NAME ${ARGS_TARGETS})
            install(FILES $<TARGET_PDB_FILE:${TARGET_NAME}> DESTINATION ${FINE_ROOT}/Win${FINE_BUILD_TYPE}${DIR_SUFFIX} OPTIONAL)
        endforeach()
    else()
        install(TARGETS ${ARGS_TARGETS}
            ARCHIVE DESTINATION ${FINE_LIBRARIES_DIR}
            LIBRARY DESTINATION ${FINE_LIBRARIES_DIR}
            RUNTIME DESTINATION ${FINE_LIBRARIES_DIR}
            FRAMEWORK DESTINATION ${FINE_LIBRARIES_DIR}
        )
        install(FILES ${ARGS_FILES} DESTINATION ${FINE_LIBRARIES_DIR})
        install(DIRECTORY ${ARGS_DIRECTORY} DESTINATION ${FINE_LIBRARIES_DIR})

        # dSYM's
        if(IOS)
            foreach(TARGET_NAME ${ARGS_TARGETS})
                install(DIRECTORY $<TARGET_FILE_DIR:${TARGET_NAME}>/../${TARGET_NAME}.framework.dSYM DESTINATION ${FINE_LIBRARIES_DIR} OPTIONAL)
            endforeach()
        endif()
    endif()
endfunction()
