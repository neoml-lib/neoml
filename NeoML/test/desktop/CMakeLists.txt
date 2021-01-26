project(NeoMLTest)

include(Utils)

if(NOT TARGET gtest)
    add_gtest_target()
endif()

add_subdirectory(../src ${CMAKE_BINARY_DIR}/NeoML/test/src)

add_executable(${PROJECT_NAME}
    main.cpp
)

configure_target(${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} PRIVATE NeoMLTestSrc gtest NeoML)
if(USE_FINE_OBJECTS)
    target_link_libraries(${PROJECT_NAME} PRIVATE FineObjects)
    if(WIN32)
        target_compile_definitions(${PROJECT_NAME} PRIVATE _CONSOLE)
        target_link_options(${PROJECT_NAME} PRIVATE "/ENTRY:wmainCRTStartup")
        target_link_libraries(${PROJECT_NAME} PRIVATE msvcprt$<$<CONFIG:Debug>:d>)
    endif()
else()
    if(WIN32)
        target_link_options(${PROJECT_NAME} PRIVATE "/SUBSYSTEM:Console")
    endif()
    target_link_libraries(${PROJECT_NAME} PRIVATE FineObjLite)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

string(TOUPPER ${PROJECT_NAME} UPPERCASE_PROJECT_NAME)
target_compile_definitions(${PROJECT_NAME} PRIVATE BUILD_${UPPERCASE_PROJECT_NAME})

add_gtest_for_target(${PROJECT_NAME} "CPU" ${CMAKE_CURRENT_SOURCE_DIR}/..)
if(WIN32 AND CMAKE_CUDA_COMPILER)
    add_gtest_for_target(${PROJECT_NAME} "CUDA" ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()

if(USE_FINE_OBJECTS)
    fine_install(TARGETS ${PROJECT_NAME})

    if(NOT IOS AND NOT ANDROID)
        fine_install(DIRECTORY ../data DESTINATION)
    endif()
else()
    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    if(NOT IOS AND NOT ANDROID)
        install(DIRECTORY ../data DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
endif()
