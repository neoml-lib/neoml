project(NeoMathEngineFullTestDesktop)

include(Utils)

add_subdirectory(../src ${CMAKE_CURRENT_BINARY_DIR}/src)

add_executable(${PROJECT_NAME}
    FullTestDesktop.cpp
)

set_target_properties( ${PROJECT_NAME} PROPERTIES
    UNITY_BUILD_MODE BATCH
    UNITY_BUILD_BATCH_SIZE ${NeoML_UNITY_BUILD_BATCH_SIZE}
)

if(MSVC)
    target_link_options(${PROJECT_NAME} PRIVATE "/SUBSYSTEM:Console")
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE 
    InferenceTests
    LearnTests

    CommonTestTools
    gtest
    NeoMathEngine
)

configure_target(${PROJECT_NAME})

if(USE_FINE_OBJECTS AND (IOS OR DARWIN))
    #TODO: fix FineGTest
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-copy>
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-warning-option>)
endif()

add_gtest_for_target(${PROJECT_NAME} CPU ${CMAKE_CURRENT_SOURCE_DIR})
if(WIN32 AND CMAKE_CUDA_COMPILER)
    add_gtest_for_target(${PROJECT_NAME} GPU ${CMAKE_CURRENT_SOURCE_DIR})
endif()

if(USE_FINE_OBJECTS)
    fine_install(TARGETS ${PROJECT_NAME})
else()
    install(
        TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()
