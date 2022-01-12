project(CommonTestTools)

add_library(${PROJECT_NAME}
	TestFixture.cpp
	TestFixture.h
	TestParams.cpp
	TestParams.h
)

set_target_properties( ${PROJECT_NAME} PROPERTIES
    UNITY_BUILD_MODE GROUP
)

configure_target(${PROJECT_NAME})

if(USE_FINE_OBJECTS AND (IOS OR DARWIN))
    #TODO: fix FineGTest
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-copy>
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-warning-option>)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${PROJECT_NAME} PRIVATE gtest NeoMathEngine)
