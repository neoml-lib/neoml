project(CommonTestTools)

add_library(${PROJECT_NAME}
	TestFixture.cpp
	TestFixture.h
	TestParams.cpp
	TestParams.h
)

configure_target(${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${PROJECT_NAME} PRIVATE gtest NeoMathEngine)