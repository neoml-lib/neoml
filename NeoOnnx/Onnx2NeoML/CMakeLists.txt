cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project( Onnx2NeoML LANGUAGES CXX)

include(Version)
include(Settings)
if(USE_FINE_OBJECTS)
    include(FineInstall)
else()
    include(GNUInstallDirs)
endif()

set_global_variables()

add_executable(${PROJECT_NAME} Onnx2NeoML.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE NeoML NeoOnnx NeoMathEngine)

if(USE_FINE_OBJECTS)
	target_link_libraries(${PROJECT_NAME} PRIVATE FineObjects)
endif()

if(MSVC)
	if(USE_FINE_OBJECTS)
		target_compile_definitions(${PROJECT_NAME} PRIVATE _CONSOLE)
		target_link_options(${PROJECT_NAME} PRIVATE "/ENTRY:wmainCRTStartup")
		target_link_libraries(${PROJECT_NAME} PRIVATE msvcprt$<$<CONFIG:Debug>:d>)
	else()
		target_link_options(${PROJECT_NAME} PRIVATE "/SUBSYSTEM:Console")
	endif()
endif()

configure_target(${PROJECT_NAME})
