project(NeoMathEngineAvx)

add_library(${PROJECT_NAME} SHARED common.cpp)

target_sources(${PROJECT_NAME}
    PRIVATE

    # Sources
    ./CpuMathEngineDnnConvAvx.cpp
)

string(TOUPPER ${CMAKE_SYSTEM_NAME} UPPERCASE_CMAKE_SYSTEM_NAME)
target_compile_definitions(${PROJECT_NAME} PUBLIC _${UPPERCASE_CMAKE_SYSTEM_NAME})

target_include_directories(${PROJECT_NAME}
    PRIVATE
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../../include>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../CPU>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

# Add some definitions
target_compile_definitions(${PROJECT_NAME}
	PUBLIC
		"$<$<OR:$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:_RELEASE>"
		"$<$<CONFIG:Debug>:_DEBUG>"
		"$<$<CONFIG:Release>:_FINAL>"
	PRIVATE
		BUILD_NEOMATHENGINE
)

# OpenMP
if(NOT IOS)
	find_package(OpenMP REQUIRED)
	target_compile_definitions(${PROJECT_NAME} PUBLIC NEOML_USE_OMP)
	string(REPLACE " " ";" OpenMP_CXX_FLAGS ${OpenMP_CXX_FLAGS})
	target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
	target_include_directories(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_INCLUDE_DIRS}>)
	if(OpenMP_CXX_LIBRARIES)
		if(ANDROID)
			get_filename_component(OMP_SUFFIX ${OpenMP_omp_LIBRARY} EXT)
			if(OMP_SUFFIX STREQUAL ".so")
				target_link_libraries(${PROJECT_NAME} PUBLIC -fopenmp -static-openmp)
			else()
				target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenMP_CXX_LIBRARIES})
			endif()
		else()
			target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenMP_CXX_LIBRARIES})
		endif()
	endif()
endif()

# AVX
if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 4)
    target_compile_options(${PROJECT_NAME} PRIVATE /arch:SSE2)
elseif(LINUX OR DARWIN OR (ANDROID AND ANDROID_ABI MATCHES "^x86.*") OR (IOS_ARCH MATCHES "^x86.*"))
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx -mfma>)
endif()

# Install
if(NeoMathEngine_INSTALL)
	if(USE_FINE_OBJECTS)
		fine_install(TARGETS ${PROJECT_NAME})
	else()
		install(
			TARGETS ${PROJECT_NAME}
			LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
			)
	endif()
endif()
