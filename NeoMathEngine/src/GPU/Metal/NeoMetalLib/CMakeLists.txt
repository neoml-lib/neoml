project(NeoMetalLib)

set(METAL_LIB ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.metallib)

set(METAL_SOURCES
    MetalDnnBlasKernels.metal
    MetalDnnConvKernels.metal
    MetalDnnPoolingKernels.metal
    MetalDnnKernels.metal
    MetalVectorMathKernels.metal
)

set(AIRFILES)
foreach(METALFILE ${METAL_SOURCES})
    string(REPLACE ".metal" ".air" AIRFILE ${METALFILE})
    add_custom_command(
        OUTPUT ${AIRFILE} 
        COMMAND xcrun -sdk iphoneos metal -O3 -std=ios-metal1.2 -mios-version-min=10.0 -c -I${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/${METALFILE} -o ${AIRFILE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${METALFILE}
    ) 
    list(APPEND AIRFILES ${AIRFILE})
endforeach()

add_custom_command(
    OUTPUT ${PROJECT_NAME}.metallib
    COMMAND xcrun -sdk iphoneos metallib ${AIRFILES} -o ${PROJECT_NAME}.metallib
    DEPENDS ${AIRFILES}
)

add_custom_target(${PROJECT_NAME} DEPENDS ${PROJECT_NAME}.metallib)

set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME ${METAL_LIB})

if(NOT NeoMathEngine_BUILD_SHARED AND NeoML_INSTALL)
    if(USE_FINE_OBJECTS)
        fine_install(FILES ${METAL_LIB})
    else()
        install(FILES ${METAL_LIB} DESTINATION ${CMAKE_INSTALL_LIBDIR})
    endif()
endif()