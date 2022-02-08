project(FineObjLite LANGUAGES CXX)

add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME} INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/NeoML/FineObjLite>
)

set(${PROJECT_NAME}_SOURCES 
    AllocFOL.h
    ArchiveFOL.h
    ArrayFOL.h
    AscendingFOL.h
    BaseFileFOL.h
    internal/CriticalSectionFOL.h
    DescendingFOL.h
    internal/DynamicBitSetFOL.h
    ErrorsFOL.h
    internal/FastArrayFOL.h
    FineObjLite.h
    internal/HashTableAllocatorFOL.h
    internal/HashTableFOL.h
    internal/IntervalFOL.h
    internal/MapFOL.h
    MathFOL.h
    MemoryFile.h
    ObjectFOL.h
    internal/PointerArrayFOL.h
    internal/PriorityQueueFOL.h
    SortFOL.h
    StringFOL.h
    TextStreamFOL.h
)
foreach(SRC ${${PROJECT_NAME}_SOURCES})
    target_sources(${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${SRC}>)
endforeach()




