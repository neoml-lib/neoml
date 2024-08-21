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
    ArrayIteratorFOL.h
    AscendingFOL.h
    BaseFileFOL.h
    CriticalSectionFOL.h
    DescendingFOL.h
    DynamicBitSetFOL.h
    DynamicBitSetIteratorFOL.h
    FastArrayFOL.h
    FineObjLite.h
    ErrorsFOL.h
    HashTableAllocatorFOL.h
    HashTableFOL.h
    HashTableIteratorFOL.h
    IntervalFOL.h
    MapFOL.h
    MapIteratorFOL.h
    MapPositionIteratorFOL.h
    MathFOL.h
    MemoryFileFOL.h
    ObjectFOL.h
    PointerArrayFOL.h
    PriorityQueueFOL.h
    PtrOwnerFOL.h
    SortFOL.h
    StringFOL.h
    TextStreamFOL.h
)
foreach(SRC ${${PROJECT_NAME}_SOURCES})
    target_sources(${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${SRC}>)
endforeach()




