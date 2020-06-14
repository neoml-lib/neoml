include(Utils)

if(NOT TARGET gtest)
    add_gtest_target()
endif()

add_subdirectory(common)
add_subdirectory(inference)

if(NOT ANDROID AND NOT IOS)
    add_subdirectory(learn)
endif()