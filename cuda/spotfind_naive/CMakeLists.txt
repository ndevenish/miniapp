cmake_minimum_required(VERSION 3.25.2 FATAL_ERROR)

project(spotfind_naive CXX CUDA)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS yes)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../../cmake/Modules")

include(SetDefaultBuildRelWithDebInfo)
include(AlwaysColourCompilation)

include_directories(../common/include)

find_package(Dials)

add_subdirectory(../../dependencies/fmt fmt)
add_subdirectory(../../h5read h5read)
# add_subdirectory(../../dependencies/TinyTIFF TinyTIFF)
add_subdirectory(../../dependencies/argparse argparse)

add_executable(spotfind_naive spotfind_naive.cu)
target_link_libraries(spotfind_naive fmt h5read argparse)
target_compile_options(spotfind_naive PRIVATE "$<$<CONFIG:Debug,RelWithDebInfo>:-G>")

if (DIALS_FOUND)
    target_compile_definitions(spotfind_naive PRIVATE HAVE_DIALS)
    add_library(baseline SHARED ../../baseline/baseline.cpp)
    # add_library(baseline SHARED baseline.cpp )
    target_link_libraries(baseline PUBLIC Dials::Dials h5read)
else()
    message(STATUS "Could not find Dials, please set DIALS_BUILD")
endif()