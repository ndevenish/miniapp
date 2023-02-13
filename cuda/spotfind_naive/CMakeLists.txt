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
target_link_libraries(spotfind_naive PRIVATE fmt h5read argparse)
target_compile_options(spotfind_naive PRIVATE "$<$<CONFIG:Debug,RelWithDebInfo>:-G>")

add_library(baseline STATIC ../../baseline/no_tbx.cc)
target_include_directories(baseline PUBLIC ../../baseline)
target_link_libraries(baseline PUBLIC h5read)
target_link_libraries(spotfind_naive PRIVATE baseline)
