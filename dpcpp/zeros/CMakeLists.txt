cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

set(CMAKE_CXX_COMPILER dpcpp)
set(CMAKE_C_COMPILER gcc)
project(dpcpp CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../../cmake/Modules")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS yes)

include(SetDefaultBuildRelWithDebInfo)
include(AlwaysColourCompilation)

find_package(FPGA REQUIRED)
include_directories(../common/include)

add_subdirectory(../../h5read h5read)
add_subdirectory(../../dependencies/fmt fmt)

add_executable(load load.cc)
target_link_libraries(load h5read fmt::fmt-header-only)
fpga_add_variants(load)
