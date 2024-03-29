cmake_minimum_required(VERSION 3.20)
# Constraints on minimum version:
# - h5read: FindHDF5 only creates interface targets on 3.20+

project(spotfind_naive CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS yes)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../../cmake/Modules")

include(SetDefaultBuildRelWithDebInfo)
include(AlwaysColourCompilation)
include(UseSpanBackportIfNeeded)

include_directories(../common/include)

add_subdirectory(../../dependencies/fmt fmt)
add_subdirectory(../../h5read h5read)
# add_subdirectory(../../dependencies/TinyTIFF TinyTIFF)
add_subdirectory(../../dependencies/argparse argparse)
add_subdirectory(../../baseline baseline)

add_executable(spotfind_naive spotfind_naive.cu)
target_link_libraries(spotfind_naive PRIVATE fmt h5read argparse standalone)
target_compile_options(spotfind_naive PRIVATE "$<$<CONFIG:Debug>:-G>")
