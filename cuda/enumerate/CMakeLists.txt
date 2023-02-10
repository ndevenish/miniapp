cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(enumerate CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CUDA_FLAGS "-Xcudafe --diag_suppress=1675")

set(CMAKE_EXPORT_COMPILE_COMMANDS yes)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../../cmake/Modules")

include(SetDefaultBuildRelWithDebInfo)
include(AlwaysColourCompilation)

add_subdirectory(../../dependencies/fmt fmt)
# add_subdirectory(../../dependencies/TinyTIFF TinyTIFF)
# add_subdirectory(../../dependencies/argparse argparse)

add_executable(enumerate enumerate.cu)
target_link_libraries(enumerate fmt)
