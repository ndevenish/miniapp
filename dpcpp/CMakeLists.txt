cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

set(CMAKE_CXX_COMPILER dpcpp)
project(dpcpp CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../cmake/Modules")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS yes)

include(SetDefaultBuildRelWithDebInfo)
include(AlwaysColourCompilation)

find_package(FPGA REQUIRED)

add_subdirectory(../h5read h5read)

add_executable(load load.cc)
target_link_libraries(load h5read)

### FPGA Emulator
add_executable(load.fpga_emu load.cc)
target_link_libraries(load.fpga_emu FPGA::EMULATED h5read)

### Generate FPGA Report
add_executable(load_report.a load.cc)
target_link_libraries(load_report.a PRIVATE h5read FPGA::FPGA)
target_link_options(load_report.a PRIVATE "-fsycl-link=early")

# FPGA Hardware
add_executable(load.fpga load.cc)
target_link_libraries(load.fpga h5read FPGA::FPGA)
set_target_properties(load.fpga PROPERTIES EXCLUDE_FROM_ALL yes)