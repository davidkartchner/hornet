cmake_minimum_required(VERSION 3.8) #language level CUDA support was added in CMake 3.8

set(PROJECT "HornetAlg")
project(${PROJECT} LANGUAGES CXX CUDA)

include(../compiler-util/CMakeLists.txt)

add_subdirectory(../hornet build)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories(../hornet/include)
include_directories(include)

###################################################################################################
# - library targets -------------------------------------------------------------------------------

file(GLOB_RECURSE CU_SRCS ${PROJECT_SOURCE_DIR}/src/*)
list(REMOVE_ITEM CU_SRCS ${PROJECT_SOURCE_DIR}/src/Dynamic/KatzCentrality/Katz.cu)
file(GLOB_RECURSE CPP_SRCS ${PROJECT_SOURCE_DIR}/externals/xlib/src/*)

add_library(hornetAlg ${CU_SRCS})

###################################################################################################
# - add executables -------------------------------------------------------------------------------


add_executable(kcore      test/KCoreTest.cu)
add_executable(corenum    test/CoreNumbers.cu)

target_link_libraries(kcore      hornetAlg hornet)
target_link_libraries(corenum    hornetAlg hornet)
