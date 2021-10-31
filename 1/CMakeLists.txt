cmake_minimum_required(VERSION 3.20)
set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
project(1)

include_directories( ./src )
set(CMAKE_CXX_STANDARD 11)
include_directories(/MPI)

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})


#add_executable(1 row_vector_matrix_multiply.cpp)
add_executable(1 column_vector_matrix_multiply.cpp)
#add_executable(1 block_vector_matrix_multiply.cpp)
target_link_libraries(1 ${MPI_C_LIBRARIES})
