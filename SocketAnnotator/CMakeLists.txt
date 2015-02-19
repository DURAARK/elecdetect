cmake_minimum_required(VERSION 2.8) 
project (SocketAnnotator)

# OpenCV
FIND_PACKAGE(OpenCV REQUIRED)
IF (OpenCV_FOUND)
  MESSAGE(STATUS "OpenCV was found.")
  INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIR})
ELSE (OpenCV_FOUND)
  MESSAGE(FATAL_ERROR "OpenCV was not found.")
ENDIF (OpenCV_FOUND)

# OpenMP
#FIND_PACKAGE(OpenMP)
#IF (OPENMP_FOUND)
#  MESSAGE(STATUS "OpenMP was found.")
#ELSE (OPENMP_FOUND)
#  MESSAGE(FATAL_ERROR "OpenMP was not found.")
#ENDIF (OPENMP_FOUND)


SET(CMAKE_SOURCE_DIR source)

FILE( GLOB SA_SRC source/*.cpp )
FILE( GLOB SA_HEADERS source/*.h )
FILE( GLOB TCLAP_HEADERS source/tclap/*.h )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -Wall")

if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Debug )
endif()

IF(CMAKE_BUILD_TYPE MATCHES "[Dd][Ee][Bb][Uu][Gg]")
  add_definitions(-D_DEBUG)
endif()


# INCLUDE THE CORE HEADERS
include_directories( ${CMAKE_SOURCE_DIR})
link_directories( ${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable( SocketAnnotator ${SA_HEADERS} ${SA_SRC} ${TCLAP_HEADERS})

target_link_libraries( SocketAnnotator ${OpenCV_LIBS})
