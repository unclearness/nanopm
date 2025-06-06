cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME nanopm)
project(${PROJECT_NAME} LANGUAGES CXX VERSION 0.0.1 DESCRIPTION "A header-only PatchMatch")

set(CMAKE_VERBOSE_MAKEFILE TRUE)

# .lib
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# .dll and .exe
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin) 

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

get_directory_property(hasParent PARENT_DIRECTORY)
if(hasParent)
  message(STATUS "Has a parent scope.")
else()
  message(STATUS "Doesn't have a parent scope.")
endif()

function(set_with_parent ARG_NAME ARG_PATH ARG_TEXT)
    set(${ARG_NAME} ${ARG_PATH} CACHE PATH ${ARG_TEXT} FORCE)
    if (hasParent)
      set(${ARG_NAME} ${ARG_PATH} PARENT_SCOPE)
    endif()
endfunction(set_with_parent)

set(Currender_LIBS)
set(Currender_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_SOURCE_DIR}/include;${CMAKE_CURRENT_SOURCE_DIR}/src")

# third_party directries
if(NOT DEFINED STB_INSTALL_DIR)
  set(STB_INSTALL_DIR "third_party/stb" CACHE PATH "stb installed directory")
endif()
message("STB_INSTALL_DIR: ${STB_INSTALL_DIR}") 
if(NOT DEFINED LODEPNG_INSTALL_DIR)
  set(LODEPNG_INSTALL_DIR "third_party/lodepng" CACHE PATH "LodePNG installed directory")
endif()
if(NOT DEFINED TINYCOLORMAP_INSTALL_DIR)
  set(TINYCOLORMAP_INSTALL_DIR "third_party/tinycolormap" CACHE PATH "tinycolormap installed directory")
endif()

# switch for optional libraries
option(NANOPM_USE_OPENMP "Use OpenMP to enable parallelization" ON)
message("NANOPM_USE_OPENMP: ${NANOPM_USE_OPENMP}")
if(NANOPM_USE_OPENMP)
  add_definitions(-DNANOPM_USE_OPENMP)
endif()

option(NANOPM_USE_STB "Use stb to enable image i/o" ON)
message("NANOPM_USE_STB: ${NANOPM_USE_STB}")
if(NANOPM_USE_STB)
  add_definitions(-DNANOPM_USE_STB)
endif()

option(NANOPM_USE_OPENCV "Use OpenCV as Image class" OFF)
message("NANOPM_USE_OPENCV: ${NANOPM_USE_OPENCV}")
if(NANOPM_USE_OPENCV)
  find_package(OpenCV REQUIRED)
  add_definitions(-DNANOPM_USE_OPENCV)
endif()

# For OpenMP
if(NANOPM_USE_OPENMP)
  find_package(OpenMP REQUIRED)
  if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${STB_INSTALL_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

if (NOT hasParent)
  add_executable(reconstruction
    nanopm.h
    example/reconstruction.cc)
  target_link_libraries(reconstruction
  ${OpenCV_LIBS}
  )
endif()

if (WIN32)
  #set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ${EXAMPLES_EXE})
endif()

