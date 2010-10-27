# - Find PLASMA library
# This module finds an installed  library that implements the PLASMA
# linear-algebra interface (see http://icl.cs.utk.edu/plasma/).
# The list of libraries searched for is taken
# from the autoconf macro file, acx_blas.m4 (distributed at
# http://ac-archive.sourceforge.net/ac-archive/acx_blas.html).
#
# This module sets the following variables:
#  PLASMA_FOUND - set to true if a library implementing the PLASMA interface
#    is found
#  PLASMA_LINKER_FLAGS - uncached list of required linker flags (excluding -l
#    and -L).
#  PLASMA_PKG_DIR - Directory where the PLASMA pkg file is stored
#  PLASMA_LIBRARIES - uncached list of libraries (using full path name) to
#    link against to use PLASMA
#  PLASMA_STATIC  if set on this determines what kind of linkage we do (static)
#  PLASMA_VENDOR  if set checks only the specified vendor, if not set checks
#     all the possibilities
##########

get_property(_LANGUAGES_ GLOBAL PROPERTY ENABLED_LANGUAGES)
if(NOT _LANGUAGES_ MATCHES Fortran)
  if(PLASMA_FIND_REQUIRED)
    message(FATAL_ERROR "Find PLASMA requires Fortran support so Fortran must be enabled.")
  else(PLASMA_FIND_REQUIRED)
    message(STATUS "Looking for PLASMA... - NOT found (Fortran not enabled)") #
    return()
  endif(PLASMA_FIND_REQUIRED)
endif(NOT _LANGUAGES_ MATCHES Fortran)

unset(PLASMA_COMPILE_SUCCESS)

# First we try to use pkg-config to find what we're looking for
# in the directory specified by the PLASMA_DIR or PLASMA_PKG_DIR
include(FindPkgConfig)
if(PLASMA_DIR)
  if(NOT PLASMA_PKG_DIR)
    set(PLASMA_PKG_DIR "${PLASMA_DIR}/lib/pkgconfig")
  endif(NOT PLASMA_PKG_DIR)
endif(PLASMA_DIR)

set(ENV{PKG_CONFIG_PATH} "${PLASMA_PKG_DIR}:$ENV{PKG_CONFIG_PATH}")
pkg_search_module(PLASMA plasma)
if(PKG_CONFIG_FOUND)
  if(NOT PLASMA_FOUND)
    message(FATAL_ERROR "No detection of PLASMA except via pkg-config available yet.")
  endif(NOT PLASMA_FOUND)
  # Validate the include file <plasma.h>
  find_path(PLASMA_INCLUDE_FOUND
    plasma.h
    "${PLASMA_INCLUDE_DIRS}"
    )
  if(NOT PLASMA_INCLUDE_FOUND)
    if(PLASMA_FIND_REQUIRED)
      message(FATAL_ERROR "Couln't find the plasma.h header in ${PLASMA_INCLUDE_DIRS}")
    endif(PLASMA_FIND_REQUIRED)
  endif(NOT PLASMA_INCLUDE_FOUND)

  # Validate the library
  include(CheckCSourceCompiles)

  set(PLASMA_tmp_libraries ${CMAKE_REQUIRED_LIBRARIES})
  set(PLASMA_tmp_flags ${CMAKE_REQUIRED_FLAGS})
  set(CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES};${PLASMA_LIBRARIES}")
  string(REGEX REPLACE ";" " " PLASMA_LDFLAGS "${PLASMA_LDFLAGS}")
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${PLASMA_CFLAGS} ${PLASMA_LDFLAGS}")
  check_c_source_compiles(
    "int main(int argc, char* argv[]) {
       PLASMA_zgeqrf(); return 0;
     }"
    PLASMA_COMPILE_SUCCESS
    )
  set(${CMAKE_REQUIRED_LIBRARIES} PLASMA_tmp_libraries)
  set(${CMAKE_REQUIRED_FLAGS} PLASMA_tmp_flags)
  unset(PLASMA_tmp_libraries)
  unset(PLASMA_tmp_includes)
  unset(PLASMA_tmp_flags)
else(PKG_CONFIG_FOUND)
  message(FATAL_ERROR "pkg-config not supported on this environment.")
endif(PKG_CONFIG_FOUND)


if(NOT PLASMA_FIND_QUIETLY)
  if(PLASMA_COMPILE_SUCCESS)
    message(STATUS "A Library with PLASMA API found.")
    string(REGEX REPLACE ";" " " PLASMA_LDFLAGS "${PLASMA_LDFLAGS}")
    find_package_message(PLASMA
      "Found PLASMA: ${PLASMA_LIBRARIES}
    PLASMA_CFLAGS       = [${PLASMA_CFLAGS}]
    PLASMA_LDFLAGS      = [${PLASMA_LDFLAGS}]
    PLASMA_INCLUDE_DIRS = [${PLASMA_INCLUDE_DIRS}]
    PLASMA_LIBRARY_DIRS = [${PLASMA_LIBRARY_DIRS}]"
      "[${PLASMA_CFLAGS}][${PLASMA_LDFLAGS}][${PLASMA_INCLUDE_DIRS}][${PLASMA_LIBRARY_DIRS}]")
  else(PLASMA_COMPILE_SUCCESS)
    if(PLASMA_FIND_REQUIRED)
      message(FATAL_ERROR
        "A required library with PLASMA API not found. Please specify library location.")
    else(PLASMA_FIND_REQUIRED)
      message(STATUS
        "A library with PLASMA API not found. Please specify library location.")
    endif(PLASMA_FIND_REQUIRED)
  endif(PLASMA_COMPILE_SUCCESS)
endif(NOT PLASMA_FIND_QUIETLY)
