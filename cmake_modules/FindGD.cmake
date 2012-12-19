# - Find GD
# Find the native GD includes and library
# This module defines
#  GD_INCLUDE_DIR, where to find gd.h, etc.
#  GD_LIBRARIES, the libraries needed to use GD.
#  GD_FOUND, If false, do not try to use GD.
# also defined, but not for general use are
#  GD_LIBRARY, where to find the GD library.
#  GD_SUPPORTS_PNG, GD_SUPPORTS_JPEG, GD_SUPPORTS_GIF, test
#  support for image formats in GD.

FIND_PATH(GD_INCLUDE_DIR gd.h
/usr/local/include
/usr/include
)

if(WIN32 AND NOT CYGWIN)
  SET(GD_NAMES ${GD_NAMES} bgd)
else(WIN32)
  SET(GD_NAMES ${GD_NAMES} gd)
endif(WIN32 AND NOT CYGWIN)

FIND_LIBRARY(GD_LIBRARY
  NAMES ${GD_NAMES}
  PATHS /usr/lib64 /usr/lib /usr/local/lib
  )

IF (GD_LIBRARY AND GD_INCLUDE_DIR)
    SET(GD_LIBRARIES ${GD_LIBRARY})
    SET(GD_FOUND "YES")
ELSE (GD_LIBRARY AND GD_INCLUDE_DIR)
  SET(GD_FOUND "NO")
ENDIF (GD_LIBRARY AND GD_INCLUDE_DIR)
IF (GD_FOUND)
	IF (WIN32 AND NOT CYGWIN)
		SET(GD_SUPPORTS_PNG ON)
		SET(GD_SUPPORTS_JPEG ON)
		SET(GD_SUPPORTS_GIF ON)
		get_filename_component(GD_LIBRARY_DIR ${GD_LIBRARY} PATH)
	ELSE (WIN32 AND NOT CYGWIN)
		INCLUDE(CheckLibraryExists)
		GET_FILENAME_COMPONENT(GD_LIB_PATH ${GD_LIBRARY} PATH)
		GET_FILENAME_COMPONENT(GD_LIB ${GD_LIBRARY} NAME)

		CHECK_LIBRARY_EXISTS("${GD_LIBRARY}" "gdImagePng" "${GD_LIB_PATH}" GD_SUPPORTS_PNG)
		IF (GD_SUPPORTS_PNG)
			find_package(PNG)
			IF (PNG_FOUND)
				SET(GD_LIBRARIES ${GD_LIBRARIES} ${PNG_LIBRARIES})
			SET(GD_INCLUDE_DIR ${GD_INCLUDE_DIR} ${PNG_INCLUDE_DIR})
			ELSE (PNG_FOUND)
				SET(GD_SUPPORTS_PNG "NO")
			ENDIF (PNG_FOUND)
		ENDIF (GD_SUPPORTS_PNG)

		CHECK_LIBRARY_EXISTS("${GD_LIBRARY}" "gdImageJpeg" "${GD_LIB_PATH}" GD_SUPPORTS_JPEG)
		IF (GD_SUPPORTS_JPEG)
			find_package(JPEG)
			IF (JPEG_FOUND)
				SET(GD_LIBRARIES ${GD_LIBRARIES} ${JPEG_LIBRARIES})
				SET(GD_INCLUDE_DIR ${GD_INCLUDE_DIR} ${JPEG_INCLUDE_DIR})
			ELSE (JPEG_FOUND)
				SET(GD_SUPPORTS_JPEG "NO")
			ENDIF (JPEG_FOUND)
		ENDIF (GD_SUPPORTS_JPEG)

		CHECK_LIBRARY_EXISTS("${GD_LIBRARY}" "gdImageGif" "${GD_LIB_PATH}" GD_SUPPORTS_GIF)

		# Trim the list of include directories
		SET(GDINCTRIM)
		FOREACH(GD_DIR ${GD_INCLUDE_DIR})
			SET(GD_TMP_FOUND OFF)
			FOREACH(GD_TRIMMED ${GDINCTRIM})
				IF ("${GD_DIR}" STREQUAL "${GD_TRIMMED}")
				SET(GD_TMP_FOUND ON)
				ENDIF ("${GD_DIR}" STREQUAL "${GD_TRIMMED}")
			ENDFOREACH(GD_TRIMMED ${GDINCTRIM})
			IF (NOT GD_TMP_FOUND)
				SET(GDINCTRIM "${GDINCTRIM}" "${GD_DIR}")
			ENDIF (NOT GD_TMP_FOUND)
		ENDFOREACH(GD_DIR ${GD_INCLUDE_DIR})
		SET(GD_INCLUDE_DIR ${GDINCTRIM})

		SET(GD_LIBRARY_DIR)

		# Generate trimmed list of library directories and list of libraries
		FOREACH(GD_LIB ${GD_LIBRARIES})
			GET_FILENAME_COMPONENT(GD_NEXTLIBDIR ${GD_LIB} PATH)
			SET(GD_TMP_FOUND OFF)
			FOREACH(GD_LIBDIR ${GD_LIBRARY_DIR})
				IF ("${GD_NEXTLIBDIR}" STREQUAL "${GD_LIBDIR}")
					SET(GD_TMP_FOUND ON)
				ENDIF ("${GD_NEXTLIBDIR}" STREQUAL "${GD_LIBDIR}")
			ENDFOREACH(GD_LIBDIR ${GD_LIBRARIES})
			IF (NOT GD_TMP_FOUND)
				SET(GD_LIBRARY_DIR "${GD_LIBRARY_DIR}" "${GD_NEXTLIBDIR}")
			ENDIF (NOT GD_TMP_FOUND)
		ENDFOREACH(GD_LIB ${GD_LIBRARIES})
	ENDIF (WIN32 AND NOT CYGWIN)
ENDIF (GD_FOUND)

IF (GD_FOUND)
   IF (NOT GD_FIND_QUIETLY)
      MESSAGE(STATUS "Found GD: ${GD_LIBRARY}")
   ENDIF (NOT GD_FIND_QUIETLY)
ELSE (GD_FOUND)
   IF (GD_FIND_REQUIRED)
      MESSAGE(FATAL_ERROR "Could not find GD library")
   ENDIF (GD_FIND_REQUIRED)
ENDIF (GD_FOUND)

MARK_AS_ADVANCED(
  GD_LIBRARY
  GD_LIBRARIES
  GD_INCLUDE_DIR
  GD_LIBRARY_DIR
  GD_SUPPORTS_PNG
  GD_SUPPORTS_JPEG
  GD_SUPPORTS_GIF
)
