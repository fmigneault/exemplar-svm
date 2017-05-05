# - Find FeatureExtractorHOG
# HOG feature extraction from image (MATLAB/C++)
# available at https://bitbucket.org/KenjiKyoTeam//FeatureExtractorHOG/
#
# The module defines the following variables:
#  FEHOG_FOUND - the system has mvector
#  FEHOG_INCLUDE_DIR - where to find header files
#  FEHOG_SOURCE_DIR - where to find source files
#  FEHOG_ROOT_DIR - root dir (ex. /usr/local)

#=============================================================================

# set paths
find_path ( FEHOG_INCLUDE_DIR
	NAMES
		feHOG.h
	PATHS
		${FEHOG_ROOT_DIR}
	PATH_SUFFIXES
		src
	DOC
		"FeatureExtractorHOG include directory"
)

if ( FEHOG_INCLUDE_DIR )
  set ( FEHOG_INCLUDE_DIRS ${FEHOG_INCLUDE_DIR} )
endif ()

find_path ( FEHOG_SOURCE_DIR
	NAMES
		feHOG.cpp
	PATHS
		${FEHOG_ROOT_DIR}
	PATH_SUFFIXES
		src
	DOC
		"FeatureExtractorHOG source directory"
)

file(GLOB FEHOG_SOURCE_FILES "${FEHOG_SOURCE_DIR}/*.cpp" "${FEHOG_SOURCE_DIR}/*.c")
file(GLOB FEHOG_HEADER_FILES "${FEHOG_INCLUDE_DIR}/*.hpp" "${FEHOG_INCLUDE_DIR}/*.h")


# handle REQUIRED and QUIET options
include ( FindPackageHandleStandardArgs )
find_package_handle_standard_args ( FEHOG DEFAULT_MSG
	FEHOG_ROOT_DIR
	FEHOG_INCLUDE_DIR
	FEHOG_INCLUDE_DIRS
	FEHOG_SOURCE_DIR
	FEHOG_SOURCE_FILES  
	FEHOG_HEADER_FILES
)

mark_as_advanced (
	FEHOG_ROOT_DIR
	FEHOG_INCLUDE_DIR
	FEHOG_INCLUDE_DIRS
	FEHOG_SOURCE_DIR
	FEHOG_SOURCE_FILES
	FEHOG_HEADER_FILES
)