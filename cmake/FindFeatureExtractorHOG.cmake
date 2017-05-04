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
  DOC
    "FeatureExtractorHOG include directory"
)

find_path ( FEHOG_SOURCE_DIR
  NAMES
    feHOG.cpp
  PATHS
    ${FEHOG_ROOT_DIR}
  DOC
    "FeatureExtractorHOG source directory"
)