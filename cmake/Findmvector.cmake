# - Find mvector
# Utility multi-dimensional vector type 
# original available at https://github.com/carlobaldassi/mvector/
# fork available at https://github.com/KenjiKyo/mvector/
#
# The module defines the following variables:
#  mvector_FOUND - the system has mvector
#  mvector_INCLUDE_DIR - where to find mvector.h
#  mvector_ROOT_DIR - root dir (ex. /usr/local)

#=============================================================================

set(mvector_ROOT_DIR $ENV{MVECTOR_ROOT_DIR})

# set MVECTOR_INCLUDE_DIR
find_path(mvector_INCLUDE_DIR
  NAMES mvector.h
  HINTS
    ${mvector_DIR}
    ${mvector_DIR}/include
    ${mvector_ROOT}
    ${mvector_ROOT}/include
    $ENV{mvector_ROOT}
    $ENV{mvector_ROOT}/include
    ${CMAKE_CURRENT_LIST_DIR}/../mvector
    ${CMAKE_CURRENT_LIST_DIR}/../mvector/include
  PATHS 
    ${mvector_ROOT_DIR}
    /usr/include
    /usr/local/include
    /usr/local/include/mvector
  DOC 	"mvector include directory"
)

