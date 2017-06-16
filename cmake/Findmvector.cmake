# - Find mvector
# Utility multi-dimensional vector type 
# original available at https://github.com/carlobaldassi/mvector/
# fork available at https://github.com/KenjiKyo/mvector/
#
# The module defines the following variables:
#  MVECTOR_FOUND - the system has mvector
#  MVECTOR_INCLUDE_DIR - where to find mvector.h
#  MVECTOR_ROOT_DIR - root dir (ex. /usr/local)

#=============================================================================

set(MVECTOR_ROOT_DIR $ENV{MVECTOR_ROOT_DIR})

# set MVECTOR_INCLUDE_DIR
find_path(MVECTOR_INCLUDE_DIR
  NAMES mvector.h
  PATHS ${MVECTOR_ROOT_DIR}
  DOC 	"mvector include directory"
)
