#=============================================================================
# - Find FeatureExtractorHOG (feHOG)
# HOG feature extraction from image
# available at https://bitbucket.org/KenjiKyoTeam//FeatureExtractorHOG/
#
# The module defines the following variables:
#  feHOG_FOUND
#  feHOG_ROOT_DIR
#=============================================================================

set( feHOG_ROOT_DIR "feHOG_ROOT_DIR-NOTFOUND" CACHE PATH "FeatureExtractorHOG root directory")

# handle REQUIRED and QUIET options
include ( FindPackageHandleStandardArgs )
find_package_handle_standard_args ( feHOG DEFAULT_MSG
	feHOG_ROOT_DIR
    feHOG_BINARY_DIRS
    feHOG_INCLUDE_DIRS
    feHOG_LIBRARY_DIRS    
    feHOG_LIBRARY_DEBUG
    feHOG_LIBRARY_RELEASE
)
