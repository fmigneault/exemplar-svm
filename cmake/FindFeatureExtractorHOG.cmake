
# - Find FeatureExtractorHOG (feHOG)
# HOG feature extraction from image
# available at https://bitbucket.org/KenjiKyoTeam//FeatureExtractorHOG/
#
# The module defines the following variables:
#  FEHOG_FOUND
#  FEHOG_ROOT_DIR
#=============================================================================

set( FEHOG_ROOT_DIR "FEHOG_ROOT_DIR-NOTFOUND" CACHE PATH "FeatureExtractorHOG root directory")

# handle REQUIRED and QUIET options
include ( FindPackageHandleStandardArgs )
find_package_handle_standard_args ( feHOG DEFAULT_MSG
	FEHOG_ROOT_DIR	
)
