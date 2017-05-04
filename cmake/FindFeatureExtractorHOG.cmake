# - Find FeatureExtractorHOG
# HOG feature extraction from image (MATLAB/C++)
# available at https://bitbucket.org/KenjiKyoTeam//FeatureExtractorHOG/
#
# The module defines the following variables:
#  feHOG_FOUND
#  feHOG_ROOT_DIR

#=============================================================================

# set paths
set(feHOG_ROOT_DIR "feHOG_ROOT_DIR-NOTFOUND" CACHE PATH "FeatureExtractorHOG root directory")

# handle REQUIRED and QUIET options
include ( FindPackageHandleStandardArgs )
find_package_handle_standard_args ( feHOG DEFAULT_MSG
	feHOG_ROOT_DIR	
)
