#include "esvmPaths.h"
#include "CommonCpp.h"

#include "boost/filesystem.hpp"
namespace bfs = boost::filesystem;

// helper function to avoid nullptr string initialization (error) if the environment variable was not set
// gets the environment variable value, test it for a valid value and adds the directory separator at end
std::string getValidEnvVar(const char* env_var)
{
    ASSERT_LOG(env_var, "Invalid environment variable specified (nullptr)");
    char* env_val = std::getenv(env_var);
    std::string env_var_str(env_var);
    ASSERT_LOG(env_val, "Environment variable not found or not set: " + env_var_str);
    std::string env_val_str(env_val);
    ASSERT_LOG(!env_val_str.empty(), "Environment variable cannot be empty: " + env_var_str);
    ASSERT_LOG(bfs::is_directory(env_val_str), "Environment variable is not a directory: " + env_var_str + " = '" + env_val_str + "'");
    bfs::path env_val_p(env_val_str);
    env_val_p += bfs::path::preferred_separator;
    return env_val_p.string();
}

/* -------------------------------------------
   OpenCV Cascade files for preprocessing
------------------------------------------- */

// OpenCV
#if defined(ESVM_HAS_TESTS) || ESVM_ROI_PREPROCESS_MODE == 1
const std::string sourcesOpenCV = getValidEnvVar("OPENCV_SOURCES");     // OpenCV's root directory (ie: Git level)
#endif

/* -------------------------------------------
   Dataset image paths for testing purposes
------------------------------------------- */

#ifdef ESVM_HAS_TESTS

// ESVM build/test
const std::string roiVideoImagesPath = refPath + "/img/roi/";                           // Person ROI tracks obtained from face detection + tracking
const std::string refStillImagesPath = refPath + "/img/ref/";                           // Reference high quality still ROIs for enrollment in SSPP
const std::string negativeSamplesDir = refPath + "/data/negatives/";                    // Pre-generated ChokePoint negative samples files
const std::string testingSamplesDir  = refPath + "/data/testing/";                      // Pre-generated ChokePoint probe samples files

// ChokePoint
#ifdef ESVM_HAS_CHOKEPOINT
const std::string rootChokePointPath = getValidEnvVar("CHOKEPOINT_ROOT");                       // ChokePoint dataset root
const std::string roiChokePointCroppedFacePath = rootChokePointPath + "cropped_faces/";         // Path of extracted 96x96 ROI from all videos
const std::string roiChokePointFastDTTrackPath = rootChokePointPath + "results/fast-dt/";       // Path of person track ROIs found with FAST-DT
const std::string roiChokePointEnrollStillPath = rootChokePointPath + "enroll/";                // Path of enroll still images for ChokePoint
#endif/*ESVM_HAS_CHOKEPOINT*/

// COX-S2V
#ifdef ESVM_HAS_COX_S2V
const std::string rootCOXS2VPath = getValidEnvVar("COX_S2V_ROOT");                              // COX-S2V dataset root
const std::string roiCOXS2VTestingVideoPath = rootCOXS2VPath + "COX-S2V-Video/";                // Path of enroll still images for COX-S2V
const std::string roiCOXS2VAllImgsStillPath = rootCOXS2VPath + "COX-S2V-Still/";                // Path of every individual's still for COX-S2V
const std::string roiCOXS2VEnrollStillsPath = rootCOXS2VPath + "Persons-for-Publication/";      // Path of pre-selected gallery stills
const std::string roiCOXS2VEyeLocaltionPath = rootCOXS2VPath + "Eye_location/";                 // Path of eye location ground truths
#endif/*ESVM_HAS_COX_S2V*/

// TITAN Unit
#ifdef ESVM_HAS_TITAN_UNIT
const std::string rootTitanUnitPath = getValidEnvVar("TITAN_UNIT_ROOT");                        // TITAN Unit dataset root
const std::string roiTitanUnitResultTrackPath = rootTitanUnitPath + "Results/";                 // Result's path on TITAN Unit of various algorithms
const std::string roiTitanUnitFastDTTrackPath = roiTitanUnitResultTrackPath +                   // Path of person track ROIs found with:
                                                "FAST-DT-compressive-3cascades/";               //   FAST-DT + CompressiveTracking + 3 HaarCascades
const std::string roiTitanUnitEnrollStillPath = rootTitanUnitPath + "Enroll Stills/";           // Path of enroll still images for TITAN Unit
#endif/*ESVM_HAS_TITAN_UNIT*/

#endif/*ESVM_HAS_TESTS*/
