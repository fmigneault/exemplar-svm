#ifndef ESVM_PATHS_H
#define ESVM_PATHS_H

#include "esvmOptions.h"
#include <string>

#define conststr extern const std::string

namespace esvm {
namespace path {

/* -------------------------------------------
   OpenCV Cascade files for preprocessing
------------------------------------------- */

// OpenCV source required by tests and general preprocessing using LBP Cascade (1)
#if defined(ESVM_HAS_TESTS) || ESVM_ROI_PREPROCESS_MODE == 1
conststr sourcesOpenCV;                 // = $OPENCV_SOURCES (environment variable)     OpenCV's root directory (ie: Git level)
#endif

/* -------------------------------------------
   Dataset image paths for testing purposes
------------------------------------------- */

#ifdef ESVM_HAS_TESTS

conststr refPath;

// ESVM build/test paths
conststr roiVideoImagesPath;            // = refPath + "/img/roi/"                          Person ROI tracks obtained from face detection + tracking
conststr refStillImagesPath;            // = refPath + "/img/ref/"                          Reference high quality still ROIs for enrollment in SSPP
conststr negativeSamplesDir;            // = refPath + "/data/negatives/"                   Pre-generated ChokePoint negative samples files
conststr testingSamplesDir ;            // = refPath + "/data/testing/"                     Pre-generated ChokePoint probe samples files

// ChokePoint
#ifdef ESVM_HAS_CHOKEPOINT
conststr rootChokePointPath;            // = $CHOKEPOINT_ROOT (environment variable)        ChokePoint dataset root
conststr roiChokePointCroppedFacePath;  // = rootChokePointPath + "cropped_faces/"          Path of extracted 96x96 ROI from all videos
conststr roiChokePointFastDTTrackPath;  // = rootChokePointPath + "results/fast-dt/"        Path of person track ROIs found with FAST-DT
conststr roiChokePointEnrollStillPath;  // = rootChokePointPath + "enroll/"                 Path of enroll still images for ChokePoint
#endif/*ESVM_HAS_CHOKEPOINT*/

// COX-S2V
#ifdef ESVM_HAS_COX_S2V
conststr rootCOXS2VPath;                // = $COX_S2V_ROOT (environment variable)           COX-S2V dataset root
conststr roiCOXS2VTestingVideoPath;     // = rootCOXS2VPath + "COX-S2V-Video/"              Path of enroll still images for COX-S2V
conststr roiCOXS2VAllImgsStillPath;     // = rootCOXS2VPath + "COX-S2V-Still/"              Path of every individual's still for COX-S2V
conststr roiCOXS2VEnrollStillsPath;     // = rootCOXS2VPath + "Persons-for-Publication/"    Path of pre-selected gallery stills
conststr roiCOXS2VEyeLocaltionPath;     // = rootCOXS2VPath + "Eye_location/"               Path of eye location ground truths
#endif/*ESVM_HAS_COX_S2V*/

// TITAN Unit
#ifdef ESVM_HAS_TITAN_UNIT
conststr rootTitanUnitPath;             // = $CHOKEPOINT_ROOT (environment variable)        TITAN Unit dataset root
conststr roiTitanUnitResultTrackPath;   // = rootTitanUnitPath + "Results/"                 Result's path on TITAN Unit various algorithms
conststr roiTitanUnitFastDTTrackPath;   // = roiTitanUnitResultTrackPath + "..."            Path of track ROI with multi-tracker+3cascades
conststr roiTitanUnitEnrollStillPath;   // = rootTitanUnitPath + "Enroll Stills/"           Path of enroll still images for TITAN Unit
#endif/*ESVM_HAS_TITAN_UNIT*/

} // namespace path
} // namespace esvm

#endif/*ESVM_HAS_TESTS*/
#endif/*ESVM_PATHS_H*/
