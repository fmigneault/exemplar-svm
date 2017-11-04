#ifndef ESVM_PATHS_H
#define ESVM_PATHS_H

#include "esvmOptions.h"

/* -------------------------------------------
   OpenCV Cascade files for preprocessing
------------------------------------------- */

// OpenCV source required by tests and general preprocessing using LBP Cascade (1)
#if defined(ESVM_HAS_TESTS) || ESVM_ROI_PREPROCESS_MODE == 1
static const std::string sourcesOpenCV;                 // = $OPENCV_SOURCES (environment variable)     OpenCV's root directory (ie: Git level)
#endif

/* -------------------------------------------
   Dataset image paths for testing purposes
------------------------------------------- */

#ifdef ESVM_HAS_TESTS

#ifdef ESVM_ROOT_PATH
static const std::string refPath = ESVM_ROOT_PATH;
#else
static const std::string refPath = "..";
#endif

// ESVM build/test
static const std::string roiVideoImagesPath;            // = refPath + "/img/roi/"          Person ROI tracks obtained from face detection + tracking
static const std::string refStillImagesPath;            // = refPath + "/img/ref/"          Reference high quality still ROIs for enrollment in SSPP
static const std::string negativeSamplesDir;            // = refPath + "/data/negatives/"   Pre-generated ChokePoint negative samples files
static const std::string testingSamplesDir ;            // = refPath + "/data/testing/"     Pre-generated ChokePoint probe samples files

// ChokePoint
#ifdef ESVM_HAS_CHOKEPOINT
static const std::string rootChokePointPath;            // = $CHOKEPOINT_ROOT (environment variable)    ChokePoint dataset root
static const std::string roiChokePointCroppedFacePath;  // = rootChokePointPath + "cropped_faces/"      Path of extracted 96x96 ROI from all videos
static const std::string roiChokePointFastDTTrackPath;  // = rootChokePointPath + "results/fast-dt/"    Path of person track ROIs found with FAST-DT
static const std::string roiChokePointEnrollStillPath;  // = rootChokePointPath + "enroll/"             Path of enroll still images for ChokePoint
#endif/*ESVM_HAS_CHOKEPOINT*/

// COX-S2V
#ifdef ESVM_HAS_COX_S2V
static const std::string rootCOXS2VPath;            // = $COX_S2V_ROOT (environment variable)           COX-S2V dataset root
static const std::string roiCOXS2VTestingVideoPath; // = rootCOXS2VPath + "COX-S2V-Video/"              Path of enroll still images for COX-S2V
static const std::string roiCOXS2VAllImgsStillPath; // = rootCOXS2VPath + "COX-S2V-Still/"              Path of every individual's still for COX-S2V
static const std::string roiCOXS2VEnrollStillsPath; // = rootCOXS2VPath + "Persons-for-Publication/"    Path of pre-selected gallery stills
static const std::string roiCOXS2VEyeLocaltionPath; // = rootCOXS2VPath + "Eye_location/"               Path of eye location ground truths
#endif/*ESVM_HAS_COX_S2V*/

// TITAN Unit
#ifdef ESVM_HAS_TITAN_UNIT
static const std::string rootTitanUnitPath;             // = $CHOKEPOINT_ROOT (environment variable)    TITAN Unit dataset root
static const std::string roiTitanUnitResultTrackPath;   // = rootTitanUnitPath + "Results/"             Result's path on TITAN Unit various algorithms
static const std::string roiTitanUnitFastDTTrackPath;   // = roiTitanUnitResultTrackPath + "..."        Path of track ROI with multi-tracker+3cascades
static const std::string roiTitanUnitEnrollStillPath;   // = rootTitanUnitPath + "Enroll Stills/"       Path of enroll still images for TITAN Unit
#endif/*ESVM_HAS_TITAN_UNIT*/

#endif/*ESVM_HAS_TESTS*/
#endif/*ESVM_PATHS_H*/
