#ifndef ESVM_OPTIONS_H
#define ESVM_OPTIONS_H

#include <string>
#include <stdlib.h>

/* ------------------------------------------------------------
   ESVM options
------------------------------------------------------------ */

#define ESVM_USE_HOG 1
#define ESVM_USE_LBP 0
#define ESVM_USE_HISTOGRAM_EQUALIZATION 0   //1
#define ESVM_USE_PREDICT_PROBABILITY 0
#define ESVM_POSITIVE_CLASS +1
#define ESVM_NEGATIVE_CLASS -1
#define ESVM_BINARY_HEADER_MODEL "ESVM bin model"
#define ESVM_BINARY_HEADER_SAMPLES "ESVM bin samples"
// Ratio to employ when running 'ESVM_ROI_PREPROCESS_MODE == 2'
#define ESVM_ROI_CROP_RATIO 0.80
/* Employ specific ROI pre-processing operation before further feature extraction operations
    
    ESVM_ROI_PREPROCESS_MODE:
        0: 'normal' procedure without additional ROI pre-processing (using ChokePoint 'cropped_faces')
        1: apply localized face ROI refinement within 'cropped_faces' using LBP improved CascadeClassifier
        2: apply specific pre-cropping of 'cropped_faces' ROI with ROI ratio defined by 'ESVM_ROI_CROP_RATIO'
*/
#define ESVM_ROI_PREPROCESS_MODE 0  //2
/*
    ESVM_WEIGHTS_MODE:
        0: (Wp = 0, Wn = 0)         unused
        1: (Wp = 1, Wn = 0.01)      enforced values
        2: (Wp = 100, Wn = 1)       enforced values
        3: (Wp = N/Np, Wn = N/Nn)   ratio of sample counts
        4: (Wp = 1, Wn = Np/Nn)     ratio of sample counts normalized for positives (Np/Nn = [N/Nn]/[N/Np])
*/
#define ESVM_WEIGHTS_MODE 2
/*
    ESVM_FEATURE_NORMALIZATION_MODE:
        0: no normalization
        1: normalization min-max overall, across patches
        2: normalization z-score overall, across patches
        3: normalization min-max per feature, across patches
        4: normalization z-score per feature, across patches
        5: normalization min-max overall, for each patch
        6: normalization z-score overall, for each patch
        7: normalization min-max per feature, for each patch
        8: normalization z-score per feature, for each patch
*/
#define ESVM_FEATURE_NORMALIZATION_MODE 1   //7
// Specify if normalized features need to be clipped if outside of [0,1]
#define ESVM_FEATURE_NORMALIZATION_CLIP 1
/*
    ESVM_SCORE_NORMALIZATION_MODE:
        0: no normalization
        1: normalization min-max
        2: normalization z-score
*/
#define ESVM_SCORE_NORMALIZATION_MODE 1
// Specify if normalized scores need to be clipped if outside of [0,1]
#define ESVM_SCORE_NORMALIZATION_CLIP 0
/*
    ESVM_PARSER_MODE:
        0: stringstream
        1: std strtol/strtod
        2: simple parser (faster strtod)
*/
#define ESVM_READ_LIBSVM_PARSER_MODE 1

/* ------------------------------------------------------------
   Test options - Enable/Disable a specific test execution
------------------------------------------------------------ */

/*
  Specify how the training samples are regrouped into training sequences
    
    TEST_CHOKEPOINT_SEQUENCES_MODE:    
        0: use all cameras in a corresponding session as a common list of training samples (ie: 4 session = 4 sequences)
        1: use each scene as an independant list of training samples (ie: 2 portals x 2 types x 4 sessions x 3 cameras = 48 sequences) 
*/
#define TEST_CHOKEPOINT_SEQUENCES_MODE 0
// Employ synthetic image generation to increase the positive samples quantity for ESVM training
#define TEST_USE_SYNTHETIC_GENERATION 0
// Employ duplication of samples by specified amount to increase positive samples quantity in ESVM training
#define TEST_DUPLICATE_COUNT 0
// Employ gallery positives samples other than the currently trained one as additional counter-examples in ESVM training
#define TEST_USE_OTHER_POSITIVES_AS_NEGATIVES 0
/*
    TEST_FEATURES_NORMALIZATION_MODE:
        0: no normalization applied
        1: normalization per corresponding features in all positive/negative/probe vectors, and separately for each patch and descriptor
        2: normalization per corresponding features in all positive/negative/probe vectors, and across all patches, separately for each descriptor
        3: normalization across all features and all positive/negative/probe vectors, and across all patches, seperately for each descriptor
*/
#define TEST_FEATURES_NORMALIZATION_MODE 3
// Validates image paths found and with expected format
#define TEST_PATHS 1
// Test functionality of patch extraction procedures
#define TEST_IMAGE_PATCH_EXTRACTION 1
// Test and display results of regular image preprocessing chain for reference still
#define TEST_IMAGE_PREPROCESSING 0
// Test functionality of 'mvector' generation, dimensions and general behaviour
#define TEST_MULTI_LEVEL_VECTORS 1
// Test functionality of all vector normalization functions
#define TEST_NORMALIZATION 1
// Test functionality of performance evaluation functions
#define TEST_PERF_EVAL_FUNCTIONS 1
// Test MATLAB code procedure (obsolete)
#define TEST_ESVM_BASIC_FUNCTIONALITY 0
// Test classification results with simple XOR data
#define TEST_ESVM_BASIC_CLASSIFICATION 0
// Evaluate timing performance for writing/reading and parsing LIBSVM/BINARY samples file
#define TEST_ESVM_WRITE_SAMPLES_FILE_TIMING 0
#define TEST_ESVM_READ_SAMPLES_FILE_TIMING 0
// Test functionality of BINARY/LIBSVM samples file reading and parsing to feature vectors
#define TEST_ESVM_READ_SAMPLES_FILE_PARSER_BINARY 0
#define TEST_ESVM_READ_SAMPLES_FILE_PARSER_LIBSVM 0
// Test functionality of samples file reading LIBSVM/BINARY format comparison
#define TEST_ESVM_READ_SAMPLES_FILE_FORMAT_COMPARE 0
// Test functionality of BINARY/LIBSVM model file loading/saving and parsing of parameters allowing valid use afterwards
#define TEST_ESVM_SAVE_LOAD_MODEL_FILE_PARSER_BINARY 0
#define TEST_ESVM_SAVE_LOAD_MODEL_FILE_PARSER_LIBSVM 0
// Test functionality of model file loading/saving from (LIBSVM/BINARY, pre-trained/from samples) format comparison
#define TEST_ESVM_SAVE_LOAD_MODEL_FILE_FORMAT_COMPARE 0
// Test model resetting using 'svm_model' struct directly populated in code, validate parameter checks
#define TEST_ESVM_MODEL_STRUCT_SVM_PARAMS 0
// Test memory deallocation of various model parameters on reset or destructor calls
#define TEST_ESVM_MODEL_MEMORY_OPERATIONS 0
// Test expected functionalities of model with reset/changed parameters (model properly updated)
#define TEST_ESVM_MODEL_MEMORY_PARAM_CHECK 0

/* -------------------------------------------------------------------
    Process options - Enable/Disable a specific procedure execution
------------------------------------------------------------------- */

/*   
    PROC_READ_DATA_FILES:
        (0)   0b00000000:   no test 
        (1)   0b00000001:   images + extract features (whole-image) 
        (2)   0b00000010:   images + extract features (patch-based)
        (4)   0b00000100:   pre-generated samples files (whole-image)
        (8)   0b00001000:   pre-generated samples files (feature+patch-based)
        (16)  0b00010000:   pre-generated negative samples files + extract features for probe images (patch-based, normal images - MATLAB HOG)
        (32)  0b00100000:   pre-generated negative samples files + pre-generated probe samples files (patch-based, normal images - MATLAB HOG)
        (64)  0b01000000:   pre-generated negative samples files + pre-generated probe samples files (patch-based, transposed images - MATLAB HOG)
        (128) 0b10000000:   pre-generated negative samples files + extract features for probe images (patch-based, normal images - C++ HOG)

         * [(1) XOR (2)]: (2) has priority over (1)
         * [(32) XOR (64)]: (32) has priority over (64)
         * (16) can be combined with [(32) XOR (64)] to run images/files sequentially, normal/transposed images files
         * (128) cannot be set with any of [(16),(32),(64)]
         * any other combination of flags is allowed (different test functions)
*/
#define PROC_READ_DATA_FILES 0b00000000
// Outputs extracted feature vectors from loaded images to samples files
#define PROC_WRITE_DATA_FILES 1
// Test alternative MATLAB procedure (obsolete)
#define PROC_ESVM_BASIC_STILL2VIDEO 0
// Test training and testing using TITAN reference images against ChokePoint negatives
#define PROC_ESVM_TITAN 0
/*
    PROC_ESVM_SAMAN:
        0: not run
        1: run with PCA feature vectors
        2: run with raw feature vectors
        3: run with raw feature vectors obtained from pre-transposed images
        4: run with raw feature vectors obtained from 'FullChokePoint' test (pre-feature norm overall, post-fusion norm)
*/
#define PROC_ESVM_SAMAN 0
/*
    PROC_ESVM_SIMPLIFIED_WORKING:
        0: not run
        1: run with LIBSVM formatted sample files
        2: run with BINARY formatted sample files
*/
#define PROC_ESVM_SIMPLIFIED_WORKING 2
// Generate some differnt image types for convenience
#define PROC_ESVM_GENERATE_CONVERTED_IMAGES 0
// Generate sample files using various enabled parameters
#define PROC_ESVM_GENERATE_SAMPLE_FILES 0
// Request binary format sample file generation
#define PROC_ESVM_GENERATE_SAMPLE_FILES_BINARY 1
// Request libsvm format sample file generation
#define PROC_ESVM_GENERATE_SAMPLE_FILES_LIBSVM 0
// Generate from scratch multiple training/testing samples and evaluate them
#define PROC_ESVM_FULL_GENERATION_TESTING 1

/* ------------------------------------------------------------
   Image paths
------------------------------------------------------------ */

// Local
const std::string roiVideoImagesPath = "../img/roi/";                               // Person ROI tracks obtained from face detection + tracking
const std::string refStillImagesPath = "../img/ref/";                               // Reference high quality still ROIs for enrollment in SSPP
const std::string negativeSamplesDir = "../data/negatives/";                        // Pre-generated ChokePoint negative samples files
const std::string testingSamplesDir = "../data/testing/";                           // Pre-generated ChokePoint probe samples files
// OpenCV
const std::string sourcesOpenCV = std::string(std::getenv("OPENCV_SOURCES")) + "/";             // OpenCV's root directory (ie: Git level)
// ChokePoint
const std::string rootChokePointPath = std::string(std::getenv("CHOKEPOINT_ROOT")) + "/";       // ChokePoint dataset root
const std::string roiChokePointCroppedFacePath = rootChokePointPath + "cropped_faces/";         // Path of extracted 96x96 ROI from all videos 
const std::string roiChokePointFastDTTrackPath = rootChokePointPath + "results/fast-dt/";       // Path of person track ROIs found with FAST-DT
const std::string roiChokePointEnrollStillPath = rootChokePointPath + "enroll/";                // Path of enroll still images for ChokePoint
// TITAN Unit
const std::string rootTitanUnitPath = std::string(std::getenv("TITAN_UNIT_ROOT")) + "/";        // TITAN Unit dataset root 
const std::string roiTitanUnitResultTrackPath = rootTitanUnitPath + "Results/";                 // Result's path on TITAN Unit of various algorithms
const std::string roiTitanUnitFastDTTrackPath = roiTitanUnitResultTrackPath +                   // Path of person track ROIs found with:
                                                "FAST-DT-compressive-3cascades/";               //   FAST-DT + CompressiveTracking + 3 HaarCascades
const std::string roiTitanUnitEnrollStillPath = rootTitanUnitPath + "Enroll Stills/";           // Path of enroll still images for TITAN Unit
// COX-S2V
const std::string rootCOXS2VPath = std::string(std::getenv("COX_S2V_ROOT")) + "/";              // COX-S2V dataset root
const std::string roiCOXS2VTestingVideoPath = rootCOXS2VPath + "COX-S2V-Video/";                // Path of enroll still images for COX-S2V
const std::string roiCOXS2VAllImgsStillPath = rootCOXS2VPath + "COX-S2V-Still/";                // Path of every individual's still for COX-S2V
const std::string roiCOXS2VEnrollStillsPath = rootCOXS2VPath + "Persons-for-Publication/";      // Path of pre-selected gallery stills
const std::string roiCOXS2VEyeLocaltionPath = rootCOXS2VPath + "Eye_location/";                 // Path of eye location ground truths

#endif/*ESVM_OPTIONS_H*/
