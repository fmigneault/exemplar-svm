#ifndef ESVM_OPTIONS_H
#define ESVM_OPTIONS_H

#include <string>
#include <stdlib.h>

/* ------------------------------------------------------------
   ESVM options
------------------------------------------------------------ */

#define ESVM_USE_HOG 1
#define ESVM_USE_LBP 0
#define ESVM_USE_HIST_EQUAL 1
#define ESVM_USE_PREDICT_PROBABILITY 0
#define ESVM_POSITIVE_CLASS +1
#define ESVM_NEGATIVE_CLASS -1
#define ESVM_BINARY_HEADER_MODEL_LIBSVM "ESVM binary model libsvm"
#define ESVM_BINARY_HEADER_MODEL_LIBLINEAR "ESVM binary model liblinear"
#define ESVM_BINARY_HEADER_SAMPLES "ESVM binary samples"
// Debug-specific configurations
#ifndef NDEBUG
#define ESVM_DEBUG
#endif
/*
    ESVM_DISPLAY_TRAIN_PARAMS:
        0: do not display obtained parameters after training
        1: display abridged parameters after training (no SV)
        2: display complete parameters after training (all available according to model status, including SV)

    * Note: display after training is executed only in 'debug' mode, in 'release' it is disabled (ie: 0) to allow parallelism
*/
#define ESVM_DISPLAY_TRAIN_PARAMS 1
/* Employ specific ROI pre-processing operation before further feature extraction operations

    ESVM_ROI_PREPROCESS_MODE:
        0: 'normal' procedure without additional ROI pre-processing (using ChokePoint 'cropped_faces')
        1: apply localized face ROI refinement within 'cropped_faces' using LBP improved CascadeClassifier
        2: apply specific pre-cropping of 'cropped_faces' ROI with ROI ratio defined by 'ESVM_ROI_CROP_RATIO'
*/
#define ESVM_ROI_PREPROCESS_MODE 2
// Ratio to employ when running 'ESVM_ROI_PREPROCESS_MODE == 2'
#define ESVM_ROI_CROP_RATIO 0.80
/*
    ESVM_WEIGHTS_MODE:
        0: (Wp = 0, Wn = 0)         unused
        1: (Wp = 1, Wn = 0.01)      enforced values
        2: (Wp = 100, Wn = 1)       enforced values
        3: (Wp = N/Np, Wn = N/Nn)   ratio of sample counts
        4: (Wp = 1, Wn = Np/Nn)     ratio of sample counts normalized for positives (Np/Nn = [N/Nn]/[N/Np])
*/
#define ESVM_WEIGHTS_MODE 2
/* Specify if random subspace method (RSM) for feature selection must be employed to generate the ensemble of eSVM
        0: RSM is not employed (directly using the basic feature extraction methods)
        #: other numeric int value, the specified value is the amount of RS operations applied
*/
#define ESVM_RANDOM_SUBSPACE_METHOD 20
// Specifies the amount of features to be randomly selected when applying RSM
#define ESVM_RANDOM_SUBSPACE_FEATURES 128
/*
    ESVM_FEATURE_NORM_MODE:
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
#define ESVM_FEATURE_NORM_MODE 7
// Specify if normalized features need to be clipped if outside of [0,1]
#define ESVM_FEATURE_NORM_CLIP 1
/*
    ESVM_SCORE_NORM_MODE:
        0: no normalization
        1: normalization min-max after score fusion
        2: normalization z-score after score fusion
        3: normalization min-max before score fusion (on patches/subspaces)
        4: normalization z-score before score fusion (on patches/subspaces)
        5: normalization min-max before and after score fusion
        6: normalization z-score before and after score fusion
*/
#define ESVM_SCORE_NORM_MODE 1
// Specify if normalized scores need to be clipped if outside of [0,1]
#define ESVM_SCORE_NORM_CLIP 0
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
    TEST_FEATURES_NORM_MODE:
        0: no normalization applied
        1: normalization per corresponding features in all positive/negative/probe vectors, and separately for each patch and descriptor
        2: normalization per corresponding features in all positive/negative/probe vectors, and across all patches, separately for each descriptor
        3: normalization across all features and all positive/negative/probe vectors, and across all patches, seperately for each descriptor
*/
#define TEST_FEATURES_NORM_MODE 3
// Validates image paths found and with expected format
#define TEST_PATHS 0
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
// Test ESVM training and testing especially in the case where Random Subspace Method (RSM) are enabled
#define TEST_ESVM_BASIC_TRAIN_TEST_RSM 1
// Evaluate timing performance for writing/reading and parsing LIBSVM/BINARY samples file
#define TEST_ESVM_WRITE_SAMPLES_FILE_TIMING 0
#define TEST_ESVM_READ_SAMPLES_FILE_TIMING 0
// Test functionality of BINARY/LIBSVM samples file reading and parsing to feature vectors
#define TEST_ESVM_READ_SAMPLES_FILE_PARSER_BINARY 1
#define TEST_ESVM_READ_SAMPLES_FILE_PARSER_LIBSVM 0
// Test functionality of samples file reading LIBSVM/BINARY format comparison
#define TEST_ESVM_READ_SAMPLES_FILE_FORMAT_COMPARE 0
// Test functionality of BINARY/LIBSVM model file loading/saving and parsing of parameters allowing valid use afterwards
#define TEST_ESVM_SAVE_LOAD_MODEL_FILE_PARSER_BINARY 0
#define TEST_ESVM_SAVE_LOAD_MODEL_FILE_PARSER_LIBSVM 0
// Test functionality of model file loading/saving from (LIBSVM/BINARY, pre-trained/from samples) format comparison
#define TEST_ESVM_SAVE_LOAD_MODEL_FILE_FORMAT_COMPARE 0
// Test model resetting using 'svm_model' struct directly populated in code, validate parameter checks
#define TEST_ESVM_MODEL_STRUCT_SVM_PARAMS 1
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
#define PROC_ESVM_SIMPLIFIED_WORKING 0
/*
    Generate from scratch multiple training/testing samples and evaluate them

    PROC_ESVM_FULL_GENERATION_TESTING:
        0: not run
        1: run 3 different and independent sessions (S1, S3, FastDT)
           apply specified normalization values from loaded and generated samples, display newly found values for each session
           run using 'proc_runSingleSamplePerPersonStillToVideo_FullGenerationAndTestProcess'
        2: run over multiple test replications and obtain average scores with specified configurations
           run using 'proc_runSingleSamplePerPersonStillToVideo_FullGeneration_ReplicationTests'
*/
#define PROC_ESVM_FULL_GENERATION_TESTING 0
// Generate some differnt image types for convenience
#define PROC_ESVM_GENERATE_CONVERTED_IMAGES 0
// Generate sample files using various enabled parameters
#define PROC_ESVM_GENERATE_SAMPLE_FILES 0
// Request binary format sample file generation
#define PROC_ESVM_GENERATE_SAMPLE_FILES_BINARY 1
// Request libsvm format sample file generation
#define PROC_ESVM_GENERATE_SAMPLE_FILES_LIBSVM 0
/*
    Specify which sequence to generate files for using the 'proc_createNegativesSampleFiles' enabled by 'PROC_ESVM_GENERATE_SAMPLE_FILES'

    PROC_ESVM_GENERATE_SAMPLE_FILES_SESSION:
        0:      generate negative sample files that employ negative individuals from all 4 sessions of ChokePoint (all available negative images)
        [1-4]:  generate negative sample files that employ negative individuals only from the specified session number of ChokePoint
*/
#define PROC_ESVM_GENERATE_SAMPLE_FILES_SESSION 0
// Specify which replication individuals to employ for negative sample files generation
// see corresponding ID values in 'getReplicationNegativeIDs' in 'createSampleFiles'
#define PROC_ESVM_GENERATE_SAMPLE_FILES_REPLICATION 0
#define PROC_ESVM_GENERATE_SAMPLE_FILES_NEGATIVE_COUNT 10

#endif/*ESVM_OPTIONS_H*/
