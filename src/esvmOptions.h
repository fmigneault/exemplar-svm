#ifndef ESVM_OPTIONS_H
#define ESVM_OPTIONS_H

#include <string>

/* ESVM options */
#define ESVM_USE_HOG 1
#define ESVM_USE_LBP 0
#define ESVM_USE_PREDICT_PROBABILITY 0
#define ESVM_POSITIVE_CLASS +1
#define ESVM_NEGATIVE_CLASS -1
/*
    ESVM_WEIGHTS_MODE:
        0: (Wp = 0, Wn = 0)         unused
        1: (Wp = 1, Wn = 0.01)      enforced values
        2: (Wp = 100, Wn = 1)       enforced values
        3: (Wp = N/Np, Wn = N/Nn)   ratio of sample counts
        4: (Wp = 1, Wn = Np/Nn)     ratio of sample counts normalized for positives (Np/Nn = [N/Nn]/[N/Np])
*/
#define ESVM_WEIGHTS_MODE 2
#define ESVM_WRITE_DATA_FILES 1
/*
    ESVM_READ_DATA_FILES:
        (0) 0b0000:     no test 
        (1) 0b0001:     images + extract features (whole-image) 
        (2) 0b0010:     images + extract features (patch-based)
        (4) 0b0100:     pre-generated samples files (whole-image)
        (8) 0b1000:     pre-generated samples files (feature+patch-based)

         * (1) XOR (2), (2) remains if both specified
*/
#define ESVM_READ_DATA_FILES 0 //0b1010
const std::string dataFilePath = "data_TITAN_48x48_HOG-descriptor+9-patches"; // "data_48x48_HOG-LBP-descriptors+9-patches_fusion-patches1st-descriptors2nd/";

/* Test options - Enable/Disable a specific test execution */
#define TEST_IMAGE_PATHS 1
#define TEST_IMAGE_PROCESSING 0
#define TEST_MULTI_LEVEL_VECTORS 1
#define TEST_NORMALIZATION 1
#define TEST_ESVM_BASIC_FUNCTIONALITY 0
#define TEST_ESVM_BASIC_STILL2VIDEO 0
#define TEST_ESVM_TITAN 1
#define TEST_ESVM_SAMAN 0

/* Image paths */
// Local
const std::string roiVideoImagesPath = "../img/roi/";                               // Person ROI tracks obtained from face detection + tracking
const std::string refStillImagesPath = "../img/ref/";                               // Reference high quality still ROIs for enrollment in SSPP
// ChokePoint
const std::string rootChokePointPath = std::string(std::getenv("CHOKEPOINT_ROOT")) + "/";       // ChokePoint dataset root
const std::string roiChokePointCroppedFacePath = rootChokePointPath + "Cropped face images/";   // Path of extracted 96x96 ROI from all videos 
const std::string roiChokePointFastDTTrackPath = rootChokePointPath + "Results/fast-dt/";       // Path of person track ROIs found with FAST-DT
const std::string roiChokePointEnrollStillPath = rootChokePointPath + "Enroll Images/";         // Path of enroll still images for ChokePoint
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
