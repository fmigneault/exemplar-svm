#include "esvmTests.h"
#include "esvmOptions.h"
#include "esvm.h"
#include "feHOG.h"
#include "feLBP.h"
#include "norm.h"
#include "eval.h"
#include "logger.h"
#include "imgUtils.h"
#include "generic.h"

#include <iomanip>

#include "boost/filesystem.hpp"
namespace bfs = boost::filesystem;

// Builds the string as "P#T_S#_C#", if the individual is non-zero, it adds the sub folder as "P#T_S#_C#/ID#"
std::string buildChokePointSequenceString(int portal, PORTAL_TYPE type, int session, int camera, int id)
{
    std::string dir = "P" + std::to_string(portal) + (type == ENTER ? "E" : type == LEAVE ? "L" : "") +
                      "_S" + std::to_string(session) + "_C" + std::to_string(camera);
    return id > 0 ? dir + "/" + buildChokePointIndividualID(id) : dir;
}

std::string buildChokePointIndividualID(int id, bool withPrefixID)
{
    return (withPrefixID ? "ID" : "") + std::string(id > 9 ? 2 : 3, '0').append(std::to_string(id));
}

bool checkPathEndSlash(std::string path)
{
    char end = *path.rbegin();
    return end == '/' || end == '\\';
}

int test_outputOptions()
{
    logstream logger(LOGGER_FILE);
    std::string tab = "   ";
    logger << "Options:" << std::endl
           << tab << "ESVM:" << std::endl
           << tab << tab << "ESVM_USE_HOG:                     " << ESVM_USE_HOG << std::endl
           << tab << tab << "ESVM_USE_LBP:                     " << ESVM_USE_LBP << std::endl
           << tab << tab << "ESVM_USE_PREDICT_PROBABILITY:     " << ESVM_USE_PREDICT_PROBABILITY << std::endl
           << tab << tab << "ESVM_POSITIVE_CLASS:              " << ESVM_POSITIVE_CLASS << std::endl
           << tab << tab << "ESVM_NEGATIVE_CLASS:              " << ESVM_NEGATIVE_CLASS << std::endl
           << tab << tab << "ESVM_WEIGHTS_MODE:                " << ESVM_WEIGHTS_MODE << std::endl
           << tab << "TEST:" << std::endl
           << tab << tab << "TEST_CHOKEPOINT_SEQUENCES_MODE:   " << TEST_CHOKEPOINT_SEQUENCES_MODE << std::endl
           << tab << tab << "TEST_USE_SYNTHETIC_GENERATION:    " << TEST_USE_SYNTHETIC_GENERATION << std::endl
           << tab << tab << "TEST_DUPLICATE_COUNT:             " << TEST_DUPLICATE_COUNT << std::endl
           << tab << tab << "TEST_FEATURES_NORMALIZATION_MODE: " << TEST_FEATURES_NORMALIZATION_MODE << std::endl
           << tab << tab << "TEST_IMAGE_PATHS:                 " << TEST_IMAGE_PATHS << std::endl
           << tab << tab << "TEST_IMAGE_PATCH_EXTRACTION:      " << TEST_IMAGE_PATCH_EXTRACTION << std::endl
           << tab << tab << "TEST_IMAGE_PREPROCESSING:         " << TEST_IMAGE_PREPROCESSING << std::endl
           << tab << tab << "TEST_MULTI_LEVEL_VECTORS:         " << TEST_MULTI_LEVEL_VECTORS << std::endl
           << tab << tab << "TEST_NORMALIZATION:               " << TEST_NORMALIZATION << std::endl
           << tab << tab << "TEST_ESVM_BASIC_FUNCTIONALITY:    " << TEST_ESVM_BASIC_FUNCTIONALITY << std::endl
           << tab << tab << "TEST_ESVM_BASIC_STILL2VIDEO:      " << TEST_ESVM_BASIC_STILL2VIDEO << std::endl
           << tab << tab << "TEST_WRITE_DATA_FILES:            " << TEST_WRITE_DATA_FILES << std::endl
           << tab << tab << "TEST_READ_DATA_FILES:             " << TEST_READ_DATA_FILES << std::endl
           << tab << tab << "TEST_ESVM_TITAN:                  " << TEST_ESVM_TITAN << std::endl
           << tab << tab << "TEST_ESVM_SAMAN:                  " << TEST_ESVM_SAMAN << std::endl;

    return 0;
}

int test_imagePaths()
{    
    // Local
    ASSERT_LOG(bfs::is_directory(roiVideoImagesPath), "Cannot find ROI directory");
    ASSERT_LOG(bfs::is_directory(refStillImagesPath), "Cannot find REF directory");
    ASSERT_LOG(bfs::is_directory(negativeSamplesDir), "Cannot find negative samples directory");
    ASSERT_LOG(bfs::is_directory(testingSamplesDir),  "Cannot find testing probe samples directory");
    ASSERT_LOG(checkPathEndSlash(roiVideoImagesPath), "ROI directory doesn't end with slash character");
    ASSERT_LOG(checkPathEndSlash(refStillImagesPath), "REF directory doesn't end with slash character");
    ASSERT_LOG(checkPathEndSlash(negativeSamplesDir), "Negative samples directory doesn't end with slash character");
    ASSERT_LOG(checkPathEndSlash(testingSamplesDir), "Testing probe samples directory doesn't end with slash character");
    // ChokePoint
    ASSERT_LOG(bfs::is_directory(rootChokePointPath), "Cannot find ChokePoint root directory");
    ASSERT_LOG(bfs::is_directory(roiChokePointCroppedFacePath), "Cannot find ChokePoint cropped faces root directory");
    ASSERT_LOG(bfs::is_directory(roiChokePointFastDTTrackPath), "Cannot find ChokePoint FAST-DT tracks root directory");
    ASSERT_LOG(bfs::is_directory(roiChokePointEnrollStillPath), "Cannot find ChokePoint enroll stills root directory");
    ASSERT_LOG(checkPathEndSlash(rootChokePointPath), "ChokePoint root directory doesn't end with slash character");
    ASSERT_LOG(checkPathEndSlash(roiChokePointCroppedFacePath), "ChokePoint cropped faces root directory doesn't end with slash character");
    ASSERT_LOG(checkPathEndSlash(roiChokePointFastDTTrackPath), "ChokePoint FAST-DT tracks root directory doesn't end with slash character");
    ASSERT_LOG(checkPathEndSlash(roiChokePointEnrollStillPath), "ChokePoint enroll stills root directory doesn't end with slash character");
    // TITAN Unit
    ASSERT_LOG(bfs::is_directory(rootTitanUnitPath), "Cannot find TITAN Unit root directory");
    ASSERT_LOG(bfs::is_directory(roiTitanUnitResultTrackPath), "Cannot find TITAN Unit results root directory");
    ASSERT_LOG(bfs::is_directory(roiTitanUnitFastDTTrackPath), "Cannot find TITAN Unit FAST-DT tracks root directory");
    ASSERT_LOG(bfs::is_directory(roiTitanUnitEnrollStillPath), "Cannot find TITAN Unit enroll stills root directory");
    ASSERT_LOG(checkPathEndSlash(rootTitanUnitPath), "TITAN Unit root directory doesn't end with slash character");
    ASSERT_LOG(checkPathEndSlash(roiTitanUnitResultTrackPath), "TITAN Unit results root directory doesn't end with slash character");
    ASSERT_LOG(checkPathEndSlash(roiTitanUnitFastDTTrackPath), "TITAN Unit FAST-DT tracks root directory doesn't end with slash character");
    ASSERT_LOG(checkPathEndSlash(roiTitanUnitEnrollStillPath), "TITAN Unit enroll stills root directory doesn't end with slash character");
    // COX-S2V
    ASSERT_LOG(bfs::is_directory(rootCOXS2VPath), "Cannot find COX-S2V root directory");
    ASSERT_LOG(bfs::is_directory(roiCOXS2VTestingVideoPath), "Cannot find COX-S2V testing video root directory");
    ASSERT_LOG(bfs::is_directory(roiCOXS2VEnrollStillsPath), "Cannot find COX-S2V enroll stills root directory");
    ASSERT_LOG(bfs::is_directory(roiCOXS2VAllImgsStillPath), "Cannot find COX-S2V all image stills root directory");
    ASSERT_LOG(bfs::is_directory(roiCOXS2VEyeLocaltionPath), "Cannot find COX-S2V eye location root directory");
    ASSERT_LOG(checkPathEndSlash(rootCOXS2VPath), "COX-S2V root directory doesn't end with slash character");
    ASSERT_LOG(checkPathEndSlash(roiCOXS2VTestingVideoPath), "COX-S2V testing video root directory doesn't end with slash character");
    ASSERT_LOG(checkPathEndSlash(roiCOXS2VEnrollStillsPath), "COX-S2V enroll stills root directory doesn't end with slash character");
    ASSERT_LOG(checkPathEndSlash(roiCOXS2VAllImgsStillPath), "COX-S2V all image stills root directory doesn't end with slash character");
    ASSERT_LOG(checkPathEndSlash(roiCOXS2VEyeLocaltionPath), "COX-S2V eye location root directory doesn't end with slash character");
    
    return 0;
}

int test_imagePatchExtraction(void)
{
    logstream logger(LOGGER_FILE);
    logger << "Testing image patch extraction..." << std::endl;
    int rawData[24] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 };
    cv::Mat testImg(4, 6, CV_32S, rawData);                                         // 6x4 image with above data filled line by line
    std::vector<cv::Mat> testPatches = imSplitPatches(testImg, cv::Size(3, 2));     // 6 patches of 2x2    
    // check number of patches extracted
    if (testPatches.size() != 6)
    {
        logger << "Invalid number of patches extracted (count: " << testPatches.size() << ", expected: 6)" << std::endl;
        return -1;
    }
    // check patch dimensions
    for (int p = 0; p < 6; p++)
    {
        if (testPatches[p].size() != cv::Size(2, 2))
        {
            logger << "Invalid image size for patch " << p << " (size: " << testPatches[p].size() << ", expected: (2,2))" << std::endl;
            return -2;
        }
    }    

    // check pixel values of patches
    if (!cv::countNonZero(testPatches[0] != cv::Mat(2, 2, CV_32S, { 1,2,7,8 })))
    {
        logger << "Invalid data for patch 0" << std::endl << testPatches[0] << std::endl;
        return -3;
    }
    if (!cv::countNonZero(testPatches[1] != cv::Mat(2, 2, CV_32S, { 3,4,9,10 })))
    {
        logger << "Invalid data for patch 1" << std::endl << testPatches[1] << std::endl;
        return -4;
    }
    if (!cv::countNonZero(testPatches[2] != cv::Mat(2, 2, CV_32S, { 5,6,11,12 })))
    {
        logger << "Invalid data for patch 2" << std::endl << testPatches[2] << std::endl;
        return -5;
    }
    if (!cv::countNonZero(testPatches[3] != cv::Mat(2, 2, CV_32S, { 13,14,19,20 })))
    {
        logger << "Invalid data for patch 3" << std::endl << testPatches[3] << std::endl;
        return -6;
    }
    if (!cv::countNonZero(testPatches[4] != cv::Mat(2, 2, CV_32S, { 15,16,21,22 })))
    {
        logger << "Invalid data for patch 4" << std::endl << testPatches[4] << std::endl;
        return -7;
    }
    if (!cv::countNonZero(testPatches[5] != cv::Mat(2, 2, CV_32S, { 17,18,23,24 })))
    {
        logger << "Invalid data for patch 5" << std::endl << testPatches[5] << std::endl;
        return -8;
    }
    return 0;
}

int test_imagePreprocessing()
{
    std::string refImgName = "roiID0003.tif";
    std::string refImgPath = refStillImagesPath + refImgName;
    ASSERT_LOG(bfs::is_regular_file(refImgPath), "Reference image employed for preprocessing test was not found");
    
    cv::Mat refImg = cv::imread(refImgPath, cv::IMREAD_GRAYSCALE);   
    ASSERT_LOG(refImg.size() == cv::Size(96, 96), "Expected reference image shhould be of dimension 96x96");
    int refImgSide = 48;
    cv::resize(refImg, refImg, cv::Size(refImgSide, refImgSide), 0, 0, cv::INTER_CUBIC);    

    cv::Size patchCount(3, 3);
    int nPatches = patchCount.area();
    std::vector<cv::Mat> refImgPatches = imSplitPatches(refImg, patchCount);
    ASSERT_LOG(refImgPatches.size() == nPatches, "Reference image should have been split to expected number of patches");
    for (size_t p = 0; p < nPatches; p++)
        ASSERT_LOG(refImgPatches[p].size() == cv::Size(16,16), "Reference image should have been split to expected patches dimensions");

    logstream logger(LOGGER_FILE);
    std::vector<FeatureVector> hogPatches(nPatches);
    FeatureExtractorHOG hog;
    hog.initialize(refImgPatches[0].size(), cv::Size(2, 2), cv::Size(2, 2), cv::Size(2, 2), 3);
    for (size_t p = 0; p < nPatches; p++)
    {
        hogPatches[p] = hog.compute(refImgPatches[p]);
        logger << refImgName << " hog588-impl-patch" << std::to_string(p) << ": " << featuresToVectorString(hogPatches[p]) << std::endl;
    }

    // enforced double type and HOG call
    std::vector<FeatureVector> hogPatchesDbl(nPatches);
    int patchSide = refImgPatches[0].size().width;    
    int* patchSize = new int[2]{ patchSide, patchSide };
    double* patchData = new double[patchSide*patchSide];
    double* hogParams = new double[5]{ 3, 2, 2, 0, 0.2 };
    int nFeatures = hogPatches[0].size();
    double *features = new double[nFeatures];
    for (size_t p = 0; p < nPatches; p++)
    {
        for (int y = 0; y < patchSide; y++)
            for (int x = 0; x < patchSide; x++)                            
                patchData[y + x * patchSide] = refImgPatches[p].data[y + x *  patchSide];
        HOG(patchData, hogParams, patchSize, features, 1);
        hogPatchesDbl[p] = std::vector<double>(features, features + nFeatures);
        logger << refImgName << " hog588-impl-DBL-patch" << std::to_string(p) << ": " << featuresToVectorString(hogPatchesDbl[p]) << std::endl;
    }

    // enforced double v2
    std::vector<FeatureVector> hogPatchesDbl2(nPatches);
    for (size_t p = 0; p < nPatches; p++)
    {
        for (int y = 0; y < patchSide; y++)
            for (int x = 0; x < patchSide; x++)                
                patchData[x + y * patchSide] = (double)refImgPatches[p].at<uchar>(x, y);    // !!!!!!!!!!!!!!!!!!!!! VERY SIMILAR
        HOG(patchData, hogParams, patchSize, features, 1);
        hogPatchesDbl2[p] = std::vector<double>(features, features + nFeatures);
        logger << refImgName << " hog588-impl-DBL2-patch" << std::to_string(p) << ": " << featuresToVectorString(hogPatchesDbl2[p]) << std::endl;
    }

    // access (NOW PROPERLY ACCESSED DIRECTLY IN HOG_IMPL?)     ---  VALIDATED with fix of issue #2 in 'FeatureExtractorHOG' repo
    std::vector<FeatureVector> hogPatchesAccess(nPatches);
    for (size_t p = 0; p < nPatches; p++)
    {
        hogPatchesAccess[p] = hog.compute(refImgPatches[p]);
        logger << refImgName << " hog588-impl-Access-patch" << std::to_string(p) << ": " << featuresToVectorString(hogPatchesAccess[p]) << std::endl;
    }


    // employ 'text' image data without image reading            
    double *refImgRawData = new double[refImgSide*refImgSide]{
        197,196,199,165, 52, 33, 28, 25, 23, 28, 34, 43, 53, 73, 94,122,132,130,129,125,113, 96, 82, 69, 44, 45, 62, 59, 64, 60, 52, 47, 43, 38, 36, 24, 23, 16, 17, 21, 23, 23, 25, 24, 22, 22, 34,141,
        196,196,196,103, 26, 33, 29, 31, 29, 33, 39, 51, 53, 81,111,129,138,141,142,140,138,132,118, 98, 53, 62, 49, 55, 63, 71, 64, 39, 31, 37, 49, 33, 27, 26, 17, 16, 18, 22, 21, 23, 26, 28, 22, 91,
        195,198,170, 51, 26, 34, 31, 32, 31, 29, 49, 61, 53, 89,121,135,144,149,147,150,151,152,145,131, 84, 63, 52, 60, 72, 76, 80, 58, 28, 35, 49, 50, 24, 36, 23, 14, 16, 18, 19, 23, 22, 30, 27, 44,
        195,199,108, 25, 32, 36, 32, 27, 27, 32, 59, 66, 53, 98,129,142,147,149,152,155,158,158,157,149,121, 62, 63, 64, 88, 82, 95, 81, 45, 30, 44, 62, 34, 39, 35, 18, 13, 15, 16, 22, 18, 26, 28, 29,
        197,181, 55, 27, 33, 32, 29, 24, 31, 44, 64, 75, 64,109,135,144,148,150,155,158,161,163,162,156,141, 69, 59, 66, 79, 86,101, 93, 71, 29, 32, 60, 53, 40, 49, 28, 15, 13, 14, 19, 21, 20, 26, 27,
        199,127, 30, 30, 30, 27, 27, 24, 41, 61, 72, 87, 80,117,136,139,147,152,157,157,159,160,159,159,150, 97, 44, 65, 82, 83,113,110, 97, 56, 23, 50, 71, 46, 56, 40, 21, 13, 12, 15, 20, 18, 23, 26,
        189, 77, 30, 32, 27, 25, 22, 24, 51, 83, 82, 99,107,133,144,145,149,154,157,156,157,158,160,158,156,127, 49, 54, 86, 86,114,125,115, 90, 34, 38, 71, 63, 52, 52, 30, 18, 13, 12, 15, 19, 22, 27,
        172, 44, 27, 30, 26, 25, 22, 29, 65, 99,106,111,130,142,145,148,151,156,158,158,160,162,162,160,162,158,103, 62, 92, 88,114,136,130,105, 66, 34, 62, 74, 56, 59, 44, 25, 15, 12, 12, 17, 25, 25,
        139, 30, 26, 27, 27, 25, 23, 47, 94,117,121,126,134,140,146,150,153,156,159,162,161,164,165,165,165,165,153,109,113,110,120,134,137,129,104, 52, 45, 76, 65, 59, 61, 36, 20, 14, 13, 15, 23, 27,
        107, 26, 25, 27, 26, 24, 30, 77,115,122,123,127,133,140,148,152,149,154,158,158,162,165,165,163,162,162,162,144,131,129,138,142,139,134,125,102, 62, 58, 74, 62, 66, 50, 25, 15, 13, 12, 16, 26,
        118, 35, 22, 27, 22, 26, 57,104,119,120,123,129,131,137,142,144,146,149,152,151,152,152,151,151,152,157,155,149,140,134,136,139,135,130,124,118,103, 75, 71, 71, 62, 62, 35, 17, 11, 10, 12, 23,
        122, 34, 25, 26, 23, 37, 85,114,121,117,118,118,107,114,117,122,129,134,135,136,136,136,138,141,145,144,140,135,130,128,128,127,121,117,111,106,102, 94, 83, 77, 70, 62, 48, 23, 12, 10, 10, 17,
        130, 36, 25, 23, 28, 53,101,118,119,111,101, 93, 84, 82, 72, 71, 74, 89,108,119,119,121,127,134,137,135,126,119,116,115,107, 88, 78, 77, 74, 77, 82, 87, 87, 85, 78, 71, 57, 32, 15, 10, 10, 14,
        114, 31, 26, 19, 33, 76,112,119,114,101, 93, 84, 78, 68, 56, 53, 48, 52, 72, 90,101,108,118,125,128,123,113,104, 95, 82, 58, 46, 40, 44, 53, 65, 70, 71, 70, 79, 82, 77, 67, 42, 20, 11, 11, 15,
        119, 41, 24, 19, 41,100,120,117,108,102, 96, 90, 85, 80, 78, 68, 62, 65, 70, 73, 82, 98,110,115,116,109, 97, 84, 73, 60, 50, 49, 45, 47, 56, 65, 70, 70, 69, 73, 83, 81, 72, 52, 25, 13, 12, 18,
        148, 47, 24, 19, 50,114,124,117,113,107,100, 97, 99,111,118,108, 92, 78, 75, 76, 74, 83,100,108,104, 94, 79, 68, 69, 61, 59, 66, 82, 90, 78, 76, 75, 71, 67, 74, 85, 85, 81, 62, 31, 14, 13, 17,
        161, 62, 27, 19, 57,121,126,113, 98, 88, 90, 96,107,118,126,128,112, 92, 82, 80, 77, 80, 95, 97, 95, 85, 69, 66, 68, 67, 76,103,126,126,110, 95, 83, 74, 71, 73, 80, 85, 84, 71, 37, 14, 13, 16,
        181, 94, 29, 19, 60,122, 97, 65, 72, 74, 69, 60, 58, 62, 65, 74, 83, 91, 86, 77, 75, 83, 97,102, 95, 84, 71, 71, 78, 82, 89,101,102, 84, 61, 53, 53, 51, 47, 54, 75, 89, 91, 79, 41, 13, 13, 16,
        190,127, 34, 23, 58, 99, 28, 89,116, 92, 58, 41, 43, 58, 18, 17, 20, 50, 75, 74, 67, 64, 83, 88, 84, 76, 71, 74, 75, 76, 52, 27, 25, 19, 28, 29, 35, 49, 54, 48, 38, 73, 91, 81, 40, 12, 12, 16,
        192,156, 53, 40, 67, 93, 26, 98,124,113, 95, 99,105,102, 57, 39, 47, 52, 74, 94,101, 63, 23, 55, 38, 25, 65, 81, 85, 69, 44, 34, 17, 19, 53, 42, 34, 36, 53, 77, 41, 25, 89, 76, 24, 12, 12, 15,
        191,175, 90, 75, 90, 99, 38, 98,132,138,133,124,120,117,113,107,105,109,111,121,125, 56, 42,113, 82, 19, 76,114,116,102,105,103, 90, 86, 98, 94, 85, 77, 78, 90, 47, 21, 92, 83, 20, 15, 19, 20,
        190,170, 74, 68,100,121, 80, 88,132,141,141,130,114,100, 92, 91, 97,116,130,136,102, 28, 45, 79, 71, 26, 47,115,127,126,112, 99, 92, 90, 90, 95,105,115,113,108, 45, 52, 89, 83, 42, 24, 36, 40,
        191,162, 77, 57,104,120,108, 86,120,123,126,123,119,115,112,110,113,118,120,115, 72, 41, 86,149,127, 48, 32, 86,110,118,114,106, 96, 92, 95,105,116,117,115,103, 60,103, 96, 84, 52, 30, 37, 50,
        190,161, 96, 73,104,120,115,108,113,110,117,121,120,119,116,116,117,118,119,107, 69, 80,130,155,138, 82, 45, 71,103,111,111,112,110,109,111,111,112,106,108, 99, 87,103, 92, 81, 53, 27, 42, 55,
        189,164,112, 89,100,116,111,101, 96,111,123,126,128,131,132,131,131,133,130,117, 97,109,139,153,139,103, 78, 90,111,122,117,118,116,114,113,110,109,106,101, 96, 90, 83, 86, 80, 52, 31, 49, 57,
        188,170,127,103,102,112,103,100, 97, 98,104,111,121,125,127,128,126,121,111, 97,101,131,144,152,140,112, 95, 85,100,113,118,123,122,118,110,105,104, 98, 86, 78, 72, 77, 82, 78, 50, 36, 55, 63,
        185,170,135,115,103,109,104,102, 96, 91, 88, 92,100,104,104,104,101, 97, 90, 91,121,134,148,160,138,112, 99, 84, 79, 90, 97,101,101, 97, 91, 86, 85, 81, 75, 67, 65, 71, 79, 80, 50, 37, 59, 76,
        186,162,128,131,108,104,100, 97,100, 99,102,112,119,130,129,123,117,112,110,114,132,139,151,165,139,115,110, 97, 87, 94, 98, 99, 98, 96, 89, 83, 75, 70, 64, 64, 67, 70, 78, 78, 53, 39, 64, 90,
        184,158,133,135,109,102, 94, 95,103,112,119,115,119,138,140,138,137,132,125,131,137,135,142,144,132,120,119,116,101,117,128,131,131,128,123,114, 99, 82, 70, 69, 70, 70, 76, 76, 61, 64, 72, 93,
        182,161,136,129,109, 99, 98,102,107,117,121,117,119,129,131,132,131,127,119,119,108,105,111,116,111,102, 95,100, 98,106,124,128,128,123,116,109,100, 87, 79, 71, 70, 69, 75, 75, 66, 83, 78, 97,
        181,175,134,102, 97,101,100, 99,104,112,117,118,122,125,128,126,124,122,105, 74, 39, 47, 68, 75, 72, 58, 38, 49, 68, 92,114,117,118,113,110,105, 91, 82, 77, 70, 70, 70, 73, 73, 62, 80, 95,109,
        179,176,165,135,120,102, 99, 99,102,104,109,109,115,120,121,123,124,130, 94, 47, 31, 29, 40, 43, 41, 34, 26, 35, 46, 89,113,108,107,107,104, 95, 85, 82, 77, 70, 68, 71, 73, 70, 56, 71, 96,120,
        179,175,171,168,149,103, 99,105,104,103,101,106,110,113,118,119,126,138,114, 56, 49, 52, 46, 40, 40, 42, 41, 38, 49, 96,113,109,102, 98, 89, 81, 77, 80, 78, 70, 66, 70, 74, 73, 73, 85,107,120,
        178,174,170,165,155,107,100,104,103, 99,101,104,103,108,114,120,130,136,129,113,100, 90, 74, 53, 47, 49, 55, 64, 81,101,110,113,105, 91, 80, 71, 73, 69, 66, 68, 67, 70, 73, 76, 88,100,110,120,
        177,173,169,164,159,118, 97, 98,101,101,101, 99,101,104,110,112,126,131,125,123,121,112,100, 72, 64, 71, 79, 85, 94,101,108,112,107, 94, 82, 75, 74, 67, 63, 63, 66, 69, 72, 76, 88, 98,109,120,
        176,172,168,164,161,134, 96, 97, 98, 97,100,101,101,104,110,114,116,115,114,116,120,126,127,109,115,119,103, 97, 95, 93, 95,100,103, 94, 85, 81, 74, 71, 66, 65, 66, 69, 72, 80, 91,102,112,121,
        175,171,168,164,160,148,103, 96, 97, 96, 98,104,104,103,107,107, 99, 96,101,113,115,107,108,125,117, 93, 94, 95, 90, 80, 73, 81, 91, 89, 86, 87, 82, 75, 70, 67, 65, 69, 72, 83, 94,104,114,123,
        175,171,167,163,159,154,117, 93, 95, 95, 97,102,106,105,101, 90, 65, 52, 59, 62, 49, 45, 49, 58, 48, 38, 39, 42, 48, 44, 42, 62, 83, 86, 91, 93, 83, 72, 70, 66, 66, 69, 74, 86, 95,105,114,124,
        174,171,167,163,159,155,136, 93, 90, 91, 92, 97,103,102, 99, 80, 47, 35, 38, 50, 64, 64, 61, 54, 54, 64, 66, 59, 42, 39, 53, 66, 78, 82, 89, 85, 73, 67, 66, 64, 66, 69, 77, 87, 96,106,115,125,
        174,171,167,163,159,154,149,107, 85, 85, 85, 88, 91, 98, 97, 87, 95,112,102, 92,101,117,124,123,124,119, 97, 76, 77, 93,106,108, 93, 86, 82, 74, 67, 63, 60, 62, 66, 67, 78, 88, 97,107,116,125,
        173,170,167,164,159,155,151,128, 83, 77, 78, 79, 81, 89, 95,106,125,128,111, 91, 79, 82, 93, 98, 94, 81, 65, 62, 76, 92,108,111, 96, 83, 74, 65, 60, 58, 58, 62, 63, 70, 80, 89, 99,108,118,125,
        173,170,167,163,159,155,151,144, 99, 72, 70, 71, 74, 81, 92,107,120,122,108, 85, 63, 55, 57, 59, 56, 51, 51, 57, 72, 91, 99, 94, 85, 76, 66, 58, 54, 53, 58, 61, 62, 74, 83, 91,100,110,119,127,
        173,170,167,164,160,155,150,148,124, 74, 65, 65, 68, 76, 87, 98,109,115,110, 95, 73, 55, 45, 39, 38, 42, 49, 62, 78, 88, 90, 86, 81, 74, 63, 54, 51, 54, 57, 58, 66, 77, 83, 92,101,111,120,127,
        173,170,167,164,160,156,151,149,118, 73, 64, 61, 63, 73, 86, 93, 99,103,103,102, 95, 84, 72, 62, 60, 62, 72, 80, 86, 88, 84, 80, 76, 69, 58, 51, 51, 53, 54, 58, 71, 78, 84, 93,103,112,120,126,
        174,171,168,164,159,155,153,139, 65, 68, 63, 58, 57, 64, 76, 86, 93, 97,103,105,106,105,107,109,109,104,100, 96, 94, 88, 81, 77, 72, 61, 54, 50, 51, 50, 51, 61, 72, 78, 85, 94,104,113,119,126,
        174,171,167,163,160,156,154,103, 42, 66, 64, 53, 51, 54, 62, 73, 88, 99,106,108,113,115,116,115,117,114,108,104, 98, 88, 80, 75, 65, 53, 48, 47, 46, 45, 50, 43, 66, 80, 86, 95,105,113,120,125,
        174,170,167,163,159,157,142, 71, 38, 64, 69, 51, 47, 47, 49, 58, 72, 90,100,105,110,118,117,112,108,108,107,100, 92, 86, 75, 64, 52, 46, 45, 42, 41, 45, 50, 22, 47, 81, 87, 96,106,113,119,124,
        174,171,167,164,160,157,114, 59, 34, 60, 72, 56, 46, 43, 43, 44, 49, 64, 81, 89, 94,100,101, 97, 93, 92, 88, 82, 74, 67, 59, 51, 44, 40, 38, 38, 40, 46, 43, 13, 29, 71, 87, 96,106,112,118,124,
    };
    cv::Mat refImgData = cv::Mat(refImgSide, refImgSide, CV_32F, refImgRawData);
    ///refImgData.convertTo(refImgData, CV_32F, 1.0 / 256.0, 0);
    std::vector<cv::Mat> refImgDataPatches = imSplitPatches(refImgData, patchCount);
    std::vector<FeatureVector> hogPatchesData(nPatches);
    for (size_t p = 0; p < nPatches; p++)
    {
        hogPatchesData[p] = hog.compute(refImgDataPatches[p]);
        logger << "roiID0003.txt hog588-impl-data-patch" << std::to_string(p) << ": " << featuresToVectorString(hogPatchesData[p]) << std::endl;
    }

    // employ 'text' + enforce 'double'
    std::vector<FeatureVector> hogPatchesDblData(nPatches);
    int patchRow = 0;
    int patchCol = 0;
    for (size_t p = 0; p < nPatches; p++)
    {
        for (int y = 0; y < patchSide; y++)
            for (int x = 0; x < patchSide; x++)
                patchData[y * patchSide + x] = refImgRawData[y + x * patchSide + patchCol * patchSide + patchRow * patchSide * refImg.size().width];
        patchCol++;
        if (patchCol >= patchCount.width)
        {
            patchCol = 0;
            patchRow++;
        }
        HOG(patchData, hogParams, patchSize, features, 1);
        hogPatchesDblData[p] = std::vector<double>(features, features + nFeatures);
        logger << "roiID0003.txt hog588-impl-DBL-data-patch" << std::to_string(p) << ": " << featuresToVectorString(hogPatchesDblData[p]) << std::endl;
    }

    // employ permuted[2,1] 'text' data (not equivalent to transpose)
    double *refImgRawDataPermute = new double[refImgSide*refImgSide]{
        197,196,195,195,197,199,189,172,139,107,118,122,130,114,119,148,161,181,190,192,191,190,191,190,189,188,185,186,184,182,181,179,179,178,177,176,175,175,174,174,173,173,173,173,174,174,174,174,
        196,196,198,199,181,127, 77, 44, 30, 26, 35, 34, 36, 31, 41, 47, 62, 94,127,156,175,170,162,161,164,170,170,162,158,161,175,176,175,174,173,172,171,171,171,171,170,170,170,170,171,171,170,171,
        199,196,170,108, 55, 30, 30, 27, 26, 25, 22, 25, 25, 26, 24, 24, 27, 29, 34, 53, 90, 74, 77, 96,112,127,135,128,133,136,134,165,171,170,169,168,168,167,167,167,167,167,167,167,168,167,167,167,
        165,103, 51, 25, 27, 30, 32, 30, 27, 27, 27, 26, 23, 19, 19, 19, 19, 19, 23, 40, 75, 68, 57, 73, 89,103,115,131,135,129,102,135,168,165,164,164,164,163,163,163,164,163,164,164,164,163,163,164,
         52, 26, 26, 32, 33, 30, 27, 26, 27, 26, 22, 23, 28, 33, 41, 50, 57, 60, 58, 67, 90,100,104,104,100,102,103,108,109,109, 97,120,149,155,159,161,160,159,159,159,159,159,160,160,159,160,159,160,
         33, 33, 34, 36, 32, 27, 25, 25, 25, 24, 26, 37, 53, 76,100,114,121,122, 99, 93, 99,121,120,120,116,112,109,104,102, 99,101,102,103,107,118,134,148,154,155,154,155,155,155,156,155,156,157,157,
         28, 29, 31, 32, 29, 27, 22, 22, 23, 30, 57, 85,101,112,120,124,126, 97, 28, 26, 38, 80,108,115,111,103,104,100, 94, 98,100, 99, 99,100, 97, 96,103,117,136,149,151,151,150,151,153,154,142,114,
         25, 31, 32, 27, 24, 24, 24, 29, 47, 77,104,114,118,119,117,117,113, 65, 89, 98, 98, 88, 86,108,101,100,102, 97, 95,102, 99, 99,105,104, 98, 97, 96, 93, 93,107,128,144,148,149,139,103, 71, 59,
         23, 29, 31, 27, 31, 41, 51, 65, 94,115,119,121,119,114,108,113, 98, 72,116,124,132,132,120,113, 96, 97, 96,100,103,107,104,102,104,103,101, 98, 97, 95, 90, 85, 83, 99,124,118, 65, 42, 38, 34,
         28, 33, 29, 32, 44, 61, 83, 99,117,122,120,117,111,101,102,107, 88, 74, 92,113,138,141,123,110,111, 98, 91, 99,112,117,112,104,103, 99,101, 97, 96, 95, 91, 85, 77, 72, 74, 73, 68, 66, 64, 60,
         34, 39, 49, 59, 64, 72, 82,106,121,123,123,118,101, 93, 96,100, 90, 69, 58, 95,133,141,126,117,123,104, 88,102,119,121,117,109,101,101,101,100, 98, 97, 92, 85, 78, 70, 65, 64, 63, 64, 69, 72,
         43, 51, 61, 66, 75, 87, 99,111,126,127,129,118, 93, 84, 90, 97, 96, 60, 41, 99,124,130,123,121,126,111, 92,112,115,117,118,109,106,104, 99,101,104,102, 97, 88, 79, 71, 65, 61, 58, 53, 51, 56,
         53, 53, 53, 53, 64, 80,107,130,134,133,131,107, 84, 78, 85, 99,107, 58, 43,105,120,114,119,120,128,121,100,119,119,119,122,115,110,103,101,101,104,106,103, 91, 81, 74, 68, 63, 57, 51, 47, 46,
         73, 81, 89, 98,109,117,133,142,140,140,137,114, 82, 68, 80,111,118, 62, 58,102,117,100,115,119,131,125,104,130,138,129,125,120,113,108,104,104,103,105,102, 98, 89, 81, 76, 73, 64, 54, 47, 43,
         94,111,121,129,135,136,144,145,146,148,142,117, 72, 56, 78,118,126, 65, 18, 57,113, 92,112,116,132,127,104,129,140,131,128,121,118,114,110,110,107,101, 99, 97, 95, 92, 87, 86, 76, 62, 49, 43,
        122,129,135,142,144,139,145,148,150,152,144,122, 71, 53, 68,108,128, 74, 17, 39,107, 91,110,116,131,128,104,123,138,132,126,123,119,120,112,114,107, 90, 80, 87,106,107, 98, 93, 86, 73, 58, 44,
        132,138,144,147,148,147,149,151,153,149,146,129, 74, 48, 62, 92,112, 83, 20, 47,105, 97,113,117,131,126,101,117,137,131,124,124,126,130,126,116, 99, 65, 47, 95,125,120,109, 99, 93, 88, 72, 49,
        130,141,149,149,150,152,154,156,156,154,149,134, 89, 52, 65, 78, 92, 91, 50, 52,109,116,118,118,133,121, 97,112,132,127,122,130,138,136,131,115, 96, 52, 35,112,128,122,115,103, 97, 99, 90, 64,
        129,142,147,152,155,157,157,158,159,158,152,135,108, 72, 70, 75, 82, 86, 75, 74,111,130,120,119,130,111, 90,110,125,119,105, 94,114,129,125,114,101, 59, 38,102,111,108,110,103,103,106,100, 81,
        125,140,150,155,158,157,156,158,162,158,151,136,119, 90, 73, 76, 80, 77, 74, 94,121,136,115,107,117, 97, 91,114,131,119, 74, 47, 56,113,123,116,113, 62, 50, 92, 91, 85, 95,102,105,108,105, 89,
        113,138,151,158,161,159,157,160,161,162,152,136,119,101, 82, 74, 77, 75, 67,101,125,102, 72, 69, 97,101,121,132,137,108, 39, 31, 49,100,121,120,115, 49, 64,101, 79, 63, 73, 95,106,113,110, 94,
         96,132,152,158,163,160,158,162,164,165,152,136,121,108, 98, 83, 80, 83, 64, 63, 56, 28, 41, 80,109,131,134,139,135,105, 47, 29, 52, 90,112,126,107, 45, 64,117, 82, 55, 55, 84,105,115,118,100,
         82,118,145,157,162,159,160,162,165,165,151,138,127,118,110,100, 95, 97, 83, 23, 42, 45, 86,130,139,144,148,151,142,111, 68, 40, 46, 74,100,127,108, 49, 61,124, 93, 57, 45, 72,107,116,117,101,
         69, 98,131,149,156,159,158,160,165,163,151,141,134,125,115,108, 97,102, 88, 55,113, 79,149,155,153,152,160,165,144,116, 75, 43, 40, 53, 72,109,125, 58, 54,123, 98, 59, 39, 62,109,115,112, 97,
         44, 53, 84,121,141,150,156,162,165,162,152,145,137,128,116,104, 95, 95, 84, 38, 82, 71,127,138,139,140,138,139,132,111, 72, 41, 40, 47, 64,115,117, 48, 54,124, 94, 56, 38, 60,109,117,108, 93,
         45, 62, 63, 62, 69, 97,127,158,165,162,157,144,135,123,109, 94, 85, 84, 76, 25, 19, 26, 48, 82,103,112,112,115,120,102, 58, 34, 42, 49, 71,119, 93, 38, 64,119, 81, 51, 42, 62,104,114,108, 92,
         62, 49, 52, 63, 59, 44, 49,103,153,162,155,140,126,113, 97, 79, 69, 71, 71, 65, 76, 47, 32, 45, 78, 95, 99,110,119, 95, 38, 26, 41, 55, 79,103, 94, 39, 66, 97, 65, 51, 49, 72,100,108,107, 88,
         59, 55, 60, 64, 66, 65, 54, 62,109,144,149,135,119,104, 84, 68, 66, 71, 74, 81,114,115, 86, 71, 90, 85, 84, 97,116,100, 49, 35, 38, 64, 85, 97, 95, 42, 59, 76, 62, 57, 62, 80, 96,104,100, 82,
         64, 63, 72, 88, 79, 82, 86, 92,113,131,140,130,116, 95, 73, 69, 68, 78, 75, 85,116,127,110,103,111,100, 79, 87,101, 98, 68, 46, 49, 81, 94, 95, 90, 48, 42, 77, 76, 72, 78, 86, 94, 98, 92, 74,
         60, 71, 76, 82, 86, 83, 86, 88,110,129,134,128,115, 82, 60, 61, 67, 82, 76, 69,102,126,118,111,122,113, 90, 94,117,106, 92, 89, 96,101,101, 93, 80, 44, 39, 93, 92, 91, 88, 88, 88, 88, 86, 67,
         52, 64, 80, 95,101,113,114,114,120,138,136,128,107, 58, 50, 59, 76, 89, 52, 44,105,112,114,111,117,118, 97, 98,128,124,114,113,113,110,108, 95, 73, 42, 53,106,108, 99, 90, 84, 81, 80, 75, 59,
         47, 39, 58, 81, 93,110,125,136,134,142,139,127, 88, 46, 49, 66,103,101, 27, 34,103, 99,106,112,118,123,101, 99,131,128,117,108,109,113,112,100, 81, 62, 66,108,111, 94, 86, 80, 77, 75, 64, 51,
         43, 31, 28, 45, 71, 97,115,130,137,139,135,121, 78, 40, 45, 82,126,102, 25, 17, 90, 92, 96,110,116,122,101, 98,131,128,118,107,102,105,107,103, 91, 83, 78, 93, 96, 85, 81, 76, 72, 65, 52, 44,
         38, 37, 35, 30, 29, 56, 90,105,129,134,130,117, 77, 44, 47, 90,126, 84, 19, 19, 86, 90, 92,109,114,118, 97, 96,128,123,113,107, 98, 91, 94, 94, 89, 86, 82, 86, 83, 76, 74, 69, 61, 53, 46, 40,
         36, 49, 49, 44, 32, 23, 34, 66,104,125,124,111, 74, 53, 56, 78,110, 61, 28, 53, 98, 90, 95,111,113,110, 91, 89,123,116,110,104, 89, 80, 82, 85, 86, 91, 89, 82, 74, 66, 63, 58, 54, 48, 45, 38,
         24, 33, 50, 62, 60, 50, 38, 34, 52,102,118,106, 77, 65, 65, 76, 95, 53, 29, 42, 94, 95,105,111,110,105, 86, 83,114,109,105, 95, 81, 71, 75, 81, 87, 93, 85, 74, 65, 58, 54, 51, 50, 47, 42, 38,
         23, 27, 24, 34, 53, 71, 71, 62, 45, 62,103,102, 82, 70, 70, 75, 83, 53, 35, 34, 85,105,116,112,109,104, 85, 75, 99,100, 91, 85, 77, 73, 74, 74, 82, 83, 73, 67, 60, 54, 51, 51, 51, 46, 41, 40,
         16, 26, 36, 39, 40, 46, 63, 74, 76, 58, 75, 94, 87, 71, 70, 71, 74, 51, 49, 36, 77,115,117,106,106, 98, 81, 70, 82, 87, 82, 82, 80, 69, 67, 71, 75, 72, 67, 63, 58, 53, 54, 53, 50, 45, 45, 46,
         17, 17, 23, 35, 49, 56, 52, 56, 65, 74, 71, 83, 87, 70, 69, 67, 71, 47, 54, 53, 78,113,115,108,101, 86, 75, 64, 70, 79, 77, 77, 78, 66, 63, 66, 70, 70, 66, 60, 58, 58, 57, 54, 51, 50, 50, 43,
         21, 16, 14, 18, 28, 40, 52, 59, 59, 62, 71, 77, 85, 79, 73, 74, 73, 54, 48, 77, 90,108,103, 99, 96, 78, 67, 64, 69, 71, 70, 70, 70, 68, 63, 65, 67, 66, 64, 62, 62, 61, 58, 58, 61, 43, 22, 13,
         23, 18, 16, 13, 15, 21, 30, 44, 61, 66, 62, 70, 78, 82, 83, 85, 80, 75, 38, 41, 47, 45, 60, 87, 90, 72, 65, 67, 70, 70, 70, 68, 66, 67, 66, 66, 65, 66, 66, 66, 63, 62, 66, 71, 72, 66, 47, 29,
         23, 22, 18, 15, 13, 13, 18, 25, 36, 50, 62, 62, 71, 77, 81, 85, 85, 89, 73, 25, 21, 52,103,103, 83, 77, 71, 70, 70, 69, 70, 71, 70, 70, 69, 69, 69, 69, 69, 67, 70, 74, 77, 78, 78, 80, 81, 71,
         25, 21, 19, 16, 14, 12, 13, 15, 20, 25, 35, 48, 57, 67, 72, 81, 84, 91, 91, 89, 92, 89, 96, 92, 86, 82, 79, 78, 76, 75, 73, 73, 74, 73, 72, 72, 72, 74, 77, 78, 80, 83, 83, 84, 85, 86, 87, 87,
         24, 23, 23, 22, 19, 15, 12, 12, 14, 15, 17, 23, 32, 42, 52, 62, 71, 79, 81, 76, 83, 83, 84, 81, 80, 78, 80, 78, 76, 75, 73, 70, 73, 76, 76, 80, 83, 86, 87, 88, 89, 91, 92, 93, 94, 95, 96, 96,
         22, 26, 22, 18, 21, 20, 15, 12, 13, 13, 11, 12, 15, 20, 25, 31, 37, 41, 40, 24, 20, 42, 52, 53, 52, 50, 50, 53, 61, 66, 62, 56, 73, 88, 88, 91, 94, 95, 96, 97, 99,100,101,103,104,105,106,106,
         22, 28, 30, 26, 20, 18, 19, 17, 15, 12, 10, 10, 10, 11, 13, 14, 14, 13, 12, 12, 15, 24, 30, 27, 31, 36, 37, 39, 64, 83, 80, 71, 85,100, 98,102,104,105,106,107,108,110,111,112,113,113,113,112,
         34, 22, 27, 28, 26, 23, 22, 25, 23, 16, 12, 10, 10, 11, 12, 13, 13, 13, 12, 12, 19, 36, 37, 42, 49, 55, 59, 64, 72, 78, 95, 96,107,110,109,112,114,114,115,116,118,119,120,120,119,120,119,118,
        141, 91, 44, 29, 27, 26, 27, 25, 27, 26, 23, 17, 14, 15, 18, 17, 16, 16, 16, 15, 20, 40, 50, 55, 57, 63, 76, 90, 93, 97,109,120,120,120,120,121,123,124,125,125,125,127,127,126,126,125,124,124,
    };
    cv::Mat refImgDataPermute = cv::Mat(refImgSide, refImgSide, CV_32F, refImgRawDataPermute);
    std::vector<cv::Mat> refImgDataPermutePatches = imSplitPatches(refImgDataPermute, patchCount);
    std::vector<FeatureVector> hogPatchesDataPermute(nPatches);
    for (size_t p = 0; p < nPatches; p++)
    {
        hogPatchesDataPermute[p] = hog.compute(refImgDataPermutePatches[p]);
        logger << "roiID0003.txt hog588-impl-data-permute-patch" << std::to_string(p) << ": " << featuresToVectorString(hogPatchesDataPermute[p]) << std::endl;
    }

    // employ permuted 'text' + enforce 'double'
    std::vector<FeatureVector> hogPatchesDblDataPermute(nPatches);
    patchRow = 0;
    patchCol = 0;
    for (size_t p = 0; p < nPatches; p++)
    {
        for (int y = 0; y < patchSide; y++)
            for (int x = 0; x < patchSide; x++)
                patchData[y * patchSide + x] = refImgRawDataPermute[y + x * patchSide + patchCol * patchSide + patchRow * patchSide * refImg.size().width];
        patchCol++;
        if (patchCol >= patchCount.width)
        {
            patchCol = 0;
            patchRow++;
        }
        HOG(patchData, hogParams, patchSize, features, 1);
        hogPatchesDblDataPermute[p] = std::vector<double>(features, features + nFeatures);
        logger << "roiID0003.txt hog588-impl-DBL-data-permute-patch" << std::to_string(p) << ": " << featuresToVectorString(hogPatchesDblDataPermute[p]) << std::endl;
    }

    return 0;
}

int test_multiLevelVectors()
{
    int vdim = 6;
    FeatureVector v1{  1,  2,  3,  4,  5,  6 };
    FeatureVector v2{  7,  8,  9, 10, 11, 12 };
    FeatureVector v3{ 13, 14, 15, 16, 17, 18 };
    FeatureVector v4{ 19, 20, 21, 22, 23, 24 };
    FeatureVector v5{ 25, 26, 27, 28, 29, 30 };
    FeatureVector v6{ 31, 32, 33, 34, 35, 36 };

    // Pre-initialized 1D-vector of FV, assigned by index
    xstd::mvector<1, FeatureVector> mv_index(2);
    mv_index[0] = v1;
    mv_index[1] = v2;
    for (int f = 0; f < vdim; f++)
    {
        ASSERT_LOG(mv_index[0][f] == f + 1, "Multi-level vector assigned by index should have correspoding feature value");
        ASSERT_LOG(mv_index[1][f] == f + 7, "Multi-level vector assigned by index should have correspoding feature value");
    }

    // Not initialized 1D-vector of FV, added by push_back
    xstd::mvector<1, FeatureVector> mv_push;
    mv_push.push_back(v1);
    mv_push.push_back(v2);
    for (int f = 0; f < vdim; f++)
    {
        ASSERT_LOG(mv_push[0][f] == f + 1, "Multi-level vector assigned by index should have correspoding feature value");
        ASSERT_LOG(mv_push[1][f] == f + 7, "Multi-level vector assigned by index should have correspoding feature value");
    }

    // Pre-initialized 2D-vector of FV using single size
    int singleSize = 5;
    xstd::mvector<2, FeatureVector> mvv_singleSize(singleSize);
    ASSERT_LOG(mvv_singleSize.size() == singleSize, "Multi-level vector assigned with single size should have same dimension on each level");
    for (int L1 = 0; L1 < singleSize; L1++)
    {
        ASSERT_LOG(mvv_singleSize[L1].size() == singleSize, "Multi-level vector assigned with single size should have same dimension on each level");
        for (int L2 = 0; L2 < singleSize; L2++)
            ASSERT_LOG(mvv_singleSize[L1][L2].size() == 0, "FeatureVector as lowest level object of multi-level vector should not be initialized");
    }

    // Pre-initialized 1D-vector of FV using sizes per level, assigned by index    
    size_t dims[2] = { 3, 2 };
    xstd::mvector<2, FeatureVector> mvv_index(dims);
    ASSERT_LOG(mvv_index.size() == dims[0], "Multi-level vector should be initialized with specified 1st level dimension");
    for (int L1 = 0; L1 < dims[0]; L1++)
        ASSERT_LOG(mvv_index[L1].size() == dims[1], "Each multi-level vector on 2nd level should be initialized with specified dimension");
    mvv_index[0][0] = v1;
    mvv_index[0][1] = v2;
    mvv_index[1][0] = v3;
    mvv_index[1][1] = v4;
    mvv_index[2][0] = v5;
    mvv_index[2][1] = v6;
    for (int L1 = 0; L1 < dims[0]; L1++)
        for (int L2 = 0; L2 < dims[1]; L2++)
        {
            ASSERT_LOG(mvv_index[L1][L2].size() == vdim, "Lowest level feature vector should have original dimension");
            for (int f = 0; f < vdim; f++)
            {
                int fval = (L1 * dims[1] + L2) * vdim + f + 1;
                ASSERT_LOG(mvv_index[L1][L2][f] == fval, "Multi-level vector assigned by index should have correspoding feature value");
            }
        }

    // Not initialized 2D-vector of FV, added by push_back of sub multi-level vectors
    xstd::mvector<2, FeatureVector> mvv_push;
    for (int L1 = 0; L1 < dims[0]; L1++)    
        mvv_push.push_back(xstd::mvector<1, FeatureVector>());
    ASSERT_LOG(mvv_push.size() == dims[0], "Multi-level vector should be initialized with number of pushed vectors");
    for (int L1 = 0; L1 < dims[0]; L1++)
        ASSERT_LOG(mvv_push[L1].size() == 0, "Second level vector should still be empty");
    mvv_push[0].push_back(v1);
    mvv_push[0].push_back(v2);
    mvv_push[1].push_back(v3);
    mvv_push[1].push_back(v4);
    mvv_push[2].push_back(v5);
    mvv_push[2].push_back(v6);
    for (int L1 = 0; L1 < dims[0]; L1++)
    {
        ASSERT_LOG(mvv_push[L1].size() == dims[1], "Each multi-level vector on 2nd level should be initialized with number of pushed vectors");
        for (int L2 = 0; L2 < dims[1]; L2++)
        {
            ASSERT_LOG(mvv_push[L1][L2].size() == vdim, "Lowest level feature vector should have original dimension");
            for (int f = 0; f < vdim; f++)
            {
                int fval = (L1 * dims[1] + L2) * vdim + f + 1;
                ASSERT_LOG(mvv_push[L1][L2][f] == fval, "Multi-level vector assigned with push back should have correspoding feature value");
            }
        }
    }
    mvv_push[1].push_back(v1);
    ASSERT_LOG(mvv_push.size() == dims[0], "Top level dimension of multi-level vector shouldn't be affected by lower level push");
    ASSERT_LOG(mvv_push[0].size() == dims[1], "Other lower level index then vector that got another push shouldn't be affected in size");
    ASSERT_LOG(mvv_push[1].size() == dims[1]+1, "Lower level vector with additional pushed feature vector should be expanded by one");
    ASSERT_LOG(mvv_push[2].size() == dims[1], "Other lower level index then vector that got another push shouldn't be affected in size");

    return 0;
}

int test_normalizationFunctions()
{
    FeatureVector v1 { -1, 2, 3, 4, 5, 14 };
    FeatureVector v2 { 8, 4, 6, 0.5, 1, 5 };
    FeatureVector v1_norm01 = { 0.0, 0.2, 4.0/15.0, 1.0/3.0, 0.4, 1.0 };
    FeatureVector v2_min =  { 2, 0, 4, 0, 0, 5 };
    FeatureVector v2_max =  { 8, 8, 8, 4, 4, 8 };
    FeatureVector v2_norm = { 1, 0.5, 0.5, 0.125, 0.25, 0 };    
    std::vector<FeatureVector> v = { v1, v2 };    

    ASSERT_LOG(normalizeMinMax(0.5, -1.0, 1.0) == 0.75, "Value should have been normalized with min-max rule");
    
    double min1 = -1, max1 = -1, min2 = -1, max2 = -1;
    int posMin1 = -1, posMax1 = -1, posMin2 = -1, posMax2 = -1;
    findMinMax(v1, &min1, &max1, &posMin1, &posMax1);
    findMinMax(v2, &min2, &max2, &posMin2, &posMax2);
    ASSERT_LOG(min1 == -1,   "Minimum value of vector should be assigned to variable by reference");
    ASSERT_LOG(max1 == 14,   "Maximum value of vector should be assigned to variable by reference");
    ASSERT_LOG(posMin1 == 0, "Index position of minimum value of vector should be assigned to variable by reference");
    ASSERT_LOG(posMax1 == 5, "Index position of maximum value of vector should be assigned to variable by reference");
    ASSERT_LOG(min2 == 0.5,  "Minimum value of vector should be assigned to variable by reference");
    ASSERT_LOG(max2 == 8,    "Maximum value of vector should be assigned to variable by reference");
    ASSERT_LOG(posMin2 == 3, "Index position of minimum value of vector should be assigned to variable by reference");
    ASSERT_LOG(posMax2 == 0, "Index position of maximum value of vector should be assigned to variable by reference");

    FeatureVector vmin, vmax;
    findMinMaxFeatures(v, &vmin, &vmax);
    ASSERT_LOG(vmin.size() == v1.size(), "Minimum features vector should be assigned values to match size of search vector");
    ASSERT_LOG(vmax.size() == v1.size(), "Maximum features vector should be assigned values to match size of search vector");
    ASSERT_LOG(vmin[0] == -1,  "Minimum value should be found");
    ASSERT_LOG(vmin[1] == 2,   "Minimum value should be found");
    ASSERT_LOG(vmin[2] == 3,   "Minimum value should be found");
    ASSERT_LOG(vmin[3] == 0.5, "Minimum value should be found");
    ASSERT_LOG(vmin[4] == 1,   "Minimum value should be found");
    ASSERT_LOG(vmin[5] == 5,   "Minimum value should be found");
    ASSERT_LOG(vmax[0] == 8,   "Maximum value should be found");
    ASSERT_LOG(vmax[1] == 4,   "Maximum value should be found");
    ASSERT_LOG(vmax[2] == 6,   "Maximum value should be found");
    ASSERT_LOG(vmax[3] == 4,   "Maximum value should be found");
    ASSERT_LOG(vmax[4] == 5,   "Maximum value should be found");
    ASSERT_LOG(vmax[5] == 14,  "Maximum value should be found");

    double minAll, maxAll;
    findMinMaxOverall(v, &minAll, &maxAll);
    ASSERT_LOG(minAll == -1, "Minimum value of all features of whole list should be found");
    ASSERT_LOG(maxAll == 14, "Maximum value of all features of whole list should be found");

    FeatureVector normAll = normalizeMinMaxAllFeatures(v1, -1, 14);     // min/max of v1 are -1,14, makes (max-min)=15
    for (int f = 0; f < normAll.size(); f++)
        ASSERT_LOG(normAll[f] == v1_norm01[f], "Feature should be normalized with specified min/max values");

    FeatureVector normAllMore = normalizeMinMaxAllFeatures(v1, -1, 29); // using max == 29 makes (max-min)=30, 1/2 norm values
    for (int f = 0; f < normAllMore.size(); f++)
        ASSERT_LOG(normAllMore[f] == v1_norm01[f] / 2.0, "Feature normalization should be enforced with specified min/max values");

    FeatureVector normAllAuto = normalizeMinMaxAllFeatures(v1);         // min/max not specified, find them
    for (int f = 0; f < normAllAuto.size(); f++)
        ASSERT_LOG(normAllAuto[f] == v1_norm01[f], "Feature should be normalized with min/max found within the specified vector");
    
    std::vector<double> scores = normalizeMinMaxClassScores(v1);        // same as 'normalizeMinMaxAllFeatures'
    for (int f = 0; f < scores.size(); f++)
        ASSERT_LOG(scores[f] == v1_norm01[f], "Score should be normalized with min/max of all scores");

    FeatureVector v2_normPerFeat = normalizeMinMaxPerFeatures(v2, v2_min, v2_max);
    for (int f = 0; f < v2_normPerFeat.size(); f++)
    {
        cout << v2_normPerFeat[f] << " " << v2_norm[f] << std::endl;
        ASSERT_LOG(v2_normPerFeat[f] == v2_norm[f], "Feature should be normalized with corresponding min/max features");
    }

    double s1pos = normalizeClassScoreToSimilarity(+1);
    double s1neg = normalizeClassScoreToSimilarity(-1);
    double s0mid = normalizeClassScoreToSimilarity(0);
    double sprob = normalizeClassScoreToSimilarity(0.5);
    ASSERT_LOG(s1pos == 1, "Positive class score should be normalized as maximum similarity");
    ASSERT_LOG(s1neg == 0, "Negative class score should be normalized as minimum similarity");
    ASSERT_LOG(s0mid == 0.5, "Indifferent class score should be normalized as middle similarity");
    ASSERT_LOG(sprob == 0.75, "Half-probable positive class score should be normalized as 3/4 simiarity");

    return 0;
}

int test_performanceEvaluationFunctions()
{    
    // test basic confusion matrix operations
    ASSERT_LOG(calcTPR(2250, 250) == 0.9,  "Invalid calculation result for TPR");
    ASSERT_LOG(calcSPC(100, 900)  == 0.1,  "Invalid calculation result for SPC");
    ASSERT_LOG(calcFPR(1500, 500) == 0.75, "Invalid calculation result for FPR");
    ASSERT_LOG(calcPPV(2250, 1500) == 0.6, "Invalid calculation result for PPV");
    ASSERT_LOG(calcTNR(500, 1500) == 0.25, "Invalid calculation result for TNR");
    ASSERT_LOG(calcACC(2450, 650, 1650, 250) == 0.62, "Invalid calculation result for ACC");

    // test area under curve calculation
    std::vector<double> FPR = { 0.000, 0.100, 0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900, 1.000 };
    std::vector<double> TPR = { 0.900, 0.950, 0.975, 0.990, 0.990, 0.990, 0.990, 0.990, 0.990, 0.995, 1.000 };
    double AUC_valid = 0.98100;         // AUC [0,1] of values
    double pAUC20_valid = 0.18875;      // pAUC(20%) of values, lands perfectly on an existing FPR           
    double pAUC35_valid = 0.33650;      // pAUC(35%) of values, lands between existing FPRs, interpolation applied   
    double AUC_result = calcAUC(FPR, TPR);
    double pAUC20_result = calcAUC(FPR, TPR, 0.20);
    double pAUC35_result = calcAUC(FPR, TPR, 0.35);
    ASSERT_LOG(doubleAlmostEquals(AUC_valid, AUC_result), "AUC calculation should return expected value");
    ASSERT_LOG(doubleAlmostEquals(pAUC20_valid, pAUC20_result), "pAUC(20%) calculation should return expected value");
    ASSERT_LOG(doubleAlmostEquals(pAUC35_valid, pAUC35_result), "pAUC(35%) calculation should return expected value");
    
    // test to display results (visual inspection in console / logger)
    // should display like the table below with corresponding valid values
    /*
        Target IDs |      AUC      |   pAUC(10%)   |   pAUC(20%)   |      AUPR     
        ---------------------------------------------------------------------------
        TEST       |             1 |           0.1 |           0.2 |           0.8
    */
    logstream logger(LOGGER_FILE);
    xstd::mvector<2, double> scores;
    xstd::mvector<2, int> groundTruths;
    std::vector<std::string> targets = { "TEST" };
    std::vector<double> targetScores{ 0.9, 0.85, 0.92, 0.89, 0.87, 0.63, 0.42, 0.56 };
    std::vector<int> targetOutputs{ 1, 1, 1, 1, 1, -1, -1, -1 };
    scores.push_back(targetScores);
    groundTruths.push_back(targetOutputs);
    logger << "Displaying results table from dummy classification scores:" << std::endl;
    eval_PerformanceClassificationSummary(targets, scores, groundTruths);

    return 0;
}

#if 0 // DISABLE - USING OBSOLETE MATLAB PROCEDURE
int test_runBasicExemplarSvmFunctionalities(void)
{
    // ------------------------------------------------------------------------------------------------------------------------
    // window to display loaded images and stream for console+file output
    // ------------------------------------------------------------------------------------------------------------------------    
    cv::namedWindow(WINDOW_NAME);
    logstream logger(LOGGER_FILE);
    logger << "Starting basic Exemplar-SVM functionality test..." << std::endl;

    // ------------------------------------------------------------------------------------------------------------------------
    // C++ parameters
    // ------------------------------------------------------------------------------------------------------------------------
    /* Positive training samples */
    std::string targetName = "person_6";
    const int NB_POSITIVE_IMAGES = 13;
    cv::Mat matPositiveSamples[NB_POSITIVE_IMAGES];
    logger << "Loading positive training samples..." << std::endl;
    matPositiveSamples[0]  = imReadAndDisplay(roiVideoImagesPath + "person_6/000246.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matPositiveSamples[1]  = imReadAndDisplay(roiVideoImagesPath + "person_6/000247.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matPositiveSamples[2]  = imReadAndDisplay(roiVideoImagesPath + "person_6/000250.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matPositiveSamples[3]  = imReadAndDisplay(roiVideoImagesPath + "person_6/000255.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matPositiveSamples[4]  = imReadAndDisplay(roiVideoImagesPath + "person_6/000260.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matPositiveSamples[5]  = imReadAndDisplay(roiVideoImagesPath + "person_6/000265.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matPositiveSamples[6]  = imReadAndDisplay(roiVideoImagesPath + "person_6/000270.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matPositiveSamples[7]  = imReadAndDisplay(roiVideoImagesPath + "person_6/000280.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matPositiveSamples[8]  = imReadAndDisplay(roiVideoImagesPath + "person_6/000285.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matPositiveSamples[9]  = imReadAndDisplay(roiVideoImagesPath + "person_6/000286.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matPositiveSamples[10] = imReadAndDisplay(roiVideoImagesPath + "person_6/000290.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matPositiveSamples[11] = imReadAndDisplay(roiVideoImagesPath + "person_6/000295.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matPositiveSamples[12] = imReadAndDisplay(roiVideoImagesPath + "person_6/000300.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    /* Negative training samples */
    const int NB_NEGATIVE_IMAGES = 36;
    cv::Mat matNegativeSamples[NB_NEGATIVE_IMAGES];
    logger << "Loading negative training samples..." << std::endl;
    matNegativeSamples[0]  = imReadAndDisplay(roiVideoImagesPath + "person_16/000350.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[1]  = imReadAndDisplay(roiVideoImagesPath + "person_16/000355.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[2]  = imReadAndDisplay(roiVideoImagesPath + "person_16/000360.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[3]  = imReadAndDisplay(roiVideoImagesPath + "person_16/000361.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[4]  = imReadAndDisplay(roiVideoImagesPath + "person_16/000365.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[5]  = imReadAndDisplay(roiVideoImagesPath + "person_16/000370.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[6]  = imReadAndDisplay(roiVideoImagesPath + "person_20/000410.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[7]  = imReadAndDisplay(roiVideoImagesPath + "person_20/000415.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[8]  = imReadAndDisplay(roiVideoImagesPath + "person_20/000420.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[9]  = imReadAndDisplay(roiVideoImagesPath + "person_20/000425.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[10] = imReadAndDisplay(roiVideoImagesPath + "person_23/000435.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[11] = imReadAndDisplay(roiVideoImagesPath + "person_23/000440.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[12] = imReadAndDisplay(roiVideoImagesPath + "person_23/000445.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[13] = imReadAndDisplay(roiVideoImagesPath + "person_23/000450.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[14] = imReadAndDisplay(roiVideoImagesPath + "person_23/000455.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[15] = imReadAndDisplay(roiVideoImagesPath + "person_23/000460.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[16] = imReadAndDisplay(roiVideoImagesPath + "person_32/000495.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[17] = imReadAndDisplay(roiVideoImagesPath + "person_32/000500.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[18] = imReadAndDisplay(roiVideoImagesPath + "person_32/000505.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[19] = imReadAndDisplay(roiVideoImagesPath + "person_32/000510.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[20] = imReadAndDisplay(roiVideoImagesPath + "person_32/000515.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[21] = imReadAndDisplay(roiVideoImagesPath + "person_32/000520.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[22] = imReadAndDisplay(roiVideoImagesPath + "person_32/000525.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[23] = imReadAndDisplay(roiVideoImagesPath + "person_34/000540.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[24] = imReadAndDisplay(roiVideoImagesPath + "person_34/000545.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[25] = imReadAndDisplay(roiVideoImagesPath + "person_34/000550.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[26] = imReadAndDisplay(roiVideoImagesPath + "person_34/000560.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[27] = imReadAndDisplay(roiVideoImagesPath + "person_34/000570.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[28] = imReadAndDisplay(roiVideoImagesPath + "person_34/000575.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[29] = imReadAndDisplay(roiVideoImagesPath + "person_34/000585.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[30] = imReadAndDisplay(roiVideoImagesPath + "person_40/000670.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[31] = imReadAndDisplay(roiVideoImagesPath + "person_40/000675.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[32] = imReadAndDisplay(roiVideoImagesPath + "person_40/000680.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[33] = imReadAndDisplay(roiVideoImagesPath + "person_40/000685.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[34] = imReadAndDisplay(roiVideoImagesPath + "person_40/000690.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    matNegativeSamples[35] = imReadAndDisplay(roiVideoImagesPath + "person_40/000700.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    /* Probe testing samples */
    const int NB_PROBE_IMAGES = 4;
    cv::Mat matProbeSamples[NB_PROBE_IMAGES];
    logger << "Loading probe testing samples..." << std::endl;
    matProbeSamples[0] = imReadAndDisplay(roiVideoImagesPath + "person_9/000295.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);    // Negative
    matProbeSamples[1] = imReadAndDisplay(roiVideoImagesPath + "person_6/000275.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);    // Positive
    matProbeSamples[2] = imReadAndDisplay(roiVideoImagesPath + "person_37/000541.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);   // Negative
    matProbeSamples[3] = imReadAndDisplay(roiVideoImagesPath + "person_45/000680.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);   // Negative

    // Destroy viewing window not required anymore
    cv::destroyWindow(WINDOW_NAME);

    // ------------------------------------------------------------------------------------------------------------------------
    // Transform into MATLAB arrays
    // NB: these are one-based indexed
    // ------------------------------------------------------------------------------------------------------------------------
    mwArray scores;
    mwArray models;
    mwArray target(targetName.c_str());
    mwArray mwPositiveSamples(NB_POSITIVE_IMAGES, 1, mxCELL_CLASS);
    mwArray mwNegativeSamples(NB_NEGATIVE_IMAGES, 1, mxCELL_CLASS);
    mwArray mwProbeSamples(NB_PROBE_IMAGES, 1, mxCELL_CLASS);
    // Quick conversion verification tests
    logger << "Testing simple image conversion..." << std::endl;
    mwArray mtrx = convertCvToMatlabMat(matPositiveSamples[0]);
    logger << "Testing cell array get first cell..." << std::endl;
    mwArray cell = mwPositiveSamples.Get(1, 1);
    logger << "Testing cell array get last cell..." << std::endl;
    mwPositiveSamples.Get(1, NB_POSITIVE_IMAGES);
    logger << "Testing cell array set first cell data..." << std::endl;
    cell.Set(mtrx);
    // Full conversion for Exemplar-SVM
    logger << "Converting positive training samples..." << std::endl;
    for (int i = 0; i < NB_POSITIVE_IMAGES; i++)
        mwPositiveSamples.Get(1, i + 1).Set(convertCvToMatlabMat(matPositiveSamples[i]));
    logger << "Converting negative training samples..." << std::endl;
    for (int i = 0; i < NB_NEGATIVE_IMAGES; i++)
        mwNegativeSamples.Get(1, i + 1).Set(convertCvToMatlabMat(matNegativeSamples[i]));
    logger << "Converting probe testing samples..." << std::endl;
    for (int i = 0; i < NB_PROBE_IMAGES; i++)
        mwProbeSamples.Get(1, i + 1).Set(convertCvToMatlabMat(matProbeSamples[i]));

    // ------------------------------------------------------------------------------------------------------------------------
    // Try Exemplar-SVM training and testing
    // ------------------------------------------------------------------------------------------------------------------------
    try
    {        
        logger << "Running Exemplar-SVM training..." << std::endl;
        esvm_train_individual(1, models, mwPositiveSamples, mwNegativeSamples, target);
        logger << "Running Exemplar-SVM testing..." << std::endl;
        esvm_test_individual(1, scores, models, mwProbeSamples);
        logger << "Success" << std::endl;
        return 0;
    }
    catch (const mwException& e)
    {
        logger << e.what() << std::endl;
        return -2;
    }
    catch (...)
    {
        logger << "Unexpected error thrown" << std::endl;
        return -3;
    }
}
#endif 

int test_runBasicExemplarSvmClassification(void)
{
    // ------------------------------------------------------------------------------------------------------------------------
    // stream for console+file output
    // ------------------------------------------------------------------------------------------------------------------------        
    logstream logger(LOGGER_FILE);
    logger << "Starting basic Exemplar-SVM classification test..." << std::endl;

    // ------------------------------------------------------------------------------------------------------------------------
    // training ESVM with samples (XOR)
    // ------------------------------------------------------------------------------------------------------------------------ 
    logger << "Training Exemplar-SVM with XOR samples..." << std::endl;
    std::vector<FeatureVector> positives(20);
    std::vector<FeatureVector> negatives(20);  
    std::srand(std::time(0));
    for (int i = 0; i < 10; i++)
    {
        int i1 = 2 * i;
        int i2 = 2 * i + 1;
        double r = (double)(std::rand() % 50 - 25) / 100.0;
        // points around (0,0) -> 0
        negatives[i1] = FeatureVector(2);
        negatives[i1][0] = 0 + r;
        negatives[i1][1] = 0 + r;
        // points around (1,1) -> 0
        negatives[i2] = FeatureVector(2);
        negatives[i2][0] = 1 + r;
        negatives[i2][1] = 1 + r;
        // points around (0,1) -> 1
        positives[i1] = FeatureVector(2);
        positives[i1][0] = 0 + r;
        positives[i1][1] = 1 + r;
        // points around (1,0) -> 1
        positives[i2] = FeatureVector(2);
        positives[i2][0] = 1 + r;
        positives[i2][1] = 0 + r;
    }
    ESVM esvm = ESVM(positives, negatives, "XOR");

    // ------------------------------------------------------------------------------------------------------------------------
    // testing ESVM
    // ------------------------------------------------------------------------------------------------------------------------  
    logger << "Testing Exemplar-SVM classification results..." << std::endl;
    std::vector<FeatureVector> samples(6);
    samples[0] = { 0, 0 };          
    samples[1] = { 0, 1 };
    samples[2] = { 0.75, 0 };
    samples[3] = { 0.90, 0.75 };
    samples[4] = { 1, 0.75 };
    samples[5] = { 1, 1 };
    for (int s = 0; s < samples.size(); s++)
    {
        double prediction = esvm.predict(samples[s]);
        logger << "  Prediction result for {" << samples[s][0] << "," << samples[s][1] << "}: " << prediction << std::endl;
    }

    return 0;
}

// Tests LIBSVM format sample file reading functionality of ESVM (index and value parsing)
int test_runBasicExemplarSvmReadSampleFile_libsvm()
{
    logstream logger(LOGGER_FILE);
    logger << "Starting basic Exemplar-SVM sample file reading test..." << std::endl;

    // create test sample files inside test directory
    std::string testDir = "test_sample-read-file/";
    bfs::create_directory(testDir);
    std::string validSampleFileName1 = testDir + "test_valid-samples1.data";  // for testing normal indexed features
    std::string validSampleFileName2 = testDir + "test_valid-samples2.data";  // for testing sparse indexed features
    std::string validSampleFileName3 = testDir + "test_valid-samples3.data";  // for testing ending index producing same feature sizes
    std::string validSampleFileName4 = testDir + "test_valid-samples4.data";  // for testing omitted ending index
    std::string wrongSampleFileName1 = testDir + "test_wrong-samples1.data";  // for testing non ascending indexes
    std::string wrongSampleFileName2 = testDir + "test_wrong-samples2.data";  // for testing repeating indexes
    std::string wrongSampleFileName3 = testDir + "test_wrong-samples3.data";  // for testing non matching sample sizes
    std::string wrongSampleFileName4 = testDir + "test_wrong-samples4.data";  // for testing not found index:value separator
    std::string wrongSampleFileName5 = testDir + "test_wrong-samples5.data";  // for testing missing target output class value    
    std::ofstream validSampleFile1(validSampleFileName1);
    std::ofstream validSampleFile2(validSampleFileName2);
    std::ofstream validSampleFile3(validSampleFileName3);
    std::ofstream validSampleFile4(validSampleFileName4);
    std::ofstream wrongSampleFile1(wrongSampleFileName1);
    std::ofstream wrongSampleFile2(wrongSampleFileName2);
    std::ofstream wrongSampleFile3(wrongSampleFileName3);
    std::ofstream wrongSampleFile4(wrongSampleFileName4);
    std::ofstream wrongSampleFile5(wrongSampleFileName5);    

    // fill test sample files
    validSampleFile1 << std::to_string(ESVM_POSITIVE_CLASS) << " 1:10.999 2:20.111 3:30.555 4:40.333 5:50.777 -1:0" << std::endl;
    validSampleFile1 << std::to_string(ESVM_NEGATIVE_CLASS) << " 1:90.123 2:80.456 3:-70.78 4:-90000 5:-50.00 -1:0" << std::endl;
    validSampleFile2 << std::to_string(ESVM_NEGATIVE_CLASS) << " 1:12.345 4:6.7890 -1:0" << std::endl;                             // sparse
    validSampleFile3 << std::to_string(ESVM_NEGATIVE_CLASS) << " 1:99.111 2:88.222 3:77.333 -1:11111 5:50.777 -1:0" << std::endl;  // -1->5
    validSampleFile3 << std::to_string(ESVM_POSITIVE_CLASS) << " 1:11.222 2:33.444 3:55.666 -1:0" << std::endl;                    // == size
    validSampleFile4 << std::to_string(ESVM_POSITIVE_CLASS) << " 1:90.111 2:80.222 3:70.333 4:60.444 5:50.555" << std::endl;;      // missing '-1:'
    validSampleFile4 << std::to_string(ESVM_NEGATIVE_CLASS) << " 1:10.999 2:20.888 3:30.777 4:40.666 5:50.000" << std::endl;;      // missing '-1:'
    wrongSampleFile1 << std::to_string(ESVM_NEGATIVE_CLASS) << " 1:10.999 2:20.111 5:30.555 4:40.333 3:50.777 -1:0" << std::endl;  // 5->4->3
    wrongSampleFile2 << std::to_string(ESVM_NEGATIVE_CLASS) << " 1:10.999 2:20.111 3:30.555 5:40.333 5:50.777 -1:0" << std::endl;  // 3->5->5    
    wrongSampleFile3 << std::to_string(ESVM_NEGATIVE_CLASS) << " 1:10.999 2:20.111 3:30.555 4:40.333 5:50.777 -1:0" << std::endl;  // != size
    wrongSampleFile3 << std::to_string(ESVM_NEGATIVE_CLASS) << " 1:10.999 2:20.111 3:30.555 -1:0" << std::endl;                    // != size
    wrongSampleFile4 << std::to_string(ESVM_NEGATIVE_CLASS) << " 10.999 20.111 30.555 40.666 50.777 60.888 70.999" << std::endl;   // missing ':'
    wrongSampleFile5 << " 10.999 20.111 30.555 40.666 50.777 60.888 70.999" << std::endl;                       // missing target output class

    // close test sample files
    if (validSampleFile1.is_open()) validSampleFile1.close();
    if (validSampleFile2.is_open()) validSampleFile2.close();
    if (validSampleFile3.is_open()) validSampleFile3.close();
    if (validSampleFile4.is_open()) validSampleFile4.close();
    if (wrongSampleFile1.is_open()) wrongSampleFile1.close();
    if (wrongSampleFile2.is_open()) wrongSampleFile2.close();
    if (wrongSampleFile3.is_open()) wrongSampleFile3.close();
    if (wrongSampleFile4.is_open()) wrongSampleFile4.close();
    if (wrongSampleFile5.is_open()) wrongSampleFile5.close();

    /* --------
       tests 
    -------- */
    ESVM esvm;
    std::vector<FeatureVector> samples;
    std::vector<int> targetOutputs;
    try
    {
        // test valid normal indexed samples (exception not expected)
        esvm.readSampleDataFile(validSampleFileName1, samples, targetOutputs);        
        ASSERT_LOG(targetOutputs.size() == 2, "File reading should result in 2 loaded target output class");
        ASSERT_LOG(targetOutputs[0] == ESVM_POSITIVE_CLASS, "First sample target output class should be positive class value");
        ASSERT_LOG(targetOutputs[1] == ESVM_NEGATIVE_CLASS, "Second sample target output class should be negative class value");
        ASSERT_LOG(samples.size() == 2, "File reading should result in 2 loaded feature vector samples");
        ASSERT_LOG(samples[0].size() == 5, "Sample feature vector should have proper size according to values in file");
        ASSERT_LOG(samples[1].size() == 5, "Sample feature vector should have proper size according to values in file");
        ASSERT_LOG(samples[0][0] == 10.999, "Sample feature value should match value in file");
        ASSERT_LOG(samples[0][1] == 20.111, "Sample feature value should match value in file");
        ASSERT_LOG(samples[0][2] == 30.555, "Sample feature value should match value in file");
        ASSERT_LOG(samples[0][3] == 40.333, "Sample feature value should match value in file");
        ASSERT_LOG(samples[0][4] == 50.777, "Sample feature value should match value in file");
        ASSERT_LOG(samples[1][0] == 90.123, "Sample feature value should match value in file");
        ASSERT_LOG(samples[1][1] == 80.456, "Sample feature value should match value in file");
        ASSERT_LOG(samples[1][2] == -70.78, "Sample feature value should match value in file");
        ASSERT_LOG(samples[1][3] == -90000, "Sample feature value should match value in file");
        ASSERT_LOG(samples[1][4] == -50.00, "Sample feature value should match value in file");
    }
    catch (std::exception& ex)
    {
        logger << "Error: Valid normal indexed samples and file reading should not have generated an exception." << std::endl 
               << "Exception: " << std::endl << ex.what() << std::endl;
        bfs::remove_all(testDir);
        return -1;
    }
    try
    {
        // test valid sparse indexed samples (exception not expected)
        esvm.readSampleDataFile(validSampleFileName2, samples, targetOutputs);
        ASSERT_LOG(targetOutputs.size() == 1, "File reading should result in 1 loaded target output class");
        ASSERT_LOG(targetOutputs[0] == ESVM_NEGATIVE_CLASS, "First sample target output class should be negative class value");
        ASSERT_LOG(samples.size() == 1, "File reading should result in 1 loaded feature vector samples");
        ASSERT_LOG(samples[0].size() == 4, "Sample feature vector should have proper size according to values in file");
        ASSERT_LOG(samples[0][0] == 12.345, "Specified sample feature value should match value in file");
        ASSERT_LOG(samples[0][1] == 0, "Not specified sparse sample feature value should be set to zero");
        ASSERT_LOG(samples[0][2] == 0, "Not specified sparse sample feature value should be set to zero");
        ASSERT_LOG(samples[0][3] == 6.7890, "Specified sample feature value should match value in file");
    }
    catch (std::exception& ex)
    {
        logger << "Error: Valid sparse indexes samples and file reading should not have generated an exception." << std::endl
               << "Exception: " << std::endl << ex.what() << std::endl;
        bfs::remove_all(testDir);
        return -2;
    }
    try
    {
        // test valid limited final index (-1) samples (exception not expected)
        esvm.readSampleDataFile(validSampleFileName3, samples, targetOutputs);
        ASSERT_LOG(targetOutputs.size() == 2, "File reading should result in 2 loaded target output class");
        ASSERT_LOG(targetOutputs[0] == ESVM_NEGATIVE_CLASS, "First sample target output class should be negative class value");
        ASSERT_LOG(targetOutputs[1] == ESVM_POSITIVE_CLASS, "Second sample target output class should be positive class value");
        ASSERT_LOG(samples.size() == 2, "File reading should result in 2 loaded feature vector samples");
        ASSERT_LOG(samples[0].size() == 3, "Sample features should be set until -1 index is reached, following ones should be ignored");
        ASSERT_LOG(samples[1].size() == 3, "Sample features should have corresponding number of feature in file");
        ASSERT_LOG(samples[0][0] == 99.111, "Sample feature value should match value in file");
        ASSERT_LOG(samples[0][1] == 88.222, "Sample feature value should match value in file");
        ASSERT_LOG(samples[0][2] == 77.333, "Sample feature value should match value in file");
        ASSERT_LOG(samples[1][0] == 11.222, "Sample feature value should match value in file");
        ASSERT_LOG(samples[1][1] == 33.444, "Sample feature value should match value in file");
        ASSERT_LOG(samples[1][2] == 55.666, "Sample feature value should match value in file");
    }
    catch (std::exception& ex)
    {
        logger << "Error: Valid final limited indexed samples and file reading should not have generated an exception." << std::endl
               << "Exception: " << std::endl << ex.what() << std::endl;
        bfs::remove_all(testDir);
        return -3;
    }
    try
    {
        // test valid omitted final index (-1) samples (exception not expected)
        esvm.readSampleDataFile(validSampleFileName4, samples, targetOutputs);
        ASSERT_LOG(targetOutputs.size() == 2, "File reading should result in 2 loaded target output class");
        ASSERT_LOG(targetOutputs[0] == ESVM_POSITIVE_CLASS, "First sample target output class should be positive class value");
        ASSERT_LOG(targetOutputs[1] == ESVM_NEGATIVE_CLASS, "Second sample target output class should be negative class value");
        ASSERT_LOG(samples.size() == 2, "File reading should result in 2 loaded feature vector samples");
        ASSERT_LOG(samples[0].size() == 5, "Sample features should be set until -1 index is reached, following ones should be ignored");
        ASSERT_LOG(samples[1].size() == 5, "Sample features should have corresponding number of feature in file");
        ASSERT_LOG(samples[0][0] == 90.111, "Sample feature value should match value in file");
        ASSERT_LOG(samples[0][1] == 80.222, "Sample feature value should match value in file");
        ASSERT_LOG(samples[0][2] == 70.333, "Sample feature value should match value in file");
        ASSERT_LOG(samples[0][3] == 60.444, "Sample feature value should match value in file");
        ASSERT_LOG(samples[0][4] == 50.555, "Sample feature value should match value in file");
        ASSERT_LOG(samples[1][0] == 10.999, "Sample feature value should match value in file");
        ASSERT_LOG(samples[1][1] == 20.888, "Sample feature value should match value in file");
        ASSERT_LOG(samples[1][2] == 30.777, "Sample feature value should match value in file");
        ASSERT_LOG(samples[1][3] == 40.666, "Sample feature value should match value in file");
        ASSERT_LOG(samples[1][4] == 50.000, "Sample feature value should match value in file");
    }
    catch (std::exception& ex)
    {
        logger << "Error: Valid final limited indexed samples and file reading should not have generated an exception." << std::endl
               << "Exception: " << std::endl << ex.what() << std::endl;
        bfs::remove_all(testDir);
        return -4;
    }
    try
    {
        // test wrong non ascending index samples (exception expected)
        esvm.readSampleDataFile(wrongSampleFileName1, samples, targetOutputs);
        logger << "Error: Indexes not specified in ascending order should have raised an exception." << std::endl;
        return -5;
    }
    catch (...) {}
    try
    {
        // test wrong repeated index samples (exception expected)
        esvm.readSampleDataFile(wrongSampleFileName2, samples, targetOutputs);
        logger << "Error: Repeating indexes should have raised an exception." << std::endl;
        return -6;
    }
    catch (...) {}
    try
    {
        // test wrong non matching size samples (exception expected)
        esvm.readSampleDataFile(wrongSampleFileName3, samples, targetOutputs);
        logger << "Error: Non matching sample sizes should have raised an exception." << std::endl;
        return -7;
    }
    catch (...) {}
    try
    {
        // test wrong missing index:value separator (exception expected)
        esvm.readSampleDataFile(wrongSampleFileName4, samples, targetOutputs);
        logger << "Error: Not found 'index:value' seperator should have raised an exception." << std::endl;
        return -8;
    }
    catch (...) {}
    try
    {
        // test wrong missing target output class (exception expected)
        esvm.readSampleDataFile(wrongSampleFileName5, samples, targetOutputs);
        logger << "Error: Missing target output class value should have raised an exception." << std::endl;
        return -9;
    }
    catch (...) {}

    // delete test directory and sample files
    bfs::remove_all(testDir);
    return 0;
}

// Tests binary sample file reading functionality of ESVM
int test_runBasicExemplarSvmReadSampleFile_binary()
{
    logstream logger(LOGGER_FILE);
    logger << "Starting basic Exemplar-SVM sample file reading test..." << std::endl;

    // create test sample files inside test directory
    std::string testDir = "test_sample-read-binary-file/";
    bfs::create_directory(testDir);
    std::string validSampleFileName1 = testDir + "test_valid-samples1.data";  // for testing valid binary formatted file
    std::string wrongSampleFileName1 = testDir + "test_wrong-samples1.data";  // for testing missing header
    std::string wrongSampleFileName2 = testDir + "test_wrong-samples2.data";  // for testing invalid header
    std::string wrongSampleFileName3 = testDir + "test_wrong-samples3.data";  // for testing invalid number of samples
    std::string wrongSampleFileName4 = testDir + "test_wrong-samples4.data";  // for testing invalid number of features
    std::string wrongSampleFileName5 = testDir + "test_wrong-samples5.data";  // for testing missing target output class value

    // fill test sample files
    ESVM esvm;
    FeatureVector v1 = { 0.999, 1.888, 2.777, 3.666, 4.555 }, v2 = { 5.444, 6.333, 7.222, 8.111, 9.000 };
    std::vector<FeatureVector> validSamples = { v1, v2 };
    std::vector<int> validTargetOutputs = { ESVM_POSITIVE_CLASS, ESVM_NEGATIVE_CLASS };
    esvm.writeSampleDataFile(validSampleFileName1, validSamples, validTargetOutputs, BINARY);

    /*=====================
    TODO
    file not found
    missing/wrong header
    <= 0 n samples/features
    invalid reading (fail status ex: not enough samples compared to nSamples)
    valid reading
    =====================*/

    std::vector<FeatureVector> readSamples;
    std::vector<int> readTargetOutputs;

    try
    {
        // test reading valid samples encoded in binary file
        esvm.readSampleDataFile(validSampleFileName1, readSamples, readTargetOutputs, BINARY);
        ASSERT_LOG(readSamples.size() == validSamples.size(), "Number of samples read from binary file should match original");
        ASSERT_LOG(readSamples[0].size() == validSamples[0].size(), "Number of features read from binary file should match original");
        ASSERT_LOG(readTargetOutputs.size() == validTargetOutputs.size(), "Number of target outputs read from binary file should match original");
        ASSERT_LOG(readSamples[0][0] == validSamples[0][0], "Sample feature value from binary file should match the original one");
        ASSERT_LOG(readSamples[0][1] == validSamples[0][1], "Sample feature value from binary file should match the original one");
        ASSERT_LOG(readSamples[0][2] == validSamples[0][2], "Sample feature value from binary file should match the original one");
        ASSERT_LOG(readSamples[0][3] == validSamples[0][3], "Sample feature value from binary file should match the original one");
        ASSERT_LOG(readSamples[0][4] == validSamples[0][4], "Sample feature value from binary file should match the original one");
        ASSERT_LOG(readSamples[1][0] == validSamples[1][0], "Sample feature value from binary file should match the original one");
        ASSERT_LOG(readSamples[1][1] == validSamples[1][1], "Sample feature value from binary file should match the original one");
        ASSERT_LOG(readSamples[1][2] == validSamples[1][2], "Sample feature value from binary file should match the original one");
        ASSERT_LOG(readSamples[1][3] == validSamples[1][3], "Sample feature value from binary file should match the original one");
        ASSERT_LOG(readSamples[1][4] == validSamples[1][4], "Sample feature value from binary file should match the original one");
        ASSERT_LOG(readTargetOutputs[0] == validTargetOutputs[0], "Target output value from binary file should match the original one");
        ASSERT_LOG(readTargetOutputs[1] == validTargetOutputs[1], "Target output value from binary file should match the original one");
    }
    catch (std::exception& ex)
    {
        logger << "Error: Valid binary samples file reading should not have generated an exception." << std::endl
               << "Exception: " << std::endl << ex.what() << std::endl;
        bfs::remove_all(testDir);
        return -1;
    }
    try
    {
        // test wrong reading file format
        esvm.readSampleDataFile(wrongSampleFileName5, readSamples, readTargetOutputs, LIBSVM);
        logger << "Error: Reading binary formatted file as LIBSVM format should result in parsing failure." << std::endl;
        return -2;
    }
    catch (...) {}

    bfs::remove_all(testDir);
    return 0;
}

int test_runTimerExemplarSvmReadSampleFile(int nSamples, int nFeatures)
{
    ASSERT_LOG(nSamples > 0, "Number of samples must be greater than zero");
    ASSERT_LOG(nFeatures > 0, "Number of samples must be greater than zero");

    logstream logger(LOGGER_FILE);

    // Generate test samples file
    logger << "Generating dummy test samples file for timing evaluation..." << std::endl;
    std::string timingSampleFileName = "test_timing-samples.data";
    std::ofstream timingSampleFile(timingSampleFileName);    
    std::srand(0);
    for (int s = 0; s < nSamples; s++)
    {
        timingSampleFile << std::to_string(ESVM_NEGATIVE_CLASS);
        for (int f = 0; f < nFeatures; f++)
            timingSampleFile << " " << f + 1 << ":" << ((double)std::rand() / (double)RAND_MAX);
        timingSampleFile << " -1:0" << std::endl;
    }
    if (timingSampleFile.is_open()) timingSampleFile.close();

    // Start reading to evaluate timing
    ESVM esvm;
    std::vector<FeatureVector> samples;
    std::vector<int> targetOutputs;
    double t0 = (double)cv::getTickCount();
    esvm.readSampleDataFile(timingSampleFileName, samples, targetOutputs);
    double dt = ((double)cv::getTickCount() - t0) / cv::getTickFrequency();
    logger << "Elapsed time to read file with " << nSamples << " samples of " << nFeatures << " features: " << dt << "s" << std::endl;

    bfs::remove(timingSampleFileName);

    return 0;
}

#if 0
/**************************************************************************************************************************
TEST DEFINITION (individual IDs & corresponding 'person' ROIs)
        
    Targets:        single high quality still image for enrollment
            
        ID0011
        ID0012
        ID0013
        ID0016
        ID0020

    Non-Targets:    counter example video ROIs, cannot correspond to probe nor positive target individual
            
        ID0001      person_23
        ID0002      person_32
        ID0006      person_45
        ID0007      person_40
        ID0010      person_6  & person_44
        ID0017      person_41
        ID0018      person_16
        ID0019      person_9
        ID0024      person_34
        ID0025      person_33
        ID0027      person_42 & person_46
        ID0028      person_0
        ID0030      person_28            

    Probes:         positive and negative video ROIs for testing

        ID0004      person_26                   negative -
        ID0009      person_20 & person_25       negative -
        ID0011      person_15                   positive +
        ID0012      person_13                   positive +
        ID0013      person_7                    positive +            
        ID0016      person_19                   positive +
        ID0020      person_36                   positive +
        ID0023      person_37                   negative -
        ID0026      person_39                   negative -
        ID0029      person_18                   negative -
        
**************************************************************************************************************************/
int test_runSingleSamplePerPersonStillToVideo(cv::Size patchCounts)
{
    // ------------------------------------------------------------------------------------------------------------------------
    // window to display loaded images and stream for console+file output
    // ------------------------------------------------------------------------------------------------------------------------    
    cv::namedWindow(WINDOW_NAME);
    logstream logger(LOGGER_FILE);
    logger << "Starting single sample per person still-to-video test..." << std::endl;
    int nPatches = patchCounts.width*patchCounts.height;
    if (nPatches == 0) nPatches = 1;

    // ------------------------------------------------------------------------------------------------------------------------
    // C++ parameters
    // ------------------------------------------------------------------------------------------------------------------------
    /* Multiple negative samples as counter-example for each individual to enroll (CANNOT BE A PROBE NOR POSITIVE SAMPLE) */
    const int NB_NEGATIVE_IMAGES = 177;
    std::vector<cv::Mat> matNegativeSamples[NB_NEGATIVE_IMAGES];
    logger << "Loading negative training samples used for all enrollments..." << std::endl;
    /* --- ID0028 --- */
    matNegativeSamples[0]   = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_0/000190.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[1]   = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_0/000195.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[2]   = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_0/000200.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[3]   = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_0/000205.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[4]   = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_0/000225.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[5]   = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_0/000230.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[6]   = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_0/000235.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[7]   = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_0/000240.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[8]   = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_0/000245.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[9]   = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_0/000250.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    /* --- ID0010 --- */
    matNegativeSamples[10]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_6/000246.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[11]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_6/000247.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[12]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_6/000250.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[13]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_6/000255.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[14]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_6/000260.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[15]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_6/000265.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[16]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_6/000270.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[17]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_6/000275.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[18]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_6/000280.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[19]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_6/000285.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[20]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_6/000286.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[21]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_6/000290.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[22]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_6/000295.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[23]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_6/000300.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[24]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_44/000635.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[25]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_44/000640.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[26]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_44/000641.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[27]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_44/000645.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[28]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_44/000650.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[29]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_44/000656.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);    
    /* --- ID0019 --- */
    matNegativeSamples[30]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000280.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[31]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000285.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[32]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000290.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[33]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000295.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[34]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000300.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[35]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000305.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[36]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000310.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);    
    matNegativeSamples[37]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000315.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[38]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000320.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[39]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000325.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[40]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000330.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[41]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000335.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[42]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000340.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[43]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000345.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[44]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000350.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[45]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000355.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    /* --- ID0018 --- */
    matNegativeSamples[46]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_16/000350.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[47]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_16/000355.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[48]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_16/000360.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[49]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_16/000361.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[50]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_16/000365.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[51]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_16/000370.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    /* --- ID0001 --- */
    matNegativeSamples[52]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000435.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[53]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000440.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[54]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000445.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[55]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000450.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[56]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000455.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[57]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000460.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[58]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000465.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[59]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000470.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[60]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000475.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[61]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000480.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[62]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000485.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[63]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000490.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[64]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000495.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[65]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000500.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[66]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000505.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[67]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000510.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    /* --- ID0030 --- */
    matNegativeSamples[68]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_28/000465.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[69]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_28/000470.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[70]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_28/000475.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[71]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_28/000480.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[72]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_28/000481.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[73]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_28/000485.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[74]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_28/000490.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[75]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_28/000495.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    /* --- ID0002 --- */
    matNegativeSamples[76]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_32/000480.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[77]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_32/000485.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[78]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_32/000490.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[79]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_32/000495.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[80]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_32/000497.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[81]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_32/000500.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[82]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_32/000505.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[83]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_32/000510.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[84]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_32/000515.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[85]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_32/000520.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[86]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_32/000525.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    /* --- ID0025 --- */
    matNegativeSamples[87]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_33/000495.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[88]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_33/000500.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[89]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_33/000510.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[90]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_33/000515.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[91]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_33/000520.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[92]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_33/000525.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[93]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_33/000530.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[94]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_33/000535.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[95]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_33/000540.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    /* --- ID0024 --- */
    matNegativeSamples[96]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000525.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[97]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000530.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[98]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000535.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[99]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000540.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[100] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000545.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[101] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000550.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[102] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000555.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[103] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000560.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[104] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000565.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[105] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000570.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[106] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000575.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[107] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000580.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[108] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000585.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[109] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000590.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[110] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000595.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[111] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000600.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[112] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000605.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[113] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000610.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[114] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000615.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[115] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000620.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[116] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000625.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);       
    /* --- ID0007 --- */
    matNegativeSamples[117] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000606.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[118] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000610.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[119] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000615.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[120] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000620.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[121] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000625.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[122] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000630.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[123] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000635.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[124] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000640.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[125] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000645.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[126] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000646.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[127] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000650.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[128] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000651.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[129] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000655.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[130] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000656.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[131] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000660.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[132] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000665.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[133] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000670.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[134] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000675.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[135] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000680.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[136] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000685.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[137] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000690.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[138] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000700.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[139] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000705.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[140] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000710.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[141] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000715.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    /* --- ID0017 --- */
    matNegativeSamples[142] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_41/000611.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[143] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_41/000615.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[144] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_41/000635.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[145] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_41/000640.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[146] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_41/000645.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[147] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_41/000650.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[148] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_41/000655.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts); 
    matNegativeSamples[149] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_41/000660.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[150] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_41/000665.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[151] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_41/000670.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[152] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_41/000675.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[153] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_41/000680.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[154] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_41/000685.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    /* --- ID0027 --- */
    matNegativeSamples[155] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_42/000611.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[156] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_42/000615.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[157] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_46/000641.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[158] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_46/000645.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[159] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_46/000650.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    /* --- ID0006 --- */
    matNegativeSamples[160] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000650.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[161] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000655.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[162] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000660.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[163] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000665.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[164] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000666.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[165] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000670.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[166] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000675.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[167] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000676.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[168] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000680.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[169] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000681.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[170] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000685.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[171] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000686.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[172] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000690.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[173] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000695.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[174] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000700.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[175] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000705.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matNegativeSamples[176] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000710.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);

    /* Single positive training samples (one per enrollment) */
    const int NB_ENROLLMENT = 5;
    std::string targetName[NB_ENROLLMENT];
    std::vector<cv::Mat> matPositiveSamples[NB_ENROLLMENT];
    // Positive targets (same as Saman paper)
    targetName[0] = "ID0011";
    targetName[1] = "ID0012";
    targetName[2] = "ID0013";
    targetName[3] = "ID0016";
    targetName[4] = "ID0020";
    logger << "Loading single positive training samples..." << std::endl;
    // Deduct full ROI size using the patch size and quantity since positive sample is high quality (different dimension)
    cv::Size imSize = matNegativeSamples[0][0].size();
    imSize.width *= patchCounts.width;
    imSize.height *= patchCounts.height;
    // Get still reference images (color high quality neutral faces) 
    // filename format: "roi<ID#>.jpg"
    for (int i = 0; i < NB_ENROLLMENT; i++)    
        matPositiveSamples[i] = imPreprocess(refStillImagesPath + "roi" + targetName[i] + ".JPG", imSize, patchCounts, WINDOW_NAME, cv::IMREAD_COLOR);
    
    /* Testing probe samples */
    const int NB_PROBE_IMAGES = 96;
    std::string probeGroundThruth[NB_PROBE_IMAGES];
    std::vector<cv::Mat> matProbeSamples[NB_PROBE_IMAGES];
    logger << "Loading testing probe samples..." << std::endl;
    /* --- ID0013 --- */
    matProbeSamples[0]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_7/000255.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[1]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_7/000260.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[2]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_7/000265.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[3]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_7/000267.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[4]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_7/000270.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[5]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_7/000272.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[6]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_7/000275.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    for (int i = 0; i <= 6; i++) probeGroundThruth[i] = "ID0013";
    /* --- ID0012 --- */
    matProbeSamples[7]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_13/000320.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[8]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_13/000325.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[9]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_13/000330.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[10] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_13/000335.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[11] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_13/000340.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[12] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_13/000345.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[13] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_13/000350.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[14] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_13/000355.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[15] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_13/000360.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    for (int i = 7; i <= 15; i++) probeGroundThruth[i] = "ID0012";
    /* --- ID0011 --- */
    matProbeSamples[16] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_15/000350.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[17] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_15/000355.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[18] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_15/000360.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[19] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_15/000365.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[20] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_15/000370.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[21] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_15/000375.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[22] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_15/000377.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[23] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_15/000380.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[24] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_15/000385.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[25] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_15/000390.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    for (int i = 16; i <= 25; i++) probeGroundThruth[i] = "ID0011";
    /* --- ID0029 --- */
    matProbeSamples[26] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_18/000365.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[27] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_18/000370.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[28] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_18/000375.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[29] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_18/000380.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[30] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_18/000381.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[31] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_18/000385.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[32] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_18/000390.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[33] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_18/000395.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[34] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_18/000400.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[35] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_18/000401.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[36] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_18/000425.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[37] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_18/000430.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    for (int i = 26; i <= 37; i++) probeGroundThruth[i] = "ID0029";
    /* --- ID0016 --- */
    matProbeSamples[38] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_19/000400.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[39] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_19/000405.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[40] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_19/000406.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[41] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_19/000410.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[42] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_19/000415.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[43] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_19/000420.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[44] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_19/000425.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[45] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_19/000430.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[46] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_19/000435.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[47] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_19/000440.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[48] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_19/000445.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[49] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_19/000450.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[50] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_19/000455.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[51] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_19/000460.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[52] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_19/000465.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    for (int i = 38; i <= 52; i++) probeGroundThruth[i] = "ID0016";
    /* --- ID0009 --- */
    matProbeSamples[53] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_20/000410.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[54] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_20/000415.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[55] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_20/000420.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[56] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_20/000425.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[57] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_25/000441.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[58] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_25/000445.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[59] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_25/000450.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    for (int i = 53; i <= 59; i++) probeGroundThruth[i] = "ID0009";
    /* --- ID0004 --- */
    matProbeSamples[60] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_26/000447.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[61] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_26/000450.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[62] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_26/000455.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[63] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_26/000460.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[64] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_26/000465.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[65] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_26/000470.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[66] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_26/000475.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[67] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_26/000480.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[68] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_26/000485.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    for (int i = 60; i <= 68; i++) probeGroundThruth[i] = "ID0004";
    /* --- ID0020 --- */
    matProbeSamples[69] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_36/000540.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[70] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_36/000545.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[71] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_36/000550.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[72] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_36/000552.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[73] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_36/000555.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[74] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_36/000556.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[75] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_36/000560.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[76] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_36/000562.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[77] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_36/000565.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[78] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_36/000566.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[79] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_36/000570.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    for (int i = 69; i <= 79; i++) probeGroundThruth[i] = "ID0020";
    /* --- ID0023 --- */
    matProbeSamples[80] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_37/000541.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[81] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_37/000545.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[82] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_37/000550.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[83] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_37/000555.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[84] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_37/000561.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[85] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_37/000565.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[86] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_37/000570.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    for (int i = 80; i <= 86; i++) probeGroundThruth[i] = "ID0023";
    /* --- ID0026 --- */
    matProbeSamples[87] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_39/000566.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[88] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_39/000570.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[89] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_39/000575.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[90] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_39/000580.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[91] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_39/000583.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[92] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_39/000585.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[93] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_39/000590.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[94] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_39/000595.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[95] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_39/000600.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    for (int i = 87; i <= 95; i++) probeGroundThruth[i] = "ID0026";

    // Destroy viewing window not required anymore
    cv::destroyWindow(WINDOW_NAME);

    // ------------------------------------------------------------------------------------------------------------------------
    // Transform into MATLAB arrays
    // NB: vector and OpenCV Mat are zero-based, but MATLAB mwArray are one-based
    // ------------------------------------------------------------------------------------------------------------------------    
    std::vector<mwArray> models(nPatches);
    std::vector<mwArray> mwScores(nPatches);    
    std::vector<mwArray> mwNegativeSamples(nPatches);
    std::vector<mwArray> mwProbeSamples(nPatches);
    // Duplication required for unique positive per individual
    int NB_POSITIVE_DUPLICATION = 1; 
    std::vector< std::vector<mwArray> > mwPositiveSamples(NB_ENROLLMENT);
    // Conversion for Exemplar-SVM
    for (int p = 0; p < nPatches; p++)
    {
        logger << "Converting patches at index " << p << "..." << std::endl;
        mwNegativeSamples[p] = mwArray(NB_NEGATIVE_IMAGES, 1, mxCELL_CLASS);
        mwProbeSamples[p]    = mwArray(NB_PROBE_IMAGES, 1, mxCELL_CLASS);

        logger << "Converting positive training samples..." << std::endl;        
        for (int i = 0; i < NB_ENROLLMENT; i++)
        {        
            // Initialize vertor only on first patch for future calls
            if (p == 0) 
                mwPositiveSamples[i] = std::vector<mwArray>(nPatches);

            // Duplicate unique positive to generate a pool samples           
            mwPositiveSamples[i][p] = mwArray(NB_POSITIVE_DUPLICATION, 1, mxCELL_CLASS);
            mwArray dupPositive = convertCvToMatlabMat(matPositiveSamples[i][p]);
            for (int j = 0; j < NB_POSITIVE_DUPLICATION; j++)
                mwPositiveSamples[i][p].Get(1, j + 1).Set(dupPositive);
        }
        
        logger << "Converting negative training samples..." << std::endl;
        for (int i = 0; i < NB_NEGATIVE_IMAGES; i++)
            mwNegativeSamples[p].Get(1, i + 1).Set(convertCvToMatlabMat(matNegativeSamples[i][p]));
        
        logger << "Converting probe testing samples..." << std::endl;
        for (int i = 0; i < NB_PROBE_IMAGES; i++)
            mwProbeSamples[p].Get(1, i + 1).Set(convertCvToMatlabMat(matProbeSamples[i][p]));
    }



    //################################################################################ DEBUG
    /*cv::Mat img = imReadAndDisplay(refStillImagesPath + "roi" + targetName[0] + ".JPG", WINDOW_NAME, cv::IMREAD_COLOR);
    cv::cvtColor(img, img, CV_BGR2GRAY);
    cv::resize(img, img, imSize, 0, 0, cv::INTER_CUBIC);    
    cv::imwrite(refStillImagesPath + "roi" + targetName[0] + "_resized.JPG", img);    
    for (int p = 0; p < nPatches; p++)
    {
        std::string name = refStillImagesPath + "roi" + targetName[0] + "_patch" + std::to_string(p) + ".jpg";        
        cv::imwrite(name, matPositiveSamples[0][p]);
    }*/
    //################################################################################ DEBUG

    // ------------------------------------------------------------------------------------------------------------------------
    // Try Exemplar-SVM training and testing with single sample per person (SSPP) in still-to-video
    // ------------------------------------------------------------------------------------------------------------------------
    try
    {        
        //################################################################################ DEBUG
        /*
        logger << "DEBUG: " << std::endl
               << models.size() << std::endl
               << mwPositiveSamples.size() << std::endl
               << mwNegativeSamples.size() << std::endl
               << "Dims Pos sample: " << mwPositiveSamples[0].Get(1, 1).GetDimensions() << std::endl
               << "Dims nb positive: " << mwPositiveSamples[0].GetDimensions().ToString() << std::endl
               << "Data:" << std::endl        
               << mwPositiveSamples[0].ToString() << std::endl;
        for (int i = 0; i < 5; i++)
        {
            logger << "Data detail " << i << ":" << std::endl;
            logger << mwPositiveSamples[0].Get(1,i+1).ToString() << std::endl;
        }

        cv::Size is = matNegativeSamples[0][0].size();
        cv::Mat im = imReadAndDisplay(refStillImagesPath + "roi" + targetName[0] + ".JPG", WINDOW_NAME, cv::IMREAD_COLOR);
        logger << "Dims resize neg image: " << is << std::endl;
        logger << "Dims original image: " << im.size() << std::endl;
        cv::cvtColor(im, im, CV_BGR2GRAY);
        cv::resize(im, im, is, 0, 0, cv::INTER_CUBIC);
        logger << "Dims resized pos image: " << im.size() << std::endl;
        auto vIm = imSplitPatches(im, patchCounts);
        logger << "Pos nb patches: " << vIm.size() << std::endl;
        logger << "Dims patch 0: " << vIm[0].size() << std::endl;


        logger << mwPositiveSamples[0].ClassID() << std::endl;     // cell: 1, double: 6, uint8: 9
        logger << mwPositiveSamples[0].Get(1, 1).ClassID() << std::endl;
        const int size = 32 * 32;
        UINT8 data[size];
        mwPositiveSamples[0].Get(1, 1).GetData(data, size);
        std::string outData = "";
        for (int i = 0; i < size; i++)
        {
            outData += data[i];
            outData += +" ";
        }
        logger << outData << std::endl;
        
        logger << "DEBUG INFO" << std::endl
               << mwPositiveSamples.size() << std::endl
               << mwPositiveSamples[0].size() << std::endl
               << mwPositiveSamples[0][0].GetDimensions() << std::endl;
        */
        //################################################################################ DEBUG

        for (int i = 0; i < NB_ENROLLMENT; i++)
        {            
            logger << "Starting for individual " << i << ": " + targetName[i] << std::endl;
            double scoreFusion[NB_PROBE_IMAGES] = { 0 };
            for (int p = 0; p < nPatches; p++)
            {                
                logger << "Running Exemplar-SVM training..." << std::endl;
                esvm_train_individual(1, models[p], mwPositiveSamples[i][p], mwNegativeSamples[p], mwArray(targetName[i].c_str()));
                logger << "Running Exemplar-SVM testing..." << std::endl;
                esvm_test_individual(1, mwScores[p], models[p], mwProbeSamples[p]);
                double scores[NB_PROBE_IMAGES];
                mwScores[p].GetData(scores, NB_PROBE_IMAGES);
                for (int j = 0; j < NB_PROBE_IMAGES; j++)
                {
                    // score accumulation from patches with normalization
                    double normPatchScore = normalizeClassScoreToSimilarity(scores[j]);
                    scoreFusion[j] += normPatchScore;
                    std::string probeGT = (probeGroundThruth[j] == targetName[i] ? "positive" : "negative");
                    logger << "Score for patch " << p << " of probe " << j << " (" << probeGT << "): " << normPatchScore << std::endl;
                }
            }
            for (int j = 0; j < NB_PROBE_IMAGES; j++)
            {
                // average of score accumulation for fusion
                std::string probeGT = (probeGroundThruth[j] == targetName[i] ? "positive" : "negative");
                logger << "Score fusion of probe " << j << " (" << probeGT << "): " << scoreFusion[j] / nPatches << std::endl;               
            }
            logger << "Completed for individual " << i << ": " + targetName[i] << std::endl;
        }
        logger << "Success" << std::endl;
        return 0;
    }
    catch (const mwException& e)
    {
        logger << e.what() << std::endl;
        return -2;
    }
    catch (...)
    {
        logger << "Unexpected error thrown" << std::endl;
        return -3;
    }
}
#endif

/**************************************************************************************************************************
TEST DEFINITION        
    
    Enrolls the training target using their still image vs. non-targets of multiple video sequence from ChokePoint dataset.
    The enrolled individuals are represented by ensembles of Exemplar-SVM and are afterward tested using the probe videos.
    Classification performances are then evaluated each positive target vs. probe samples in term of FPR/TPR for ROC curbe.
**************************************************************************************************************************/
int test_runSingleSamplePerPersonStillToVideo_FullChokePoint(cv::Size imageSize, cv::Size patchCounts)
{
    /* Training Targets:        single high quality still image for enrollment */    
    /*std::vector<std::string> positivesID = { "0011", "0012", "0013", "0016", "0020" };*/  // same as Saman paper
    /// change to fit same positives as in SAMAN code
    std::vector<std::string> positivesID = { "0003", "0005", "0006", "0010", "0024" };
    /* Training Non-Targets:    as many video negatives as possible */    
    /*std::vector<std::string> negativesID = { "0001", "0002", "0006", "0007", "0010",
                                             "0017", "0018", "0019", "0024", "0025",
                                             "0027", "0028", "0030" };*/
    /// change to fit new positives/probes defined
    std::vector<std::string> negativesID = { "0001", "0002", "0007", "0009", "0011",
                                             "0013", "0014", "0016", "0017", "0018",
                                             "0019", "0020", "0021", "0022", "0025" };
    /* Testing Probes:          some video positives and negatives */    
    /*std::vector<std::string> probesID = { "0004", "0009", "0011", "0012", "0013",
                                          "0016", "0020", "0023", "0026", "0029" }; */
    /// change to include new positives defined, and not any of the negatives
    std::vector<std::string> probesID = { "0003", "0004", "0005", "0006", "0010",
                                          "0012", "0015", "0023", "0024", "0028" }; 
    /// not used:       0026, 0027, 0029, 0030
    /// doesn't exist:  0008

    // Display and output
    cv::namedWindow(WINDOW_NAME);
    logstream logger(LOGGER_FILE);
    size_t nPatches = patchCounts.width * patchCounts.height;
    if (nPatches == 0) nPatches = 1;    
    bool useHistEqual = false;
    logger << "Starting single sample per person still-to-video full ChokePoint test..." << std::endl
           << "   useSyntheticPositives:    " << TEST_USE_SYNTHETIC_GENERATION << std::endl
           << "   imageSize:                " << imageSize << std::endl
           << "   patchCounts:              " << patchCounts << std::endl
           << "   useHistEqual:             " << useHistEqual << std::endl;
    
    size_t nDescriptors = 0;
    std::vector<std::string> descriptorNames;
    #if ESVM_USE_HOG
    nDescriptors++;
    std::string descriptorHOG = "hog";
    descriptorNames.push_back(descriptorHOG);
    FeatureExtractorHOG hog;
    /*OLD PARAMS
    cv::Size patchSize = cv::Size(imageSize.width / patchCounts.width, imageSize.height / patchCounts.height);
    cv::Size hogBlock = cv::Size(patchSize.width / 2, patchSize.height / 2);
    cv::Size hogCell = cv::Size(hogBlock.width / 2, hogBlock.height / 2);
    int nBins = 8;
    */
    cv::Size patchSize = cv::Size(imageSize.width / patchCounts.width, imageSize.height / patchCounts.height);
    cv::Size blockSize = cv::Size(2, 2);
    cv::Size blockStride = cv::Size(2, 2);
    cv::Size cellSize = cv::Size(2, 2);
    int nBins = 3;
    hog.initialize(patchSize, blockSize, blockStride, cellSize, nBins);
    logger << "HOG feature extraction initialized..." << std::endl
           << "   imageSize:   " << imageSize << std::endl
           << "   patchSize:   " << patchSize << std::endl
           << "   blockSize:   " << blockSize << std::endl
           << "   blockStride: " << blockStride << std::endl
           << "   cellSize:    " << blockStride << std::endl
           << "   nBins:       " << nBins << std::endl;
    #endif/*ESVM_USE_HOG*/
    
    #if ESVM_USE_LBP
    nDescriptors++;
    std::string descriptorLBP = "lbp";
    descriptorNames.push_back(descriptorLBP);
    FeatureExtractorLBP lbp;
    int points = 8;
    int radius = 8;
    MappingType map = LBP_MAPPING_U2;
    lbp.initialize(points, radius, map);
    logger << "LBP feature extraction initialized..." << std::endl
           << "   imageSize: " << imageSize << std::endl
           << "   points:    " << points << std::endl
           << "   radius:    " << radius << std::endl
           << "   mapping:   " << lbp::MappingTypeStr[map] << std::endl;    
    #endif/*ESVM_USE_LBP*/  

    ASSERT_LOG(nDescriptors > 0, "At least one of the feature extraction method must be enabled");

    // Samples container with expected indexes usage
    /*
        NB:     When using the 'mvector' class, if only a single 'int' value is specified for size, it is assigned for each dimension.
                Using 'push_back' later on a deeper level of 'vector' will put the new values after the N default values would have been
                added because of the single 'int' dimension value.
                Dimensions should therefore be mentionned explicitely using an array of size for each 'vector' level, be initialized later
                as required with lower dimension 'mvector', or using a 'zero' dimension (empty).
    */
    size_t nPositives = positivesID.size();
    size_t nRepresentations = 1;                                                // single original enroll still
    size_t nDuplications = TEST_DUPLICATE_COUNT;
    size_t dimsGroundTruths[2] { nPositives, 0 };
    size_t dimsImgPositives[3] { nPositives, nRepresentations, nPatches };   
    xstd::mvector<2, int> probeGroundTruth(dimsGroundTruths);                   // [positive][probe](int)
    xstd::mvector<3, cv::Mat> matPositiveSamples(dimsImgPositives);             // [positive][representation][patch](Mat[x,y])
    xstd::mvector<2, cv::Mat> matNegativeSamples;                               // [negative][patch](Mat[x,y])
    xstd::mvector<2, cv::Mat> matProbeSamples;                                  // [probe][patch](Mat[x,y])     
    std::vector<std::string> negativeSamplesID;                                 // [negative](string)
    std::vector<std::string> probeSamplesID;                                    // [probe](string)

    // Add samples to containers
    logger << "Loading positives image for all test sequences..." << std::endl;
    for (size_t pos = 0; pos < nPositives; pos++)
    {        
        // Add additional positive representations as requested
        #if TEST_USE_SYNTHETIC_GENERATION
        
            // Get original positive image with preprocessing but without patches splitting
            cv::Mat img = imPreprocess(refStillImagesPath + "roiID" + positivesID[pos] + ".tif", imageSize, cv::Size(1, 1),
                                       useHistEqual, WINDOW_NAME, cv::IMREAD_GRAYSCALE)[0];
            // Get synthetic representations from original and apply patches splitting each one
            std::vector<cv::Mat> representations = imSyntheticGeneration(img);
            // Reinitialize sub-container for augmented representations using synthetic images
            nRepresentations = representations.size();
            size_t dimsRepresentation[2] { nRepresentations, nPatches };
            matPositiveSamples[pos] = xstd::mvector<2, cv::Mat>(dimsRepresentation);

            /// ############################################# #pragma omp parallel for
            for (size_t r = 0; r < nRepresentations; r++)
            {
                std::vector<cv::Mat> patches = imSplitPatches(representations[r], patchCounts);
                for (size_t p = 0; p < nPatches; p++)
                    matPositiveSamples[pos][r][p] = patches[p];
            }
        
        // Only original representation otherwise (no synthetic images)
        #else/*!TEST_USE_SYNTHETIC_GENERATION*/

            std::vector<cv::Mat> patches = imPreprocess(refStillImagesPath + "roiID" + positivesID[pos] + ".tif", imageSize, patchCounts,
                                                        useHistEqual, WINDOW_NAME, cv::IMREAD_GRAYSCALE);
            for (size_t p = 0; p < nPatches; p++)
                matPositiveSamples[pos][0][p] = patches[p];
        
        #endif/*TEST_USE_SYNTHETIC_GENERATION*/
    }
    
    /// ################################################################################ DEBUG DISPLAY POSITIVES (+SYNTH)
    /*
    logger << "SHOWING DEBUG POSITIVE SAMPLES" << std::endl;
    for (int i = 0; i < nPositives; i++)
    {
        for (int j = 0; j < matPositiveSamples[i].size(); j++)
        {
            for (int k = 0; k < matPositiveSamples[i][j].size(); k++)
            {
                cv::imshow(WINDOW_NAME, matPositiveSamples[i][j][k]);
                cv::waitKey(500);
            }
        }
    }
    logger << "DONE SHOWING DEBUG POSITIVE SAMPLES" << std::endl;
    */
    // Destroy viewing window not required anymore    
    // cv::destroyWindow(WINDOW_NAME);
    /// ################################################################################ DEBUG

    logger << "Executing feature extraction of positive images for all test sequences..." << std::endl
           << "   nPositives:       " << nPositives << std::endl
           << "   nPatches:         " << nPatches << std::endl
           << "   nRepresentations: " << nRepresentations << std::endl
           << "   nDuplications:    " << nDuplications << std::endl
           << "   nDescriptors:     " << nDescriptors << std::endl;

    // Containers for feature vectors extracted from samples
    size_t dimsPositives[4] = { nPositives, nPatches, nDescriptors, nRepresentations }; // note: 'mshape' fails in this case, size_t array works...
    size_t dimsESVM[3] { nPositives, nPatches, nDescriptors };
    xstd::mvector<3, ESVM> esvmModels(dimsESVM);                            // [target][patch][descriptor](ESVM)
    xstd::mvector<4, FeatureVector> fvPositiveSamples(dimsPositives);       // [target][patch][descriptor][positive](FeatureVector)
    xstd::mvector<3, FeatureVector> fvNegativeSamples;                      // [patch][descriptor][negative](FeatureVector)
    xstd::mvector<3, FeatureVector> fvProbeSamples;                         // [patch][descriptor][probe](FeatureVector)

    // Convert unique positive samples (or with synthetic representations)
    /// ################################################## #pragma omp parallel for    
    for (size_t pos = 0; pos < nPositives; pos++)
    {                
        /// ################################################## #pragma omp parallel for
        for (size_t p = 0; p < nPatches; p++)
        {
            for (size_t d = 0; d < nDescriptors; d++)
            {                
                /// ################################################## #pragma omp parallel for
                for (int r = 0; r < nRepresentations; r++)
                {
                    // switch to (i,p,d,r) order for (patch,feature)-based training of sample representations
                    #if ESVM_USE_HOG
                    if (descriptorNames[d] == descriptorHOG)
                        fvPositiveSamples[pos][p][d][r] = hog.compute(matPositiveSamples[pos][r][p]);
                    #endif/*ESVM_USE_HOG*/
                    #if ESVM_USE_LBP
                    if (descriptorNames[d] == descriptorLBP)
                        fvPositiveSamples[pos][p][d][r] = lbp.compute(matPositiveSamples[pos][r][p]);                    
                    #endif/*ESVM_USE_LBP*/
                }

                /// ################################################################################ DEBUG DISPLAY FEATURES
                /*
                logger << "Positive " << positivesID[pos] << "-patch" << std::to_string(p) << ": " 
                       << featuresToVectorString(fvPositiveSamples[pos][p][d][0]) << std::endl;
                */
                /// ################################################################################ DEBUG DISPLAY FEATURES

                /// ################################################## 
                // DUPLICATE EVEN MORE REPRESENTATIONS TO HELP LIBSVM PROBABILITY ESTIMATES CROSS-VALIDATION
                // Add x-times the number of representations
                if (nDuplications > 1)
                    for (size_t r = 0; r < nRepresentations; r++)
                        for (size_t dup = 1; dup < nDuplications; dup++)
                            fvPositiveSamples[pos][p][d].push_back(fvPositiveSamples[pos][p][d][r]);
            }
        }
    }
    if (nDuplications > 1)
        nRepresentations *= nDuplications;
    for (size_t d = 0; d < nDescriptors; d++)
        logger << "Features dimension (" + descriptorNames[d] + "): " << fvPositiveSamples[0][0][d][0].size() << std::endl;

    // Tests divided per sequence information according to selected mode
    std::vector<PORTAL_TYPE> types = { ENTER, LEAVE };
    bfs::directory_iterator endDir;
    std::string seq;
    for (int sn = 1; sn <= SESSION_QUANTITY; sn++)
    {
        #if TEST_CHOKEPOINT_SEQUENCES_MODE == 0
        cv::namedWindow(WINDOW_NAME);
        #endif/*TEST_CHOKEPOINT_SEQUENCES_MODE*/

        for (int pn = 1; pn <= PORTAL_QUANTITY; pn++) {
        for (auto it = types.begin(); it != types.end(); ++it) {
        for (int cn = 1; cn <= CAMERA_QUANTITY; cn++)
        {     
            #if TEST_CHOKEPOINT_SEQUENCES_MODE == 1
            cv::namedWindow(WINDOW_NAME);
            // Reset vectors for next test sequences                    
            matNegativeSamples.clear();
            matProbeSamples.clear();
            negativeSamplesID.clear();
            probeSamplesID.clear();            
            for (size_t pos = 0; pos < nPositives; pos++)
                probeGroundTruth[pos].clear();
            #endif/*TEST_CHOKEPOINT_SEQUENCES_MODE*/

            seq = buildChokePointSequenceString(pn, *it, sn, cn);
            logger << "Loading negative and probe images for sequence " << seq << "..." << std::endl;
            #if TEST_CHOKEPOINT_SEQUENCES_MODE == 0
            seq = "S" + std::to_string(sn);
            #endif/*TEST_CHOKEPOINT_SEQUENCES_MODE*/

            // Add ROI to corresponding sample vectors according to individual IDs            
            for (int id = 1; id <= INDIVIDUAL_QUANTITY; id++)
            {
                std::string dirPath = roiChokePointCroppedFacePath + buildChokePointSequenceString(pn, *it, sn, cn, id) + "/";
                if (bfs::is_directory(dirPath))
                {
                    for (bfs::directory_iterator itDir(dirPath); itDir != endDir; ++itDir)
                    {
                        if (bfs::is_regular_file(*itDir) && itDir->path().extension() == ".pgm")
                        {
                            std::string strID = buildChokePointIndividualID(id);
                            if (contains(negativesID, strID))
                            {
                                size_t neg = matNegativeSamples.size();
                                matNegativeSamples.push_back(xstd::mvector<1, cv::Mat>(nPatches));
                                std::vector<cv::Mat> patches = imPreprocess(itDir->path().string(), imageSize, patchCounts,
                                                                            useHistEqual, WINDOW_NAME, cv::IMREAD_GRAYSCALE);
                                for (size_t p = 0; p < nPatches; p++)
                                    matNegativeSamples[neg][p] = patches[p];

                                negativeSamplesID.push_back(strID);
                            }
                            else if (contains(probesID, strID))
                            {
                                size_t prb = matProbeSamples.size();
                                matProbeSamples.push_back(xstd::mvector<1, cv::Mat>(nPatches));
                                std::vector<cv::Mat> patches = imPreprocess(itDir->path().string(), imageSize, patchCounts,
                                                                            useHistEqual, WINDOW_NAME, cv::IMREAD_GRAYSCALE);
                                for (size_t p = 0; p < nPatches; p++)
                                    matProbeSamples[prb][p] = patches[p];

                                probeSamplesID.push_back(strID);
                                for (size_t pos = 0; pos < nPositives; pos++)
                                    probeGroundTruth[pos].push_back(strID == positivesID[pos] ? ESVM_POSITIVE_CLASS : ESVM_NEGATIVE_CLASS);                                
                            } 
                        }                  
                    }
                }                        
            }

        // Add end of loops if sequences must be combined per session (accumulate cameras and scenes)
        #if TEST_CHOKEPOINT_SEQUENCES_MODE == 0
        } } }
        #endif/*TEST_CHOKEPOINT_SEQUENCES_MODE*/
        
            // Destroy viewing window not required while training/testing is in progress
            cv::destroyWindow(WINDOW_NAME);

            // Validation of negatives and probe samples
            size_t nProbes = matProbeSamples.size();
            size_t nNegatives = matNegativeSamples.size();
            ASSERT_LOG(negativeSamplesID.size() == nNegatives, "Mismatch between negative samples count and IDs");
            ASSERT_LOG(probeSamplesID.size() == nProbes, "Mismatch between probe samples count and IDs");
            for (int pos = 0; pos < nPositives; pos++)
                ASSERT_LOG(!contains(negativeSamplesID, positivesID[pos]), "Positive ID found within negative samples ID");  
            for (int neg = 0; neg < negativesID.size(); neg++)
                ASSERT_LOG(!contains(probeSamplesID, negativesID[neg]), "Negative ID found within probe samples ID");            
            for (int prb = 0; prb < probesID.size(); prb++)
                ASSERT_LOG(!contains(negativeSamplesID, probesID[prb]), "Probe ID found within negative samples ID");

            // Feature extraction of negatives and probes
            logger << "Executing feature extraction of negative and probe samples (total negatives: " << nNegatives
                   << ", total probes: " << nProbes << ")..." << std::endl;
            /// ############################################# #pragma omp parallel for
                     
            // Initialize feature vector containers now that the number of samples is known
            size_t dimsNegatives[3] = { nPatches, nDescriptors, nNegatives };
            size_t dimsProbes[3] = { nPatches, nDescriptors, nProbes };
            fvNegativeSamples = xstd::mvector<3, FeatureVector>(dimsNegatives);
            fvProbeSamples = xstd::mvector<3, FeatureVector>(dimsProbes);

            // Populate feature vector containers according to number of found negative/probe sample images
            for (size_t p = 0; p < nPatches; p++)
            {    
                for (size_t d = 0; d < nDescriptors; d++)
                {
                    // switch to (p,d,i) order for patch-based training
                    /// ############################################# #pragma omp parallel for
                    for (size_t neg = 0; neg < nNegatives; neg++)
                    {
                        #if ESVM_USE_HOG
                        if (descriptorNames[d] == descriptorHOG)
                            fvNegativeSamples[p][d][neg] = hog.compute(matNegativeSamples[neg][p]);
                        #endif/*ESVM_USE_HOG*/
                        #if ESVM_USE_LBP
                        if (descriptorNames[d] == descriptorLBP)
                            fvNegativeSamples[p][d][neg] = lbp.compute(matNegativeSamples[neg][p]);
                        #endif/*ESVM_USE_LBP*/
                    }
                    /// ############################################# #pragma omp parallel for
                    for (size_t prb = 0; prb < nProbes; prb++)
                    {
                        #if ESVM_USE_HOG
                        if (descriptorNames[d] == descriptorHOG)
                            fvProbeSamples[p][d][prb] = hog.compute(matProbeSamples[prb][p]);
                        #endif/*ESVM_USE_HOG*/
                        #if ESVM_USE_LBP
                        if (descriptorNames[d] == descriptorLBP)
                            fvProbeSamples[p][d][prb] = lbp.compute(matProbeSamples[prb][p]);
                        #endif/*ESVM_USE_LBP*/
                    }
                }
            }

            // Enroll positive individuals with Ensembles of Exemplar-SVM
            logger << "Starting enrollment for sequence: " << seq << "..." << std::endl;

            // Feature vector normalization
            #if !TEST_FEATURES_NORMALIZATION_MODE
            logger << "Skipping features normalization" << std::endl;            
            #else // Prepare some containers employed by each normalization method
            size_t dimsAllVectors[3]{ nPatches, nDescriptors, nPositives * nRepresentations + nNegatives + nProbes };
            size_t dimsMinMax[2]{ nDescriptors, nPatches };
            xstd::mvector<3, FeatureVector> allFeatureVectors(dimsAllVectors);      // [patch][descriptor][sample](FeatureVector)            
            xstd::mvector<2, FeatureVector> minFeaturesCumul(dimsMinMax);           // [descriptor][patch](FeatureVector)
            xstd::mvector<2, FeatureVector> maxFeaturesCumul(dimsMinMax);           // [descriptor][patch](FeatureVector)
            #endif/*TEST_FEATURES_NORMALIZATION_MODE*/                        

            // Specific min/max containers according to methods
            #if TEST_FEATURES_NORMALIZATION_MODE == 1       // Per feature and per patch normalization
            logger << "Searching feature normalization values (per feature, per patch)..." << std::endl;
            #elif TEST_FEATURES_NORMALIZATION_MODE == 2     // Per feature and across patches normalization
            logger << "Searching feature normalization values (per feature, across patches)..." << std::endl;
            std::vector<FeatureVector> minFeatures(nDescriptors);                   // [descriptor](FeatureVector)
            std::vector<FeatureVector> maxFeatures(nDescriptors);                   // [descriptor](FeatureVector)
            #elif TEST_FEATURES_NORMALIZATION_MODE == 3     // Across features and across patches normalization
            logger << "Searching feature normalization values (across features, across patches)..." << std::endl;
            std::vector<double> minFeatures(nDescriptors, DBL_MAX);                 // [descriptor](double)
            std::vector<double> maxFeatures(nDescriptors, -DBL_MAX);                // [descriptor](double)            
            #endif/*ESVM_USE_FEATURES_NORMALIZATION == (1|2|3)*/

            // Accumulate all positive/negative/probes samples to find min/max features according to normalization mode
            for (size_t d = 0; d < nDescriptors; d++)
            {                
                for (size_t p = 0; p < nPatches; p++)
                {
                    size_t s = 0;  // Sample index
                    for (size_t pos = 0; pos < nPositives; pos++)
                        for (size_t r = 0; r < nRepresentations; r++)
                            allFeatureVectors[p][d][s++] = fvPositiveSamples[pos][p][d][r];
                    for (size_t neg = 0; neg < nNegatives; neg++)
                        allFeatureVectors[p][d][s++] = fvNegativeSamples[p][d][neg];
                    for (size_t prb = 0; prb < nProbes; prb++)
                        allFeatureVectors[p][d][s++] = fvProbeSamples[p][d][prb];
                    
                    // Find min/max features according to normalization mode
                    findMinMaxFeatures(allFeatureVectors[p][d], &(minFeaturesCumul[d][p]), &(maxFeaturesCumul[d][p]));
                    #if TEST_FEATURES_NORMALIZATION_MODE == 1   // Per feature and per patch normalization
                    logger << "Found min/max features for (descriptor,patch) (" << descriptorNames[d] << "," << p << "):" << std::endl
                           << "   MIN: " << featuresToVectorString(minFeaturesCumul[d][p]) << std::endl
                           << "   MAX: " << featuresToVectorString(minFeaturesCumul[d][p]) << std::endl;                   
                    #endif/*ESVM_USE_FEATURES_NORMALIZATION == 1*/
                }
                #if TEST_FEATURES_NORMALIZATION_MODE == 2       // Per feature and across patches normalization
                FeatureVector dummyFeatures(minFeaturesCumul[d][0].size());
                findMinMaxFeatures(minFeaturesCumul[d], &(minFeatures[d]), &dummyFeatures);
                findMinMaxFeatures(maxFeaturesCumul[d], &dummyFeatures, &(maxFeatures[d]));
                logger << "Found min/max features for descriptor '" << descriptorNames[d] << "':" << std::endl
                       << "   MIN: " << featuresToVectorString(minFeatures[d]) << std::endl
                       << "   MAX: " << featuresToVectorString(maxFeatures[d]) << std::endl;
                #elif TEST_FEATURES_NORMALIZATION_MODE == 3     // Across features and across patches normalization
                double dummyMinMax;
                findMinMaxOverall(minFeaturesCumul[d], &(minFeatures[d]), &dummyMinMax);
                findMinMaxOverall(maxFeaturesCumul[d], &dummyMinMax, &(maxFeatures[d]));
                logger << "Found min/max features for descriptor '" << descriptorNames[d] << "':" << std::endl
                       << "   MIN: " << minFeatures[d] << std::endl
                       << "   MAX: " << maxFeatures[d] << std::endl;
                #endif/*ESVM_USE_FEATURES_NORMALIZATION == (2|3)*/
            }
            
            #if TEST_FEATURES_NORMALIZATION_MODE == 1   // Per feature and per patch normalization
            logger << "Applying features normalization (per feature, per patch)..." << std::endl;
            #elif TEST_FEATURES_NORMALIZATION_MODE == 2 // Per feature and across patches normalization
            logger << "Applying features normalization (per feature, across patches)..." << std::endl;
            #elif TEST_FEATURES_NORMALIZATION_MODE == 3 // Across features and across patches normalization
            logger << "Applying features normalization (across feature, across patches)..." << std::endl;
            #endif/*ESVM_USE_FEATURES_NORMALIZATION == (1|2|3)*/
            FeatureVector minNorm, maxNorm;
            for (size_t p = 0; p < nPatches; p++)
            {    
                for (size_t d = 0; d < nDescriptors; d++)
                {
                    #if TEST_FEATURES_NORMALIZATION_MODE == 1   // Per feature and per patch normalization
                    minNorm = minFeaturesCumul[d][p];
                    maxNorm = minFeaturesCumul[d][p];
                    #elif TEST_FEATURES_NORMALIZATION_MODE == 2 // Per feature and across patches normalization
                    minNorm = minFeatures[d];
                    maxNorm = minFeatures[d];
                    #elif TEST_FEATURES_NORMALIZATION_MODE == 3 // Across features and across patches normalization
                    int nFeatures = fvPositiveSamples[0][0][0][0].size();
                    minNorm = FeatureVector(nFeatures, minFeatures[d]);
                    maxNorm = FeatureVector(nFeatures, maxFeatures[d]);
                    #endif/*ESVM_USE_FEATURES_NORMALIZATION == (1|2|3)*/

                    for (size_t pos = 0; pos < nPositives; pos++)
                        for (size_t r = 0; r < nRepresentations; r++)
                            fvPositiveSamples[pos][p][d][r] = normalizeMinMaxPerFeatures(fvPositiveSamples[pos][p][d][r], minNorm, maxNorm);
                    for (size_t neg = 0; neg < nNegatives; neg++)
                        fvNegativeSamples[p][d][neg] = normalizeMinMaxPerFeatures(fvNegativeSamples[p][d][neg], minNorm, maxNorm);
                    for (size_t prb = 0; prb < nProbes; prb++)
                        fvProbeSamples[p][d][prb] = normalizeMinMaxPerFeatures(fvProbeSamples[p][d][prb], minNorm, maxNorm);
                }
            }

            // ESVM samples files for each (sequence,positive,feature-extraction,train/test,patch) combination
            #if TEST_WRITE_DATA_FILES
            logger << "Writing ESVM train/test samples files..." << std::endl;
            for (size_t p = 0; p < nPatches; p++)
            {
                std::string strPatch = std::to_string(p);
                for (size_t d = 0; d < nDescriptors; d++)
                {
                    for (size_t pos = 0; pos < nPositives; pos++)
                    {        
                        std::string fileTemplate = "chokepoint-" + seq + "-id" + positivesID[pos] + "-" + descriptorNames[d] + "-patch" + strPatch;
                        std::string trainFileName = fileTemplate + "-train.data";
                        std::string testFileName = fileTemplate + "-test.data";
                        logger << "   Writing ESVM files:" << std::endl
                               << "      '" << trainFileName << "'" << std::endl
                               << "      '" << testFileName << "'" << std::endl;

                        std::ofstream trainFile(trainFileName);
                        std::ofstream testFile(testFileName);
                                        
                        #if TEST_USE_OTHER_POSITIVES_AS_NEGATIVES
                        // Add other gallery positives than the current one as additional negative representations (counter examples)
                        for (size_t galleryPos = 0; galleryPos < nPositives; galleryPos++)
                            for (size_t r = 0; r < nRepresentations; r++)
                        {
                            int gt = (pos == galleryPos ? ESVM_POSITIVE_CLASS : ESVM_NEGATIVE_CLASS);
                            trainFile << featuresToSvmString(fvPositiveSamples[galleryPos][p][d][r], gt) << std::endl;
                        }
                        #else/*TEST_USE_OTHER_POSITIVES_AS_NEGATIVES*/
                        // Add only corresponding positive representations
                        for (size_t r = 0; r < nRepresentations; r++)
                            trainFile << featuresToSvmString(fvPositiveSamples[pos][p][d][r], ESVM_POSITIVE_CLASS) << std::endl;
                        #endif/*TEST_USE_OTHER_POSITIVES_AS_NEGATIVES*/
                        for (size_t neg = 0; neg < nNegatives; neg++)
                            trainFile << featuresToSvmString(fvNegativeSamples[p][d][neg], ESVM_NEGATIVE_CLASS) << std::endl;
                        for (size_t prb = 0; prb < nProbes; prb++)
                            testFile << featuresToSvmString(fvProbeSamples[p][d][prb], probeGroundTruth[pos][prb]) << std::endl;
                    }
                }
            }
            #endif/*TEST_WRITE_DATA_FILES*/
            
            // Classifiers training and testing
            logger << "Starting classification training/testing..." << std::endl;
            for (size_t pos = 0; pos < nPositives; pos++)
            {                        
                logger << "Starting for individual " << pos << ": " + positivesID[pos] << std::endl;
                std::vector<double> fusionScores(nProbes, 0.0); 
                std::vector<double> combinedScores(nProbes, 0.0);
                std::vector<double> combinedScoresRaw(nProbes, 0.0);
                for (size_t d = 0; d < nDescriptors; d++)
                {     
                    std::vector<double> descriptorScores(nProbes, 0.0);
                    for (size_t p = 0; p < nPatches; p++)
                    {                        
                        try
                        {
                            logger << "Running Exemplar-SVM training..." << std::endl;
                            esvmModels[pos][p][d] = ESVM(fvPositiveSamples[pos][p][d], fvNegativeSamples[p][d], positivesID[pos]);

                            #if TEST_WRITE_DATA_FILES
                            std::string esvmModelFile = "chokepoint-" + seq + "-id" + positivesID[pos] + "-" +
                                                        descriptorNames[d] + "-patch" + std::to_string(p) + ".model";
                            logger << "Saving Exemplar-SVM model to file..." << std::endl;
                            bool isSaved = esvmModels[pos][p][d].saveModelFile(esvmModelFile);
                            logger << std::string(isSaved ? "Saved" : "Failed to save") + 
                                      " Exemplar-SVM model to file: '" + esvmModelFile + "'" << std::endl;                            
                            #endif/*TEST_WRITE_DATA_FILES*/
                        }
                        catch (const std::exception& e)
                        {
                            logger << e.what() << std::endl;
                            return -2;
                        }
                        catch (...)
                        {
                            logger << "Unexpected error thrown" << std::endl;
                            return -3;
                        }

                        logger << "Running Exemplar-SVM testing..." << std::endl;
                        std::vector<double> patchScores(nProbes, 0.0);
                        // test probes per patch and normalize scores
                        for (size_t prb = 0; prb < nProbes; prb++)
                            patchScores[prb] = esvmModels[pos][p][d].predict(fvProbeSamples[p][d][prb]);
                        std::vector<double> patchScoresNorm = normalizeMinMaxClassScores(patchScores);

                        /*########################################### DEBUG */
                        std::string strPatch = std::to_string(p);
                        logger << "PATCH " + strPatch + " SCORES:      " << featuresToVectorString(patchScores) << std::endl;
                        logger << "PATCH " + strPatch + " SCORES NORM: " << featuresToVectorString(patchScoresNorm) << std::endl;
                        /*########################################### DEBUG */                    
                    
                        for (size_t prb = 0; prb < nProbes; prb++)
                        {                            
                            descriptorScores[prb] += patchScoresNorm[prb];  // accumulation with normalized scores for patch-based score fusion
                            combinedScores[prb] += patchScoresNorm[prb];    // accumulation of scores for (patch,descriptor)-based score fusion
                            combinedScoresRaw[prb] += patchScores[prb];     // accumulation of all probe scores without any pre-fusion normalization

                            std::string probeGT = (probeSamplesID[prb] == positivesID[pos] ? "positive" : "negative");
                            logger << "Score for patch " << p << " of probe " << prb << " (ID" << probeSamplesID[prb] << ", "
                                   << probeGT << "): " << patchScoresNorm[prb] << std::endl;
                        }
                    }
                    for (size_t prb = 0; prb < nProbes; prb++)
                    {
                        // average of score accumulation for fusion per patch
                        descriptorScores[prb] /= (double)nPatches;
                        // accumulation with normalized patch-fusioned scores for descriptor-based score fusion
                        fusionScores[prb] = descriptorScores[prb];
                        std::string probeGT = (probeGroundTruth[pos][prb] > 0 ? "positive" : "negative");
                        logger << "Score for descriptor " << descriptorNames[d] << " (patch-fusion) of probe " << prb
                               << " (ID" << probeSamplesID[prb] << ", " << probeGT << "): " << descriptorScores[prb] << std::endl;
                    }
                    logger << "Performance evaluation for patch-based score fusion for '" + descriptorNames[d] + "' descriptor:" << std::endl;
                    eval_PerformanceClassificationScores(descriptorScores, probeGroundTruth[pos]);
                }

                int nCombined = nDescriptors * nPatches;
                for (size_t prb = 0; prb < nProbes; prb++)
                {
                    // average of score accumulation for fusion per descriptor
                    std::string probeGT = (probeGroundTruth[pos][prb] > 0 ? "positive" : "negative");
                    fusionScores[prb] /= (double)nDescriptors;
                    combinedScores[prb] /= (double)nCombined;
                    combinedScoresRaw[prb] /= (double)nCombined;
                    
                    logger << "Score fusion (descriptor,patch) of probe " << prb << " (ID" << probeSamplesID[prb] << ", "
                           << probeGT << "): " << fusionScores[prb] << std::endl;
                }

                // Normalization of scores post-fusion
                std::vector<double> combinedScoresNorm = normalizeMinMaxClassScores(combinedScoresRaw);
                logger << "Score fusion (descriptor,patch) with post-fusion normalization for '" << positivesID[pos] << "':" << std::endl
                       << featuresToVectorString(combinedScoresNorm) << std::endl;

                /*########################################### DEBUG */
                logger << "SCORE FUSION: " << featuresToVectorString(fusionScores) << std::endl;
                /*########################################### DEBUG */

                // Evaluate results
                logger << "Performance evaluation for sequential patch-based + descriptor-based score fusion:" << std::endl;
                eval_PerformanceClassificationScores(fusionScores, probeGroundTruth[pos]);
                logger << "Performance evaluation for combined (patch,descriptor)-based score fusion:" << std::endl;
                eval_PerformanceClassificationScores(combinedScores, probeGroundTruth[pos]);
                logger << "Performance evaluation for combined (patch,descriptor)-based norm scores post-fusion:" << std::endl;
                eval_PerformanceClassificationScores(combinedScoresNorm, probeGroundTruth[pos]);

                logger << "Completed for individual " << pos << ": " + positivesID[pos] << std::endl;
            }

            #if TEST_CHOKEPOINT_SEQUENCES_MODE == 0
            logger << "Completed for sequence: S" << sn << "..." << std::endl;
            #elif TEST_CHOKEPOINT_SEQUENCES_MODE == 1
            logger << "Completed for sequence: " << seq << std::endl;
            #endif/*TEST_CHOKEPOINT_SEQUENCES_MODE*/            
        
        // Add end of loops if sequences must be separated per scene
        #if TEST_CHOKEPOINT_SEQUENCES_MODE == 1
        } } }
        #endif/*TEST_CHOKEPOINT_SEQUENCES_MODE*/
            
    } // End session loop 

    logger << "Test complete" << std::endl;
    return 0;
}

/**************************************************************************************************************************
TEST DEFINITION
    
    Similar procedure as in 'test_runSingleSamplePerPersonStillToVideo_FullChokePoint' but using pre-computed feature
    vectors stored in the data files.

    NB:
        Vectors depend on the configuration of images, patches, data duplication, feature extraction method, etc.
        Changing any configuration will require new data file to be generated by running "FullChokePoint" at least once.
**************************************************************************************************************************/
int test_runSingleSamplePerPersonStillToVideo_DataFiles_WholeImage()
{
    logstream logger(LOGGER_FILE);

    std::vector< std::vector< std::string > > filenames;    // list if { TRAIN/TEST/ID }   
    std::string dataFileDir = "data_ChokePoint_64x64_HOG-LBP-descriptors_whole-image/";
    #if ESVM_USE_HOG
    filenames.push_back({ dataFileDir + "chokepoint-S1-id0011-hog-train.data", dataFileDir + "chokepoint-S1-id0011-hog-test.data", "id0011" });
    filenames.push_back({ dataFileDir + "chokepoint-S2-id0011-hog-train.data", dataFileDir + "chokepoint-S2-id0011-hog-test.data", "id0011" });
    filenames.push_back({ dataFileDir + "chokepoint-S3-id0011-hog-train.data", dataFileDir + "chokepoint-S3-id0011-hog-test.data", "id0011" });
    filenames.push_back({ dataFileDir + "chokepoint-S4-id0011-hog-train.data", dataFileDir + "chokepoint-S4-id0011-hog-test.data", "id0011" });
    #elif ESVM_USE_LBP
    filenames.push_back({ dataFileDir + "chokepoint-S1-id0011-lbp-train.data", dataFileDir + "chokepoint-S1-id0011-lbp-test.data", "id0011" });
    filenames.push_back({ dataFileDir + "chokepoint-S2-id0011-lbp-train.data", dataFileDir + "chokepoint-S2-id0011-lbp-test.data", "id0011" });
    filenames.push_back({ dataFileDir + "chokepoint-S3-id0011-lbp-train.data", dataFileDir + "chokepoint-S3-id0011-lbp-test.data", "id0011" });
    filenames.push_back({ dataFileDir + "chokepoint-S4-id0011-lbp-train.data", dataFileDir + "chokepoint-S4-id0011-lbp-test.data", "id0011" });
    #endif/* ESVM_USE_HOG || ESVM_USE_LBP */

    for (auto itFileNames = filenames.begin(); itFileNames != filenames.end(); ++itFileNames)
    {
        std::string trainFileName = (*itFileNames)[0];
        std::string testFileName = (*itFileNames)[1];
        std::string id = (*itFileNames)[2];
               
        // Train/test ESVM from files
        logger << "Training ESVM with data file: '" << trainFileName << "'" << std::endl;
        ESVM esvm = ESVM(trainFileName, id);
        logger << "Testing ESVM with data file: '" << testFileName << "'" << std::endl;
        std::vector<int> probeGroundTruths;
        std::vector<double> scores = esvm.predict(testFileName, &probeGroundTruths);
        std::vector<double> normScores = normalizeMinMaxClassScores(scores);
        for (int prb = 0; prb < scores.size(); prb++)
        {
            std::string probeGT = (probeGroundTruths[prb] > 0 ? "positive" : "negative");
            logger << "Score for probe " << prb << " (" << probeGT << "): " << scores[prb] << " | normalized: " << normScores[prb] << std::endl;
        }

        // Evaluate results
        eval_PerformanceClassificationScores(normScores, probeGroundTruths);
    }

    logger << "Test complete" << std::endl;
    return 0;
}

/**************************************************************************************************************************
TEST DEFINITION
    
    Similar procedure as in 'test_runSingleSamplePerPersonStillToVideo_FullChokePoint' but using pre-computed feature
    vectors stored in the data files.

    This test allows score fusion first for patch-based files, and then for descriptor-based files.
            
        S_pos* = ∑_d [ ∑_p [ s_(p,d) ] / N_p ] / N_d     ∀pos positives (targets), ∀d descriptor, ∀p patches

    NB:
        Vectors depend on the configuration of images, patches, data duplication, feature extraction method, etc.
        Changing any configuration will require new data files to be generated by running "FullChokePoint" at least once.
**************************************************************************************************************************/
int test_runSingleSamplePerPersonStillToVideo_DataFiles_DescriptorAndPatchBased(int nPatches)
{
    ASSERT_LOG(nPatches > 0, "Number of patches must be greater than zero");

    logstream logger(LOGGER_FILE);
    
    std::string dataFileDir = "data_48x48_HOG-LBP-descriptors+9-patches_fusion-patches1st-descriptors2nd/";
    std::vector< std::string > positivesID = { "id0011", "id0012", "id0013", "id0016", "id0020" };    
    std::vector< std::string > descriptorNames;
    
    #if ESVM_USE_HOG
    descriptorNames.push_back("hog");
    #endif/*ESVM_USE_HOG*/
    #if ESVM_USE_LBP
    descriptorNames.push_back("lbp");
    #endif/*ESVM_USE_LBP*/
    
    size_t nDescriptors = descriptorNames.size();
    size_t nProbes = 0;     // Gets updated after first ESVM testing
    for (auto posID = positivesID.begin(); posID != positivesID.end(); ++posID)
    {
        logger << "Starting training/testing ESVM evaluation for '" << *posID << "'..." << std::endl;

        std::vector<int> probeGroundTruths;
        std::vector<double> patchFusionScores, descriptorFusionScores;  // score fusion of patches, then fusion over descriptors
        std::vector<double> combinedFusionScores(nProbes, 0.0);         // simultaneous score fusion over patches and descriptors
        for (auto d = descriptorNames.begin(); d != descriptorNames.end(); ++d)
        {            
            for (size_t p = 0; p < nPatches; p++)
            {    
                std::string strPatch = std::to_string(p);
                std::string trainFileName = dataFileDir + "chokepoint-S1-id0011-" + *d + "-patch" + strPatch + "-train.data";
                std::string testFileName = dataFileDir + "chokepoint-S1-id0011-" + *d + "-patch" + strPatch + "-test.data";
               
                // Train/test ESVM from files
                logger << "Training ESVM with data file: '" << trainFileName << "'..." << std::endl;
                ESVM esvm = ESVM(trainFileName, *posID);
                logger << "Testing ESVM with data file: '" << testFileName << "'..." << std::endl;                
                std::vector<double> scores = esvm.predict(testFileName, &probeGroundTruths);
                std::vector<double> normScores = normalizeMinMaxClassScores(scores);
                
                nProbes = scores.size();
                if (p == 0)
                {
                    // Initialize fusion scores accumulators on first patch / feature extraction method as required
                    patchFusionScores = std::vector<double>(nProbes, 0.0);
                    if (d == descriptorNames.begin())
                        descriptorFusionScores = std::vector<double>(nProbes, 0.0);
                }
                for (size_t prb = 0; prb < nProbes; prb++)
                {
                    patchFusionScores[prb] += normScores[prb];          // Accumulation of patch-based scores
                    combinedFusionScores[prb] += normScores[prb];       // Accumulation of (patch,descriptor)-based scores

                    std::string probeGT = (probeGroundTruths[prb] > 0 ? "positive" : "negative");
                    logger << "Score for probe " << prb << " (" << probeGT << "): " << scores[prb] 
                           << " | normalized: " << normScores[prb] << std::endl;
                }
            }
            
            for (size_t prb = 0; prb < nProbes; prb++)
            {
                patchFusionScores[prb] /= (double)nPatches;             // Average of accumulated patch-based scores
                descriptorFusionScores[prb] += patchFusionScores[prb];  // Accumulation of feature-based scores
            }

            // Evaluate results per feature extraction method
            logger << "Performance evaluation for patch-based score fusion of '" + *d + "' descriptor:" << std::endl;
            eval_PerformanceClassificationScores(patchFusionScores, probeGroundTruths);
        }
        
        int nCombined = nDescriptors * nPatches;
        for (size_t prb = 0; prb < nProbes; prb++)
        {
            descriptorFusionScores[prb] /= (double)nDescriptors;        // Average of accumulated patch-based scores
            combinedFusionScores[prb] /= (double)nCombined;             // Average of accumulated (patch,descriptor)-based scores
        }

        // Evaluate results with fusioned descriptors and patches
        logger << "Performance evaluation for sequential patch-based + descriptor-based score fusion:" << std::endl;
        eval_PerformanceClassificationScores(descriptorFusionScores, probeGroundTruths);
        logger << "Performance evaluation for combined (patch,descriptor)-based score fusion:" << std::endl;
        eval_PerformanceClassificationScores(combinedFusionScores, probeGroundTruths);
    }
    logger << "Test complete" << std::endl;
    return 0;
}

/**************************************************************************************************************************
TEST DEFINITION

    Use pre-generated negative samples data files (HOG-588 9-patches) and extract features from enroll still to train an
    Ensemble of Exemplar-SVM. Test against various probes from corresponding process pre-generated data files, and also
    against probes using feature extraction process with first ChokePoint sequence (mode 0).
**************************************************************************************************************************/
int test_runSingleSamplePerPersonStillToVideo_NegativesDataFiles_PositivesExtraction_PatchBased()
{
    // Paths and logging
    logstream logger(LOGGER_FILE);
        
    ASSERT_LOG(!(TEST_READ_DATA_FILES & 0b10000000) != !(TEST_READ_DATA_FILES & 0b01110000),
               "Invalid 'TEST_READ_DATA_FILES' options flag (128) cannot be used simultaneously with [(16),(32),(64)]"); 
    const std::string hogTypeFilesPreGen = (TEST_READ_DATA_FILES & 0b10000000) ? "-C++" : "-MATLAB";
    const std::string imageTypeFilesPreGen = (TEST_READ_DATA_FILES & 0b10110000) ? "" : "-transposed";
    const std::string negativesDir = "negatives" + hogTypeFilesPreGen + imageTypeFilesPreGen + "/";
    const std::string probesFileDir = "data_SAMAN_48x48" + hogTypeFilesPreGen + imageTypeFilesPreGen + "_HOG-descriptor+9-patches/";

    // Check requirements
    ASSERT_LOG(ESVM_USE_HOG != 0, "HOG feature extraction is required for this test");
    ASSERT_LOG(TEST_CHOKEPOINT_SEQUENCES_MODE == 0, "ChokePoint sequence mode 0 is required for this test");
    ASSERT_LOG(bfs::is_directory(negativesDir), "Negatives file directory was not found");
    ASSERT_LOG(checkPathEndSlash(negativesDir), "Negatives file directory should end with a slash character");
    ASSERT_LOG(bfs::is_directory(probesFileDir), "Probes file directory was not found");
    ASSERT_LOG(checkPathEndSlash(probesFileDir), "Probes file directory should end with a slash character");

    /* Training Targets:    single high quality still image for enrollment (same as Saman code) */
    std::vector<int> positivesID = { 3, 5, 6, 10, 24 };
    /* Testing Probes:      some video positives and negatives */
    std::vector<int> probesID = { 3, 4, 5, 6, 10, 12, 15, 23, 24, 28 };
    
    // Predefined parameters to match pre-generated files    
    cv::Size imageSize = cv::Size(48, 48);
    cv::Size patchSize = cv::Size(16, 16);
    cv::Size patchCounts = cv::Size(3, 3);
    cv::Size blockSize = cv::Size(2, 2);
    cv::Size blockStride = cv::Size(2, 2);
    cv::Size cellSize = cv::Size(2, 2);
    int nBins = 3;
    FeatureExtractorHOG hog;
    hog.initialize(patchSize, blockSize, blockStride, cellSize, nBins);
    logger << "HOG feature extraction initialized..." << std::endl
           << "   imageSize:   " << imageSize << std::endl
           << "   patchSize:   " << patchSize << std::endl
           << "   blockSize:   " << blockSize << std::endl
           << "   blockStride: " << blockStride << std::endl
           << "   cellSize:    " << blockStride << std::endl
           << "   nBins:       " << nBins << std::endl;

    // Containers 
    size_t nPatches = 9;
    size_t nPositives = positivesID.size();
    size_t dimsPositives[2] = { nPositives, nPatches };    
    size_t dimsProbes[2] = { nPatches, 0 };                                 // known patch quantity, dynamically add probes
    xstd::mvector<2, ESVM> esvm(dimsPositives);                             // [target][patch](ESVM)
    xstd::mvector<2, FeatureVector> fvPositiveSamples(dimsPositives);       // [target][patch](FeatureVector)
    xstd::mvector<2, FeatureVector> fvProbeLoadedSamples(dimsProbes);       // [patch][probe](FeatureVector) - reversed indexing for easier access
    std::vector<std::string> probesLoadedID;                                // [probe](string)

    cv::namedWindow(WINDOW_NAME);

    // Load positive stills and extract features
    logger << "Loading positive enroll image stills..." << std::endl;
    for (size_t pos = 0; pos < nPositives; pos++)
    {
        std::string stillPath = roiChokePointEnrollStillPath + "roi" + buildChokePointIndividualID(positivesID[pos], true) + ".tif";
        std::vector<cv::Mat> patches = imPreprocess(stillPath, imageSize, patchCounts, false, WINDOW_NAME, cv::IMREAD_GRAYSCALE);       
        for (size_t p = 0; p < nPatches; p++)
            fvPositiveSamples[pos][p] = hog.compute(patches[p]);
    }

    // Load probe images and extract features if required (TEST_READ_DATA_FILES & 16|128)
    #if TEST_READ_DATA_FILES & 0b10010000
    std::vector<PORTAL_TYPE> types = { ENTER, LEAVE };
    bfs::directory_iterator endDir;
    std::string seq;    
    int sn = 1;                                                 // session number
    for (int pn = 1; pn <= PORTAL_QUANTITY; pn++)               // portal number
    for (auto it = types.begin(); it != types.end(); ++it)      // portal type
    for (int cn = 1; cn <= CAMERA_QUANTITY; cn++)               // camera number
    {
        seq = buildChokePointSequenceString(pn, *it, sn, cn);
        logger << "Loading probe images and extracting features from sequence " << seq << "..." << std::endl;

        // Add ROI to corresponding sample vectors according to individual IDs            
        for (int id = 1; id <= INDIVIDUAL_QUANTITY; id++)
        {
            std::string dirPath = roiChokePointCroppedFacePath + buildChokePointSequenceString(pn, *it, sn, cn, id) + "/";
            if (bfs::is_directory(dirPath))
            {
                for (bfs::directory_iterator itDir(dirPath); itDir != endDir; ++itDir)
                {
                    if (bfs::is_regular_file(*itDir) && itDir->path().extension() == ".pgm")
                    {                        
                        if (contains(probesID, id))
                        {                            
                            std::vector<cv::Mat> patches = imPreprocess(itDir->path().string(), imageSize, patchCounts,
                                                                        false, WINDOW_NAME, cv::IMREAD_GRAYSCALE);
                            for (size_t p = 0; p < nPatches; p++)
                                fvProbeLoadedSamples[p].push_back(hog.compute(patches[p]));

                            probesLoadedID.push_back(buildChokePointIndividualID(id, true));
                        } 
                    }
                }
            }                        
        }
    }
    #endif/*TEST_READ_DATA_FILES & (16|128)*/
    cv::destroyWindow(WINDOW_NAME);

    // load negatives from pre-generated files
    ESVM tmpLoadESVM;                                                   // temporary ESVM only for file loading
    size_t dimsNegatives[2]{ nPatches, 0 };                             // dynamically fill patches from file loading
    xstd::mvector<2, FeatureVector> fvNegativeSamples(dimsNegatives);   // [patch][negative](FeatureVector)
    for (int p = 0; p < nPatches; p++)
    {
        std::string strPatch = std::to_string(p);
        std::string negativeTrainFile = negativesDir + "negatives-patch" + strPatch + ".data";
        logger << "Loading pre-generated negative samples file for patch " << strPatch << "..." << std::endl 
               << "   Using file: '" << negativeTrainFile << "'" << std::endl;
        std::vector<FeatureVector> fvNegativeSamplesPatch;
        std::vector<int> negativeGroundTruths;
        tmpLoadESVM.readSampleDataFile(negativeTrainFile, fvNegativeSamplesPatch, negativeGroundTruths);
        fvNegativeSamples[p] = xstd::mvector<1, FeatureVector>(fvNegativeSamplesPatch);
    }

    // execute feature normalization as required
    //    N.B. Pre-generated negatives and probes samples from files are already normalized
    #if TEST_FEATURES_NORMALIZATION_MODE == 3
    logger << "Applying feature normalization (across features, across patches) for loaded positives and probes..." << std::endl;
    double hardcodedFoundMin = 0;               // Min found using 'FullChokePoint' test
    double hardcodedFoundMax = 0.675058;        // Max found using 'FullChokePoint' test
    int nProbesLoaded = fvProbeLoadedSamples[0].size();
    for (int p = 0; p < nPatches; p++)
    {
        for (int pos = 0; pos < nPositives; pos++)
            fvPositiveSamples[pos][p] = normalizeMinMaxAllFeatures(fvPositiveSamples[pos][p], hardcodedFoundMin, hardcodedFoundMax);
        for (int prb = 0; prb < nProbesLoaded; prb++)
            fvProbeLoadedSamples[p][prb] = normalizeMinMaxAllFeatures(fvProbeLoadedSamples[p][prb], hardcodedFoundMin, hardcodedFoundMax);
    }
    #endif/*TEST_FEATURES_NORMALIZATION_MODE == 3*/

    // train and test ESVM
    for (int pos = 0; pos < nPositives; pos++)
    {        
        std::vector<int> probeGroundTruthsPreGen, probeGroundTruthsLoaded;                                  // [probe](int)
        std::vector<double> probeFusionScoresPreGen, probeFusionScoresLoaded;                               // [probe](double)        
        xstd::mvector<2, double> probePatchScoresPreGen(dimsProbes), probePatchScoresLoaded(dimsProbes);    // [patch][probe](double)        
        std::string posID = buildChokePointIndividualID(positivesID[pos], true);
        logger << "Starting ESVM training/testing for '" << posID << "'..." << std::endl;
        for (int p = 0; p < nPatches; p++)
        {
            // train with positive extracted features and negative loaded features
            std::string strPatch = std::to_string(p);
            logger << "Starting ESVM training for '" << posID << "', patch " << strPatch << "..." << std::endl;
            std::vector<FeatureVector> fvPositiveSingleSamplePatch = { fvPositiveSamples[pos][p] };
            std::vector<FeatureVector> fvNegativeSamplesPatch = std::vector<FeatureVector>(fvNegativeSamples[p]);
            esvm[pos][p] = ESVM(fvPositiveSingleSamplePatch, fvNegativeSamplesPatch, posID + "-" + strPatch);

            // test against pre-generated probes and loaded probes
            #if TEST_READ_DATA_FILES & 0b10010000   // (16|128) use feature extraction on probe images
            logger << "Starting ESVM testing for '" << posID << "', patch " << strPatch << " (probe images and extract feature)..." << std::endl;
            std::vector<double> scoresLoaded = esvm[pos][p].predict(fvProbeLoadedSamples[p]);
            probePatchScoresLoaded[p] = xstd::mvector<1, double>(scoresLoaded);
            #endif/*TEST_READ_DATA_FILES & (16|128)*/
            #if TEST_READ_DATA_FILES & 0b01100000   // (32|64) use pre-generated probe sample file
            std::string probePreGenTestFile = probesFileDir + "test-target" + posID + "-patch" + strPatch + ".data";
            logger << "Starting ESVM testing for '" << posID << "', patch " << strPatch << " (probe pre-generated samples files)..." << std::endl
                   << "   Using file: '" << probePreGenTestFile << "'" << std::endl;                  
            std::vector<double> scoresPreGen = esvm[pos][p].predict(probePreGenTestFile, &probeGroundTruthsPreGen);
            probePatchScoresPreGen[p] = xstd::mvector<1, double>(scoresPreGen);
            #endif/*TEST_READ_DATA_FILES & (32|64)*/
        }

        logger << "Starting score fusion and normalization for '" << posID << "'..." << std::endl;
        
        /* ----------------------------------------------
           (16|128) use feature extraction on probe images
        ---------------------------------------------- */
        #if TEST_READ_DATA_FILES & 0b10010000

        // accumulated sum of scores for score fusion
        int nProbesLoaded = probePatchScoresLoaded[0].size();
        probeFusionScoresLoaded = std::vector<double>(nProbesLoaded, 0.0);
        for (int p = 0; p < nPatches; p++)
            for (int prb = 0; prb < nProbesLoaded; prb++)
                probeFusionScoresLoaded[prb] += probePatchScoresLoaded[p][prb];

        // average accumulated scores and execute post-fusion normalization
        // also find ground truths of feature vectors
        for (int prb = 0; prb < nProbesLoaded; prb++)
        {
            probeGroundTruthsLoaded.push_back(probesLoadedID[prb] == posID ? ESVM_POSITIVE_CLASS : ESVM_NEGATIVE_CLASS);
            probeFusionScoresLoaded[prb] /= (double)nPatches;
        }        
        probeFusionScoresLoaded = normalizeMinMaxClassScores(probeFusionScoresLoaded);
        
        // evaluate results with fusioned patch scores
        logger << "Performance evaluation for loaded/extracted probes (no pre-norm, post-fusion norm) of '" << posID << "':" << std::endl;
        eval_PerformanceClassificationScores(probeFusionScoresLoaded, probeGroundTruthsLoaded);
        
        #endif/*TEST_READ_DATA_FILES & (16|128)*/  

        /* -------------------------------------------------------------------------------------------------------
           (32|64) use pre-generated probe sample file ([normal|transposed] images employed to generate files) 
        ------------------------------------------------------------------------------------------------------- */
        #if TEST_READ_DATA_FILES & 0b01100000 

        // accumulated sum of scores for score fusion
        int nProbesPreGen = probePatchScoresPreGen[0].size();
        probeFusionScoresPreGen = std::vector<double>(nProbesPreGen, 0.0);
        for (int p = 0; p < nPatches; p++)
            for (int prb = 0; prb < nProbesPreGen; prb++)
                probeFusionScoresPreGen[prb] += probePatchScoresPreGen[p][prb];
        
        // average accumulated scores and execute post-fusion normalization
        for (int prb = 0; prb < nProbesPreGen; prb++)
            probeFusionScoresPreGen[prb] /= (double)nPatches;
        probeFusionScoresPreGen = normalizeMinMaxClassScores(probeFusionScoresPreGen);
        
        // evaluate results with fusioned patch scores
        logger << "Performance evaluation for pre-generated probes (no pre-norm, post-fusion norm) of '" << posID << "':" << std::endl;
        eval_PerformanceClassificationScores(probeFusionScoresPreGen, probeGroundTruthsPreGen);
        
        #endif/*TEST_READ_DATA_FILES & (32|64)*/
    }
    
    logger << "Test complete" << std::endl;
    return 0;
}

/**************************************************************************************************************************
TEST DEFINITION
    
    Use person-based track ROIs obtained from (FD+FT) of 'Fast-DT + CompressiveTracking + 3 Haar Cascades' extracted from
    TITAN Unit's videos dataset to enroll stills with ESVM and test against probes under the same environment.
    Use negative samples from ChokePoint dataset across multiple camera angles.
**************************************************************************************************************************/
int test_runSingleSamplePerPersonStillToVideo_TITAN(cv::Size imageSize, cv::Size patchCounts, bool useSyntheticPositives)
{
    /* Training Targets: single high quality still image for enrollment */
    std::vector<std::string> positivesID = 
    {   
        "01 Eric",
        "02 Ferry", 
        "03 Ghena", 
        "04 Irina",
        "05 Rene",
        "06 Roman"
    };
    class PositiveImageStill 
    { 
    public:
        PositiveImageStill(std::string id, std::string path) { ID = id; Path = path; }
        std::string ID; 
        std::string Path;
    };

    std::vector<PositiveImageStill> positiveImageStills =
    {
        //roiTitanUnitEnrollStillPath + positivesID[0] + " face ROI.png",
        //roiTitanUnitEnrollStillPath + positivesID[1] + " face ROI.png",
        PositiveImageStill(positivesID[2], roiTitanUnitEnrollStillPath + positivesID[2] + " face ROI.png"),
        PositiveImageStill(positivesID[3], roiTitanUnitEnrollStillPath + positivesID[3] + " face ROI.png"),
        PositiveImageStill(positivesID[4], roiTitanUnitEnrollStillPath + positivesID[4] + " face ROI.png"),
        PositiveImageStill(positivesID[5], roiTitanUnitEnrollStillPath + positivesID[5] + " face ROI.png")
    };

    /* Testing Probes: some video positives and negatives */
    std::vector<std::string> probePersonDirs = 
    {
        roiTitanUnitFastDTTrackPath + "1 fixed/images/person_1",    // Ghena
        roiTitanUnitFastDTTrackPath + "1 fixed/images/person_3",    // Irina
        roiTitanUnitFastDTTrackPath + "1 fixed/images/person_5",    // Irina
        roiTitanUnitFastDTTrackPath + "1 fixed/images/person_8",    // Roman
    };

    std::string writingDataFileDir = "data_TITAN_48x48_HOG-descriptor+9-patches/";

    // Display and output
    cv::namedWindow(WINDOW_NAME);
    logstream logger(LOGGER_FILE);
    size_t nPatches = patchCounts.width * patchCounts.height;
    if (nPatches == 0) nPatches = 1;    
    bool useHistEqual = false;
    logger << "Starting single sample per person still-to-video full ChokePoint test..." << std::endl
           << "   useSyntheticPositives: " << useSyntheticPositives << std::endl
           << "   imageSize:             " << imageSize << std::endl
           << "   patchCounts:           " << patchCounts << std::endl
           << "   useHistEqual:          " << useHistEqual << std::endl;
    
    size_t nDescriptors = 0;
    std::vector<std::string> descriptorNames;
    #if ESVM_USE_HOG
    nDescriptors++;
    std::string descriptorHOG = "hog";
    descriptorNames.push_back(descriptorHOG);
    FeatureExtractorHOG hog;
    cv::Size patchSize = cv::Size(imageSize.width / patchCounts.width, imageSize.height / patchCounts.height);
    cv::Size hogBlock = cv::Size(patchSize.width / 2, patchSize.height / 2);
    cv::Size hogCell = cv::Size(hogBlock.width / 2, hogBlock.height / 2);
    int nBins = 8;
    hog.initialize(patchSize, hogBlock, hogBlock, hogCell, nBins);
    logger << "HOG feature extraction initialized..." << std::endl
           << "   imageSize: " << imageSize << std::endl
           << "   patchSize: " << patchSize << std::endl
           << "   hogBlock:  " << hogBlock << std::endl
           << "   hogCell:   " << hogCell << std::endl 
           << "   nBins:     " << nBins << std::endl;
    #endif/*ESVM_USE_HOG*/
    
    #if ESVM_USE_LBP
    nDescriptors++;
    std::string descriptorLBP = "lbp";
    descriptorNames.push_back(descriptorLBP);
    FeatureExtractorLBP lbp;
    int points = 8;
    int radius = 8;
    MappingType map = LBP_MAPPING_U2;
    lbp.initialize(points, radius, map);
    logger << "LBP feature extraction initialized..." << std::endl
           << "   imageSize: " << imageSize << std::endl
           << "   points:    " << points << std::endl
           << "   radius:    " << radius << std::endl
           << "   mapping:   " << lbp::MappingTypeStr[map] << std::endl;    
    #endif/*ESVM_USE_LBP*/  

    ASSERT_LOG(nDescriptors > 0, "At least one of the feature extraction method must be enabled");

    // Samples container with expected indexes usage
    /*
        NB:     When using the 'mvector' class, if only a single 'int' value is specified for size, it is assigned for each dimension.
                Using 'push_back' later on a deeper level of 'vector' will put the new values after the N default values would have been
                added because of the single 'int' dimension value.
                Dimensions should therefore be mentionned explicitely using an array of size for each 'vector' level, be initialized later
                as required with lower dimension 'mvector', or using a 'zero' dimension (empty).
    */
    
    size_t nPositives = positiveImageStills.size();
    size_t nRepresentations = 1;
    size_t nDuplications = 10;
    size_t dimsGroundTruths[2] { nPositives, 0 };
    size_t dimsImgPositives[3] { nPositives, nRepresentations, nPatches };   
    xstd::mvector<2, int> probeGroundTruth(dimsGroundTruths);                   // [positive][probe](int)
    xstd::mvector<3, cv::Mat> matPositiveSamples(dimsImgPositives);             // [positive][representation][patch](Mat[x,y])
    xstd::mvector<2, cv::Mat> matNegativeSamples;                               // [negative][patch](Mat[x,y])
    xstd::mvector<2, cv::Mat> matProbeSamples;                                  // [probe][patch](Mat[x,y])     
    std::vector<std::string> probeID;                                           // [probe](string)

    // Add samples to containers
    logger << "Loading positives image for all test sequences..." << std::endl;
    for (size_t pos = 0; pos < nPositives; pos++)
    {        
        // Add additional positive representations as requested
        if (useSyntheticPositives)
        {
            // Get original positive image with preprocessing but without patches splitting
            cv::Mat img = imPreprocess(positiveImageStills[pos].Path, imageSize, cv::Size(1,1), useHistEqual, WINDOW_NAME, cv::IMREAD_COLOR)[0];
            // Get synthetic representations from original and apply patches splitting each one
            std::vector<cv::Mat> representations = imSyntheticGeneration(img);
            // Reinitialize sub-container for augmented representations using synthetic images
            nRepresentations = representations.size();
            size_t dimsRepresentation[2] { nRepresentations, nPatches };
            matPositiveSamples[pos] = xstd::mvector<2, cv::Mat>(dimsRepresentation);

            /// ############################################# #pragma omp parallel for
            for (size_t r = 0; r < nRepresentations; r++)
            {
                std::vector<cv::Mat> patches = imSplitPatches(representations[r], patchCounts);
                for (size_t p = 0; p < nPatches; p++)
                    matPositiveSamples[pos][r][p] = patches[p];
            }
        }
        // Only original representation otherwise (no synthetic images)
        else
        {
            //// matPositiveSamples[pos] = std::vector< std::vector< cv::Mat> >(1);
            std::vector<cv::Mat> patches = imPreprocess(positiveImageStills[pos].Path, imageSize, patchCounts,
                                                        useHistEqual, WINDOW_NAME, cv::IMREAD_COLOR);
            for (size_t p = 0; p < nPatches; p++)
                matPositiveSamples[pos][0][p] = patches[p];
        }
    }

    /// ################################################################################ DEBUG DISPLAY POSITIVES (+SYNTH)
    /*
    logger << "SHOWING DEBUG POSITIVE SAMPLES" << std::endl;
    for (int i = 0; i < nPositives; i++)
    {
        for (int j = 0; j < matPositiveSamples[i].size(); j++)
        {
            for (int k = 0; k < matPositiveSamples[i][j].size(); k++)
            {
                cv::imshow(WINDOW_NAME, matPositiveSamples[i][j][k]);
                cv::waitKey(500);
            }
        }
    }
    logger << "DONE SHOWING DEBUG POSITIVE SAMPLES" << std::endl;
    */
    // Destroy viewing window not required anymore    
    // cv::destroyWindow(WINDOW_NAME);
    /// ################################################################################ DEBUG

    logger << "Feature extraction of positive images for all test sequences..." << std::endl
           << "   nPositives:       " << nPositives << std::endl
           << "   nPatches:         " << nPatches << std::endl
           << "   nRepresentations: " << nRepresentations << std::endl
           << "   nDuplications:    " << nDuplications << std::endl
           << "   nDescriptors:     " << nDescriptors << std::endl;

    // Containers for feature vectors extracted from samples
    size_t dimsPositives[4] = { nPositives, nPatches, nDescriptors, nRepresentations }; // note: 'mshape' fails in this case, size_t array works...
    size_t dimsESVM[3] { nPositives, nPatches, nDescriptors };
    xstd::mvector<3, ESVM> esvmModels(dimsESVM);                            // [target][patch][descriptor](ESVM)
    xstd::mvector<4, FeatureVector> fvPositiveSamples(dimsPositives);       // [target][patch][descriptor][positive](FeatureVector)
    xstd::mvector<3, FeatureVector> fvNegativeSamples;                      // [patch][descriptor][negative](FeatureVector)
    xstd::mvector<3, FeatureVector> fvProbeSamples;                         // [patch][descriptor][probe](FeatureVector)

    // Convert unique positive samples (or with synthetic representations)
    /// ################################################## #pragma omp parallel for    
    for (size_t pos = 0; pos < nPositives; pos++)
    {                
        /// ################################################## #pragma omp parallel for
        for (size_t p = 0; p < nPatches; p++)
        {
            for (size_t d = 0; d < nDescriptors; d++)
            {                
                /// ################################################## #pragma omp parallel for
                for (int r = 0; r < nRepresentations; r++)
                {
                    // switch to (i,p,fe,r) order for (patch,feature)-based training of sample representations
                    #if ESVM_USE_HOG
                    if (descriptorNames[d] == descriptorHOG)
                        fvPositiveSamples[pos][p][d][r] = hog.compute(matPositiveSamples[pos][r][p]);
                    #endif/*ESVM_USE_HOG*/
                    #if ESVM_USE_LBP
                    if (descriptorNames[d] == descriptorLBP)
                        fvPositiveSamples[pos][p][d][r] = lbp.compute(matPositiveSamples[pos][r][p]);                    
                    #endif/*ESVM_USE_LBP*/
                }

                /// ################################################## 
                // DUPLICATE EVEN MORE REPRESENTATIONS TO HELP LIBSVM PROBABILITY ESTIMATES CROSS-VALIDATION
                // Add x-times the number of representations
                if (nDuplications > 1)
                    for (size_t r = 0; r < nRepresentations; r++)
                        for (size_t dup = 1; dup < nDuplications; dup++)
                            fvPositiveSamples[pos][p][d].push_back(fvPositiveSamples[pos][p][d][r]);
            }
        }
    }
    nRepresentations *= nDuplications;
    for (size_t d = 0; d < nDescriptors; d++)
        logger << "Features dimension (" + descriptorNames[d] + "): " << fvPositiveSamples[0][0][d][0].size() << std::endl;

    // Load negative samples from ChokePoint dataset
    std::vector<PORTAL_TYPE> types = { ENTER, LEAVE };
    bfs::directory_iterator endDir;
    int sn = 1;     // session number    
    for (int pn = 1; pn <= PORTAL_QUANTITY; pn++)
    {
        for (auto it = types.begin(); it != types.end(); ++it)
        {
            for (int cn = 1; cn <= CAMERA_QUANTITY; cn++)
            {
                std::string seq = buildChokePointSequenceString(pn, *it, sn, cn);
                logger << "Loading negative and probe images for sequence " << seq << "..." << std::endl;

                // Add ROI to corresponding sample vectors
                for (int id = 1; id <= INDIVIDUAL_QUANTITY; id++)
                {
                    std::string dirPath = roiChokePointCroppedFacePath + buildChokePointSequenceString(pn, *it, sn, cn, id) + "/";
                    if (bfs::is_directory(dirPath))
                    {
                        for (bfs::directory_iterator itDir(dirPath); itDir != endDir; ++itDir)
                        {
                            if (bfs::is_regular_file(*itDir) && itDir->path().extension() == ".pgm")
                            {                                
                                size_t neg = matNegativeSamples.size();
                                matNegativeSamples.push_back(xstd::mvector<1, cv::Mat>(nPatches));
                                std::vector<cv::Mat> patches = imPreprocess(itDir->path().string(), imageSize, patchCounts,
                                                                            useHistEqual, WINDOW_NAME, cv::IMREAD_GRAYSCALE);
                                for (size_t p = 0; p < nPatches; p++)
                                    matNegativeSamples[neg][p] = patches[p];
    } } } } } } }   // End of negatives loading

    // Load probe samples
    /*
    else if (contains(probesID, strID))
    {
        size_t prb = matProbeSamples.size();
        matProbeSamples.push_back(xstd::mvector<1, cv::Mat>(nPatches));
        std::vector<cv::Mat> patches = imPreprocess(itDir->path().string(), imageSize, patchCounts, 
                                                    useHistEqual, WINDOW_NAME, cv::IMREAD_GRAYSCALE);
        for (size_t p = 0; p < nPatches; p++)
            matProbeSamples[prb][p] = patches[p];

        probeID.push_back(strID);
        for (size_t pos = 0; pos < nPositives; pos++)
            probeGroundTruth[pos].push_back(strID == positivesID[pos] ? ESVM_POSITIVE_CLASS : ESVM_NEGATIVE_CLASS);
    }
    */




    return -1;
}

/**************************************************************************************************************************
TEST DEFINITION

    Uses pre-generated sample feature train/test files from SAMAN MATLAB code to enroll targets and test against probes. 
    Parameters are pre-defined according to MATLAB code. 
    Sample features are pre-extracted with HOG+PCA and normalized.
**************************************************************************************************************************/
int test_runSingleSamplePerPersonStillToVideo_DataFiles_SAMAN()
{
    ASSERT_LOG(TEST_ESVM_SAMAN != 0, "Test 'test_runSingleSamplePerPersonStillToVideo_DataFiles_SAMAN' not selected for running");
    #if TEST_ESVM_SAMAN

    std::vector<std::string> positivesID = { "ID0003", "ID0005", "ID0006", "ID0010", "ID0024" };
    size_t nPositives = positivesID.size();
    size_t nPatches = 9;    
    size_t nProbes = 0;     // set when read from testing file
    #if TEST_ESVM_SAMAN == 1
    size_t nFeatures = 128;
    std::string dataFileDir = "data_SAMAN_48x48-MATLAB_HOG-PCA-descriptor+9-patches/";
    #elif TEST_ESVM_SAMAN == 2
    size_t nFeatures = 588;
    std::string dataFileDir = "data_SAMAN_48x48-MATLAB_HOG-descriptor+9-patches/";
    #elif TEST_ESVM_SAMAN == 3
    size_t nFeatures = 588;
    std::string dataFileDir = "data_SAMAN_48x48-MATLAB-transposed_HOG-descriptor+9-patches/";
    #elif TEST_ESVM_SAMAN == 4
    size_t nFeatures = 588;
    std::string dataFileDir = "data_ChokePoint_48x48_HOG-impl-588_9-patches_PreNormOverall-Mode3/";
    #endif/*TEST_ESVM_SAMAN*/

    size_t dimsESVM[2] = { nPositives, nPatches };
    xstd::mvector<2, ESVM> esvm(dimsESVM);          // [positive][patch](ESVM)

    logstream logger(LOGGER_FILE);
    for (size_t pos = 0; pos < nPositives; pos++)
    {
        std::string posID = positivesID[pos];
        logger << "Starting for '" << posID << "'..." << std::endl;

        std::vector<int> probeGroundTruths;                         // assigned when running prediction
        std::vector<std::vector<double> > probeFusionScoresCumul;   // accumulation of patch scores for probe-based normalization [patch][probe]
        std::vector<double> probeFusionScoresNormGradual;           // gradual normalization across corresponding patches for score fusion
        std::vector<double> probeFusionScoresNormFinal;             // final normalization across all patches of same probe for score fusion
        std::vector<double> probeFusionScoresNormSkipped;           // skipped any pre-normalization procedure for score fusion with raw scores
        std::vector<double> probeFusionScoresNormGradualPostNorm;   // post-fusion normalization of scores obtained from gradual normalization
        std::vector<double> probeFusionScoresNormFinalPostNorm;     // post-fusion normalization of scores obtained from final normalization
        std::vector<double> probeFusionScoresNormSkippedPostNorm;   // post-fusion normalization of scores obtained from skipped normalization
        for (size_t p = 0; p < nPatches; p++)
        {
            // run training / testing from files            
            std::vector<double> probePatchScores;
            std::string strPatch = std::to_string(p);
            #if TEST_ESVM_SAMAN == 4        // Using files generated by 'FullChokePoint' test             
            std::string trainFile = dataFileDir + "chokepoint-S1-" + posID + "-hog-patch" + strPatch + "-train.data";
            std::string testFile = dataFileDir + "chokepoint-S1-" + posID + "-hog-patch" + strPatch + "-test.data";
            #else/*TEST_ESVM_SAMAN != 4*/   // Using files generated by the SAMAN MATLAB code
            std::string trainFile = dataFileDir + "train-target" + posID + "-patch" + strPatch + ".data";
            std::string testFile = dataFileDir + "test-target" + posID + "-patch" + strPatch + ".data";
            #endif/*TEST_ESVM_SAMAN*/
            logger << "Starting ESVM training from pre-generated file for '" << posID << "', patch " << strPatch << "..." << std::endl
                   << "   Using file: '" << trainFile << "'" << std::endl;
            esvm[pos][p] = ESVM(trainFile, posID);
            logger << "Starting ESVM testing from pre-generated file for '" << posID << "', patch " << strPatch << "..." << std::endl
                   << "   Using file: '" << testFile << "'" << std::endl;
            probePatchScores = esvm[pos][p].predict(testFile, &probeGroundTruths);

            // score normalization for patch
            std::vector<double> normPatchScores = normalizeMinMaxClassScores(probePatchScores);
            logger << "Scores for '" << posID << "', patch " << strPatch << ":" << std::endl
                   << featuresToVectorString(probePatchScores) << std::endl;
            logger << "Scores normalized for '" << posID << "', patch " << strPatch << ":" << std::endl
                   << featuresToVectorString(normPatchScores) << std::endl;

            // score fusion accumulation
            nProbes = probePatchScores.size();
            probeFusionScoresCumul.push_back(probePatchScores);                 // scores as is for final normalization
            if (probeFusionScoresNormGradual.size() == 0)
                probeFusionScoresNormGradual = normPatchScores;                 // initialize in case of first run
            else
                for (size_t prb = 0; prb < nProbes; prb++)
                    probeFusionScoresNormGradual[prb] += normPatchScores[prb];  // gradually accumulate normalized scores       
        } 

        ASSERT_LOG(nProbes > 0, "Number of probes should have been updated from loaded samples and be greater than zero");

        // score fusion of patches
        probeFusionScoresNormFinal = std::vector<double>(nProbes);
        probeFusionScoresNormSkipped = std::vector<double>(nProbes);
        for (size_t prb = 0; prb < nProbes; prb++)
        {
            probeFusionScoresNormGradual[prb] /= (double)nProbes;               // average of gradually accumulated and normalized patch scores
            double probeScoresSum = 0, probeNormScoresSum = 0;
            std::vector<double> tmpProbeScores;
            for (size_t p = 0; p < nPatches; p++)
            {
                tmpProbeScores.push_back(probeFusionScoresCumul[p][prb]);
                probeScoresSum += probeFusionScoresCumul[p][prb];               // accumulate across patch scores of corresponding probe 
            }
            std::vector<double> normProbeScores = normalizeMinMaxClassScores(tmpProbeScores);
            for (size_t p = 0; p < nPatches; p++)
                probeNormScoresSum += normProbeScores[p];                       // accumulate across normalized patch scores of corresponding probe 
            probeFusionScoresNormFinal[prb] = probeNormScoresSum / (double)nPatches;
            probeFusionScoresNormSkipped[prb] = probeScoresSum / (double)nPatches;
        }
        probeFusionScoresNormGradualPostNorm = normalizeMinMaxClassScores(probeFusionScoresNormGradual);
        probeFusionScoresNormFinalPostNorm = normalizeMinMaxClassScores(probeFusionScoresNormFinal);
        probeFusionScoresNormSkippedPostNorm = normalizeMinMaxClassScores(probeFusionScoresNormSkipped);
        
        // output resulting score fusion
        logger << "Score fusion of gradual normalization scores for '" << posID << "':" << std::endl 
               << featuresToVectorString(probeFusionScoresNormGradual) << std::endl;
        logger << "Score fusion of final normalization scores for '" << posID << "':" << std::endl
               << featuresToVectorString(probeFusionScoresNormFinal) << std::endl;
        logger << "Score fusion of skipped normalization scores for '" << posID << "':" << std::endl
               << featuresToVectorString(probeFusionScoresNormSkipped) << std::endl;
        logger << "Score fusion post-normalization of gradual normalization scores for '" << posID << "':" << std::endl
               << featuresToVectorString(probeFusionScoresNormGradualPostNorm) << std::endl;
        logger << "Score fusion post-normalization of final normalization scores for '" << posID << "':" << std::endl
               << featuresToVectorString(probeFusionScoresNormFinalPostNorm) << std::endl;
        logger << "Score fusion post-normalization of skipped normalization scores for '" << posID << "':" << std::endl
               << featuresToVectorString(probeFusionScoresNormSkippedPostNorm) << std::endl;

        // performance evaluation
        logger << "Performance evaluation for '" << posID << "' (gradual norm scores, no post-fusion norm):" << std::endl;
        eval_PerformanceClassificationScores(probeFusionScoresNormGradual, probeGroundTruths);
        logger << "Performance evaluation for '" << posID << "' (final norm scores, no post-fusion norm):" << std::endl;
        eval_PerformanceClassificationScores(probeFusionScoresNormFinal, probeGroundTruths);
        logger << "Performance evaluation for '" << posID << "' (skipped norm scores, no post-fusion norm):" << std::endl;
        eval_PerformanceClassificationScores(probeFusionScoresNormSkipped, probeGroundTruths);
        logger << "Performance evaluation for '" << posID << "' (gradual norm scores, with post-fusion norm):" << std::endl;
        eval_PerformanceClassificationScores(probeFusionScoresNormGradualPostNorm, probeGroundTruths);
        logger << "Performance evaluation for '" << posID << "' (final norm scores, with post-fusion norm):" << std::endl;
        eval_PerformanceClassificationScores(probeFusionScoresNormFinalPostNorm, probeGroundTruths);
        logger << "Performance evaluation for '" << posID << "' (skipped norm scores, with post-fusion norm):" << std::endl;
        eval_PerformanceClassificationScores(probeFusionScoresNormSkippedPostNorm, probeGroundTruths);
    }

    #endif/*TEST_ESVM_SAMAN*/
    return 0;
}

/**************************************************************************************************************************
TEST DEFINITION

    This test corresponds to the complete and working procedure to enroll and test image stills against pre-generated 
    negative samples files from the ChokePoint dataset (Sequence 1).  
**************************************************************************************************************************/
int test_runSingleSamplePerPersonStillToVideo_DataFiles_SimplifiedWorkingProcedure()
{
    logstream logger(LOGGER_FILE);

    // image and patch dimensions 
    cv::Size imageSize(48, 48);
    cv::Size patchCounts(3, 3);    
    size_t nPatches = patchCounts.area();

    // positive samples
    std::vector<std::string> positivesID = { "ID0003", "ID0005", "ID0006", "ID0010", "ID0024" };
    size_t nPositives = positivesID.size();    
    size_t dimsPositives[2]{ nPatches, nPositives };
    xstd::mvector<2, FeatureVector> positiveSamples(dimsPositives);     // [patch][positives](FeatureVector)

    // negative samples    
    size_t dimsNegatives[2]{ nPatches, 0 };                             // number of negatives unknown (loaded from file)
    xstd::mvector<2, FeatureVector> negativeSamples(dimsNegatives);     // [patch][negative](FeatureVector)
    
    // probe samples
    size_t dimsProbes[3]{ nPatches, nPositives, 0 };                    // number of probes unknown (loaded from file)
    xstd::mvector<3, FeatureVector> probeSamples(dimsProbes);           // [patch][positive][probe](FeatureVector)    

    // classification results    
    size_t dimsResults[2]{ nPositives, 0 };                             // number of probes unknown (loaded from file)
    xstd::mvector<3, double> scores(dimsProbes);                        // [patch][positive][probe](double)
    xstd::mvector<2, double> classificationScores(dimsResults);         // [positive][probe](double)    
    xstd::mvector<2, int> probeGroundTruths(dimsResults);               // [positive][probe](int)

    // Exemplar-SVM
    ESVM FileLoaderESVM;
    xstd::mvector<2, ESVM> esvm(dimsPositives);                         // [patch][positive](ESVM)    

    // prepare hog feature extractor    
    cv::Size blockSize(2, 2);
    cv::Size blockStride(2, 2);
    cv::Size cellSize(2, 2);
    int nBins = 3;
    FeatureExtractorHOG hog(imageSize, blockSize, blockStride, cellSize, nBins);
    double hogHardcodedFoundMin = 0;            // Min found using 'FullChokePoint' test with SAMAN pre-generated files
    double hogHardcodedFoundMax = 0.675058;     // Max found using 'FullChokePoint' test with SAMAN pre-generated files
    
    // load positive target still images, extract features and normalize
    logger << "Loading positive image stills, extracting feature vectors and normalizing..." << std::endl;
    for (size_t pos = 0; pos < nPositives; pos++)
    {        
        std::vector<cv::Mat> patches = imPreprocess(refStillImagesPath + "roi" + positivesID[pos] + ".tif", imageSize, patchCounts);
        for (size_t p = 0; p < nPatches; p++)
            positiveSamples[p][pos] = normalizeMinMaxAllFeatures(hog.compute(patches[p]), hogHardcodedFoundMin, hogHardcodedFoundMax);        
    }

    // load negative samples from pre-generated files for training (samples in files are pre-normalized)
    logger << "Loading negative samples from files..." << std::endl;
    for (size_t p = 0; p < nPatches; p++)
        FileLoaderESVM.readSampleDataFile(negativeSamplesDir + "negatives-hog-patch" + std::to_string(p) + ".data", negativeSamples[p]);    
        
    for (size_t p = 0; p < nPatches; p++)
    {
        std::vector<int> dummyOutput(negativeSamples[p].size(), ESVM_NEGATIVE_CLASS);
        FileLoaderESVM.writeSampleDataFile(negativeSamplesDir + "negatives-hog-patch" + std::to_string(p) + ".bin", negativeSamples[p], dummyOutput, BINARY);
    }

    // load probe samples from pre-generated files for testing (samples in files are pre-normalized)
    logger << "Loading probe samples from files..." << std::endl;
    for (size_t p = 0; p < nPatches; p++)
        for (size_t pos = 0; pos < nPositives; pos++)
            FileLoaderESVM.readSampleDataFile(testingSamplesDir + positivesID[pos] + "-probes-hog-patch" + std::to_string(p) + ".data",
                                              probeSamples[p][pos], probeGroundTruths[pos]);  

        for (size_t p = 0; p < nPatches; p++)
        for (size_t pos = 0; pos < nPositives; pos++)
            FileLoaderESVM.writeSampleDataFile(testingSamplesDir + positivesID[pos] + "-probes-hog-patch" + std::to_string(p) + ".bin",
                                              probeSamples[p][pos], probeGroundTruths[pos], BINARY);  

    // training
    logger << "Training ESVM with positives and negatives..." << std::endl;
    for (size_t p = 0; p < nPatches; p++)
        for (size_t pos = 0; pos < nPositives; pos++)
            esvm[p][pos] = ESVM({ positiveSamples[p][pos] }, negativeSamples[p], positivesID[pos] + "-patch" + std::to_string(p));

    // testing, score fusion, normalization
    logger << "Testing probe samples against enrolled targets..." << std::endl;
    for (size_t pos = 0; pos < nPositives; pos++) 
    {
        int nProbes = probeSamples[0][pos].size();      // variable number of probes according to tested positive
        classificationScores[pos] = xstd::mvector<1, double>(nProbes, 0);
        for (size_t prb = 0; prb < nProbes; prb++)
        {            
            for (size_t p = 0; p < nPatches; p++)
            {
                scores[p][pos].push_back( esvm[p][pos].predict(probeSamples[p][pos][prb]) );
                classificationScores[pos][prb] += scores[p][pos][prb];                          // score accumulation
            }
            classificationScores[pos][prb] /= (double)nPatches;                                 // average score fusion
        }
        classificationScores[pos] = normalizeMinMaxClassScores(classificationScores[pos]);      // score normalization post-fusion
    }

    // performance evaluation

    for (size_t pos = 0; pos < nPositives; pos++)
    {
        logger << "Performance evaluation results for target " << positivesID[pos] << ":" << std::endl;
        eval_PerformanceClassificationScores(classificationScores[pos], probeGroundTruths[pos]);
    }
    logger << "Summary of performance evaluation results:" << std::endl;
    eval_PerformanceClassificationSummary(positivesID, classificationScores, probeGroundTruths);

    return 0;
}

/*
    Evaluates various performance mesures of classification scores according to ground truths
*/
void eval_PerformanceClassificationScores(std::vector<double> normScores, std::vector<int> probeGroundTruths)
{
    std::vector<double> FPR, TPR;
    eval_PerformanceClassificationScores(normScores, probeGroundTruths, FPR, TPR);
}

/*
    Evaluates various performance mesures of classification scores according to ground truths and return (FPR,TPR) results
*/
void eval_PerformanceClassificationScores(std::vector<double> normScores, std::vector<int> probeGroundTruths,
                                          std::vector<double>& FPR, std::vector<double>& TPR)
{
    ASSERT_LOG(normScores.size() == probeGroundTruths.size(), "Number of classification scores and ground truth must match");

    logstream logger(LOGGER_FILE);

    // Evaluate results    
    int steps = 100;
    FPR = std::vector<double>(steps + 1, 0);
    TPR = std::vector<double>(steps + 1, 0);
    for (int i = 0; i <= steps; i++)
    {
        int FP, FN, TP, TN;
        double T = (double)(steps - i) / (double)steps; // Go in reverse threshold order to respect 'calcAUC' requirement
        countConfusionMatrix(normScores, probeGroundTruths, T, &TP, &TN, &FP, &FN);
        TPR[i] = calcTPR(TP, FN);
        FPR[i] = calcFPR(FP, TN);
    }
    double AUC = calcAUC(FPR, TPR);
    double pAUC10 = calcAUC(FPR, TPR, 0.10);
    double pAUC20 = calcAUC(FPR, TPR, 0.20);
    for (size_t j = 0; j < FPR.size(); j++)
        logger << "(FPR,TPR)[" << j << "] = " << FPR[j] << "," << TPR[j] << std::endl;
    logger << "AUC = " << AUC << std::endl              // Area Under ROC Curve
           << "pAUC(10%) = " << pAUC10 << std::endl     // Partial Area Under ROC Curve (FPR=10%)
           << "pAUC(20%) = " << pAUC20 << std::endl;    // Partial Area Under ROC Curve (FPR=20%)
}

/*
    Makes a summary evaluation and display of multiple targets using their corresponding (FPR,TPR) values.

    Format Requirements:

        - positivesID:          matches the first dimension of the results multi-vectors
        - normScores:           2D-vector indexed as [target][probe] scores
        - probeGroundTruths:    2D-vector indexed as [target][probe] ground truths matching scores indexes
*/
void eval_PerformanceClassificationSummary(std::vector<std::string> positivesID, 
                                           xstd::mvector<2, double> normScores, xstd::mvector<2, int> probeGroundTruths)
{
    // check targets
    size_t nTargets = positivesID.size();
    ASSERT_LOG(nTargets > 0, "Cannot make performance classification summary without target IDs");
    ASSERT_LOG(nTargets == normScores.size(), "Number of target IDs must match number of scores (1st dimension)");
    ASSERT_LOG(nTargets == probeGroundTruths.size(), "Number of target IDs must match number of ground truths (1st dimension)");

    // evaluate results    
    int steps = 100;
    size_t dimsSummary[2]{ nTargets, 4 };                   
    size_t dimsThresholds[2]{ nTargets, steps + 1 };        
    xstd::mvector<2, double> summaryResults(dimsSummary);   // [target][0: AUC | 1: pAUC(10%) | 2: pAUC(20%) | 3: AUPR](double)
    xstd::mvector<2, ConfusionMatrix> CM(dimsThresholds);   // [target][threshold](ConfusionMatrix)
    for (size_t pos = 0; pos < nTargets; pos++)
    {
        for (int i = 0; i <= steps; i++)
        {
            ConfusionMatrix cm;
            double T = (double)(steps - i) / (double)steps; // Go in reverse threshold order to respect 'calcAUC' requirement
            countConfusionMatrix(normScores[pos], probeGroundTruths[pos], T, &cm);
            CM[pos][i] = cm;
        }
        summaryResults[pos][0] = calcAUC(CM[pos]);          // AUC
        summaryResults[pos][1] = calcAUC(CM[pos], 0.10);    // pAUC(10%)
        summaryResults[pos][2] = calcAUC(CM[pos], 0.20);    // pAUC(20%)
        summaryResults[pos][3] = calcAUPR(CM[pos]);         // AUPR
    }

    // display results
    logstream logger(LOGGER_FILE);
    std::string header = "Target IDs";
    std::vector<std::string> cols = { "      AUC      ", "   pAUC(10%)   ", "   pAUC(20%)   ", "      AUPR     " };
    size_t targetLen = header.size();    
    for (size_t pos = 0; pos < nTargets; pos++)
        targetLen = std::max(positivesID[pos].size(), targetLen);
    targetLen++;
    header += std::string(targetLen - header.size(), ' ');
    for (size_t c = 0; c < cols.size(); c++)
        header += "|" + cols[c];
    logger << header << std::endl << std::string(header.size(), '-') << std::endl;    
    for (size_t pos = 0; pos < nTargets; pos++)
    {
        logger << positivesID[pos] << std::string(targetLen - positivesID[pos].size() - 1, ' ');
        for (size_t c = 0; c < cols.size(); c++)
            logger << " |" << std::setw(cols[c].size() - 1) << summaryResults[pos][c];        
        logger << std::endl;
    }
}
