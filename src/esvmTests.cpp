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
#include "mvector.hpp"      // Multi-Dimension vectors

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
           << tab << tab << "ESVM_USE_HOG:                    " << ESVM_USE_HOG << std::endl
           << tab << tab << "ESVM_USE_LBP:                    " << ESVM_USE_LBP << std::endl
           << tab << tab << "ESVM_USE_SYNTHETIC_GENERATION:   " << ESVM_USE_SYNTHETIC_GENERATION << std::endl
           << tab << tab << "ESVM_DUPLICATE_COUNT:            " << ESVM_DUPLICATE_COUNT << std::endl
           << tab << tab << "ESVM_USE_FEATURES_NORMALIZATION: " << ESVM_USE_FEATURES_NORMALIZATION << std::endl
           << tab << tab << "ESVM_USE_PREDICT_PROBABILITY:    " << ESVM_USE_PREDICT_PROBABILITY << std::endl
           << tab << tab << "ESVM_POSITIVE_CLASS:             " << ESVM_POSITIVE_CLASS << std::endl
           << tab << tab << "ESVM_NEGATIVE_CLASS:             " << ESVM_NEGATIVE_CLASS << std::endl
           << tab << tab << "ESVM_WEIGHTS_MODE:               " << ESVM_WEIGHTS_MODE << std::endl
           << tab << tab << "ESVM_WRITE_DATA_FILES:           " << ESVM_WRITE_DATA_FILES << std::endl
           << tab << tab << "ESVM_READ_DATA_FILES:            " << ESVM_READ_DATA_FILES << std::endl
           << tab << "TEST:" << std::endl
           << tab << tab << "TEST_CHOKEPOINT_SEQUENCES_MODE:  " << TEST_CHOKEPOINT_SEQUENCES_MODE << std::endl
           << tab << tab << "TEST_IMAGE_PATHS:                " << TEST_IMAGE_PATHS << std::endl
           << tab << tab << "TEST_IMAGE_PROCESSING:           " << TEST_IMAGE_PROCESSING << std::endl
           << tab << tab << "TEST_MULTI_LEVEL_VECTORS:        " << TEST_MULTI_LEVEL_VECTORS << std::endl
           << tab << tab << "TEST_NORMALIZATION:              " << TEST_NORMALIZATION << std::endl
           << tab << tab << "TEST_ESVM_BASIC_FUNCTIONALITY:   " << TEST_ESVM_BASIC_FUNCTIONALITY << std::endl
           << tab << tab << "TEST_ESVM_BASIC_STILL2VIDEO:     " << TEST_ESVM_BASIC_STILL2VIDEO << std::endl
           << tab << tab << "TEST_ESVM_TITAN:                 " << TEST_ESVM_TITAN << std::endl
           << tab << tab << "TEST_ESVM_SAMAN:                 " << TEST_ESVM_SAMAN << std::endl;

    return 0;
}

int test_imagePaths()
{    
    // Local
    ASSERT_LOG(bfs::is_directory(roiVideoImagesPath), "Cannot find ROI directory");
    ASSERT_LOG(bfs::is_directory(refStillImagesPath), "Cannot find REF directory");
    ASSERT_LOG(checkPathEndSlash(roiVideoImagesPath), "ROI directory doesn't end with slash character");
    ASSERT_LOG(checkPathEndSlash(refStillImagesPath), "REF directory doesn't end with slash character");
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
            return -1;
        }
    }    

    // check pixel values of patches
    if (!cv::countNonZero(testPatches[0] != cv::Mat(2, 2, CV_32S, { 1,2,7,8 })))
    {
        logger << "Invalid data for patch 0" << std::endl << testPatches[0] << std::endl;
        return -1;
    }
    if (!cv::countNonZero(testPatches[1] != cv::Mat(2, 2, CV_32S, { 3,4,9,10 })))
    {
        logger << "Invalid data for patch 1" << std::endl << testPatches[1] << std::endl;
        return -1;
    }
    if (!cv::countNonZero(testPatches[2] != cv::Mat(2, 2, CV_32S, { 5,6,11,12 })))
    {
        logger << "Invalid data for patch 2" << std::endl << testPatches[2] << std::endl;
        return -1;
    }
    if (!cv::countNonZero(testPatches[3] != cv::Mat(2, 2, CV_32S, { 13,14,19,20 })))
    {
        logger << "Invalid data for patch 3" << std::endl << testPatches[3] << std::endl;
        return -1;
    }
    if (!cv::countNonZero(testPatches[4] != cv::Mat(2, 2, CV_32S, { 15,16,21,22 })))
    {
        logger << "Invalid data for patch 4" << std::endl << testPatches[4] << std::endl;
        return -1;
    }
    if (!cv::countNonZero(testPatches[5] != cv::Mat(2, 2, CV_32S, { 17,18,23,24 })))
    {
        logger << "Invalid data for patch 5" << std::endl << testPatches[5] << std::endl;
        return -1;
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
    std::vector < FeatureVector > positives(20);
    std::vector < FeatureVector > negatives(20);  
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
    std::vector< FeatureVector > samples(6);
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

// Tests sample file reading functionality of ESVM (index and value parsing)
int test_runBasicExemplarSvmSampleFileRead()
{
    logstream logger(LOGGER_FILE);
    logger << "Starting basic Exemplar-SVM sample file reading test..." << std::endl;

    // create test sample files
    std::string testDir = "test_sample-read-file/";
    bfs::create_directory(testDir);
    std::string validIndexSampleFileName1 = testDir + "test_valid-index-samples1.data";
    std::string wrongIndexSampleFileName1 = testDir + "test_wrong-index-samples1.data";
    std::string wrongIndexSampleFileName2 = testDir + "test_wrong-index-samples2.data";
    std::string wrongIndexSampleFileName3 = testDir + "test_wrong-index-samples3.data";
    std::string validSparseSampleFileName1 = testDir + "test_valid-sparse-samples1.data";
    std::string wrongSparseSampleFileName1 = testDir + "test_wrong-sparse-samples1.data";
    std::ofstream validIndexSampleFile1(validIndexSampleFileName1);
    std::ofstream wrongIndexSampleFile1(wrongIndexSampleFileName1);
    std::ofstream wrongIndexSampleFile2(wrongIndexSampleFileName2);
    std::ofstream wrongIndexSampleFile3(wrongIndexSampleFileName3);
    std::ofstream validSparseSampleFile1(validSparseSampleFileName1);
    std::ofstream wrongSparseSampleFile1(wrongSparseSampleFileName1);

    // fill test sample files
    validIndexSampleFile1 << std::to_string(ESVM_POSITIVE_CLASS) << " 1:10.999 2:20.111 3:30.555 4:40.333 5:50.777 -1:0" << std::endl;
    validIndexSampleFile1 << std::to_string(ESVM_NEGATIVE_CLASS) << " 1:90.123 2:80.456 3:-70.78 4:-90000 5:-50.00 -1:0" << std::endl;
    wrongIndexSampleFile1 << std::to_string(ESVM_NEGATIVE_CLASS) << " 1:10.999 2:20.111 5:30.555 4:40.333 3:50.777 -1:0" << std::endl;  // 5->4->3
    wrongIndexSampleFile2 << std::to_string(ESVM_NEGATIVE_CLASS) << " 1:10.999 2:20.111 3:30.555 5:40.333 5:50.777 -1:0" << std::endl;  // 3->5->5
    wrongIndexSampleFile3 << std::to_string(ESVM_NEGATIVE_CLASS) << " 1:10.999 2:20.111 3:30.555 -1:11111 5:50.777 -1:0" << std::endl;  // -1->5

    // close test sample files
    if (validIndexSampleFile1.is_open()) validIndexSampleFile1.close();
    if (wrongIndexSampleFile1.is_open()) wrongIndexSampleFile1.close();
    if (wrongIndexSampleFile2.is_open()) wrongIndexSampleFile2.close();
    if (wrongIndexSampleFile3.is_open()) wrongIndexSampleFile3.close();
    if (validSparseSampleFile1.is_open()) validSparseSampleFile1.close();
    if (wrongSparseSampleFile1.is_open()) wrongSparseSampleFile1.close();

    // tests
    ESVM esvm;
    std::vector<FeatureVector> samples;
    std::vector<int> targetOutputs;
    try
    {
        esvm.readSampleDataFile(validIndexSampleFileName1, samples, targetOutputs);
        ASSERT_LOG(samples.size() == 2, "File reading should result in 2 loaded feature vector samples");
        
    }
    catch (std::exception& ex)
    {

    }

    // asserts to check:
    //   ASSERT_LOG(target == ESVM_POSITIVE_CLASS || target == ESVM_NEGATIVE_CLASS)
    //   ASSERT_LOG(offset != std::string::npos, "Failed to find feature 'index:value' delimiter");
    //   ASSERT_LOG(index - prev > 0, "Feature indexes must be in ascending order");
    //   assert skip index reading if -1 found before
    //   ASSERT_LOG(nFeatures == features.size()
    //   ASSERT_LOG(trainingFile.eof()  ===>   add final fv without newline?





    // delete test sample files
    bfs::remove_all(testDir);

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
    /* Training Targets:        single high quality still image for enrollment (same as Saman paper) */
    std::vector<std::string> positivesID = { "0011", "0012", "0013", "0016", "0020" };
    /* Training Non-Targets:    as many video negatives as possible */
    std::vector<std::string> negativesID = { "0001", "0002", "0006", "0007", "0010",
                                             "0017", "0018", "0019", "0024", "0025",
                                             "0027", "0028", "0030" };
    /* Testing Probes:          some video positives and negatives */
    std::vector<std::string> probesID = { "0004", "0009", "0011", "0012", "0013",
                                          "0016", "0020", "0023", "0026", "0029" }; 

    // Display and output
    cv::namedWindow(WINDOW_NAME);
    logstream logger(LOGGER_FILE);
    size_t nPatches = patchCounts.width * patchCounts.height;
    if (nPatches == 0) nPatches = 1;    
    logger << "Starting single sample per person still-to-video full ChokePoint test..." << std::endl
           << "   useSyntheticPositives: " << ESVM_USE_SYNTHETIC_GENERATION << std::endl
           << "   imageSize:             " << imageSize << std::endl
           << "   patchCounts:           " << patchCounts << std::endl;
    
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
    size_t nDuplications = ESVM_DUPLICATE_COUNT;
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
        #if ESVM_USE_SYNTHETIC_GENERATION
        
            // Get original positive image with preprocessing but without patches splitting
            cv::Mat img = imPreprocess(refStillImagesPath + "roiID" + positivesID[pos] + ".jpg",
                                       imageSize, cv::Size(1,1), WINDOW_NAME, cv::IMREAD_COLOR)[0];
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
        #else/*!ESVM_USE_SYNTHETIC_GENERATION*/
        
            //// matPositiveSamples[pos] = std::vector< std::vector< cv::Mat> >(1);
            std::vector<cv::Mat> patches = imPreprocess(refStillImagesPath + "roiID" + positivesID[pos] + ".jpg",
                                                        imageSize, patchCounts, WINDOW_NAME, cv::IMREAD_COLOR);
            for (size_t p = 0; p < nPatches; p++)
                matPositiveSamples[pos][0][p] = patches[p];
        
        #endif/*ESVM_USE_SYNTHETIC_GENERATION*/
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
            probeID.clear();
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
                                std::vector<cv::Mat> patches = imPreprocess(itDir->path().string(), imageSize,
                                                                            patchCounts, WINDOW_NAME, cv::IMREAD_GRAYSCALE);
                                for (size_t p = 0; p < nPatches; p++)
                                    matNegativeSamples[neg][p] = patches[p];
                            }
                            else if (contains(probesID, strID))
                            {
                                size_t prb = matProbeSamples.size();
                                matProbeSamples.push_back(xstd::mvector<1, cv::Mat>(nPatches));
                                std::vector<cv::Mat> patches = imPreprocess(itDir->path().string(), imageSize,
                                                                            patchCounts, WINDOW_NAME, cv::IMREAD_GRAYSCALE);
                                for (size_t p = 0; p < nPatches; p++)
                                    matProbeSamples[prb][p] = patches[p];

                                probeID.push_back(strID);
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
                            
            // Feature extraction of negatives and probes
            size_t nProbes = matProbeSamples.size();
            size_t nNegatives = matNegativeSamples.size();
            logger << "Feature extraction of negative and probe samples (total negatives: " << nNegatives
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
            #if !ESVM_USE_FEATURES_NORMALIZATION
            logger << "Skipping features normalization" << std::endl;
            #else/*ESVM_USE_FEATURES_NORMALIZATION*/
            logger << "Getting feature normalization values..." << std::endl;
            size_t dimsMinMax[2] { nPatches, nDescriptors };
            size_t dimsAllVectors[3] { nPatches, nDescriptors, nPositives * nRepresentations + nNegatives + nProbes };            
            xstd::mvector<2, FeatureVector> minFeatures(dimsMinMax);                // [patch][descriptor]
            xstd::mvector<2, FeatureVector> maxFeatures(dimsMinMax);                // [patch][descriptor]
            xstd::mvector<3, FeatureVector> allFeatureVectors(dimsAllVectors);      // [patch][descriptor][sample]
            /// ############################################# #pragma omp parallel for
            for (size_t p = 0; p < nPatches; p++)
            {
                //// allFeatureVectors[p] = std::vector< std::vector< FeatureVector > >(nFeatureExtraction);
                for (size_t d = 0; d < nDescriptors; d++)
                {
                    size_t s = 0;  // Sample index
                    for (size_t pos = 0; pos < nPositives; pos++)
                        for (size_t r = 0; r < nRepresentations; r++)
                            allFeatureVectors[p][d][s++] = fvPositiveSamples[pos][p][d][r];
                    for (size_t neg = 0; neg < nNegatives; neg++)
                        allFeatureVectors[p][d][s++] = fvNegativeSamples[p][d][neg];
                    for (size_t prb = 0; prb < nProbes; prb++)
                        allFeatureVectors[p][d][s++] = fvProbeSamples[p][d][prb];

                    // Min/Max of each (patch,feature extraction) combination for normalization 
                    findMinMaxFeatures(allFeatureVectors[p][d], &(minFeatures[p][d]), &(maxFeatures[p][d]));
                    logger << "Found min/max features for (descriptor,patch) " << descriptorNames[d] << "," << p << ":" << std::endl
                           << "   MIN: " << featuresToVectorString(minFeatures[p][d]) << std::endl
                           << "   MAX: " << featuresToVectorString(maxFeatures[p][d]) << std::endl;
                }
            }      
            
            logger << "Applying features normalization..." << std::endl;
            for (size_t p = 0; p < nPatches; p++)
            {    
                for (size_t d = 0; d < nDescriptors; d++)
                {
                    FeatureVector mins = minFeatures[p][d];
                    FeatureVector maxs = maxFeatures[p][d];
                    for (size_t pos = 0; pos < nPositives; pos++)
                        for (size_t r = 0; r < nRepresentations; r++)
                            fvPositiveSamples[pos][p][d][r] = normalizeMinMaxPerFeatures(fvPositiveSamples[pos][p][d][r], mins, maxs);
                    for (size_t neg = 0; neg < nNegatives; neg++)
                        fvNegativeSamples[p][d][neg] = normalizeMinMaxPerFeatures(fvNegativeSamples[p][d][neg], mins, maxs);
                    for (size_t prb = 0; prb < nProbes; prb++)
                        fvProbeSamples[p][d][prb] = normalizeMinMaxPerFeatures(fvProbeSamples[p][d][prb], mins, maxs);
                }
            }            
            #endif/*ESVM_USE_FEATURES_NORMALIZATION*/

            // ESVM samples files for each (sequence,positive,feature-extraction,train/test,patch) combination
            #if ESVM_WRITE_DATA_FILES
            logger << "Writing ESVM train/test samples files..." << std::endl;
            for (size_t p = 0; p < nPatches; p++)
            {
                std::string strPatch = std::to_string(p);
                for (size_t d = 0; d < nDescriptors; d++)
                {
                    for (size_t pos = 0; pos < nPositives; pos++)
                    {        
                        std::string fileTemplate = "chokepoint-" + seq + "-id" + positivesID[pos] + "-" +
                                                    descriptorNames[d] + "-patch" + strPatch;
                        std::string trainFileName = fileTemplate + "-train.data";
                        std::string testFileName = fileTemplate + "-test.data";
                        logger << "   Writing ESVM files:" << std::endl
                               << "      '" << trainFileName << "'" << std::endl
                               << "      '" << testFileName << "'" << std::endl;

                        std::ofstream trainFile(trainFileName);
                        std::ofstream testFile(testFileName);
                
                        // Add other gallery positives than the current one as additional negative representations (counter examples)                    
                        for (size_t galleryPos = 0; galleryPos < nPositives; galleryPos++)
                            for (size_t r = 0; r < nRepresentations; r++)
                        {
                            int gt = (pos == galleryPos ? ESVM_POSITIVE_CLASS : ESVM_NEGATIVE_CLASS);
                            trainFile << featuresToSvmString(fvPositiveSamples[galleryPos][p][d][r], gt) << std::endl;
                        }
                        for (size_t neg = 0; neg < nNegatives; neg++)
                            trainFile << featuresToSvmString(fvNegativeSamples[p][d][neg], ESVM_NEGATIVE_CLASS) << std::endl;
                        for (size_t prb = 0; prb < nProbes; prb++)
                            testFile << featuresToSvmString(fvProbeSamples[p][d][prb], probeGroundTruth[pos][prb]) << std::endl;
                    }
                }
            }
            #endif/*ESVM_WRITE_DATA_FILES*/
            
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

                            #if ESVM_WRITE_DATA_FILES
                            std::string esvmModelFile = "chokepoint-" + seq + "-id" + positivesID[pos] + "-" +
                                                        descriptorNames[d] + "-patch" + std::to_string(p) + ".model";
                            logger << "Saving Exemplar-SVM model to file..." << std::endl;
                            bool isSaved = esvmModels[pos][p][d].saveModelFile(esvmModelFile);
                            logger << std::string(isSaved ? "Saved" : "Failed to save") + 
                                      " Exemplar-SVM model to file: '" + esvmModelFile + "'" << std::endl;                            
                            #endif/*ESVM_WRITE_DATA_FILES*/
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

                            std::string probeGT = (probeID[prb] == positivesID[pos] ? "positive" : "negative");
                            logger << "Score for patch " << p << " of probe " << prb << " (ID" << probeID[prb] << ", "
                                   << probeGT << "): " << patchScoresNorm[prb] << std::endl;
                        }
                    }
                    for (size_t prb = 0; prb < nProbes; prb++)
                    {
                        // average of score accumulation for fusion per patch
                        descriptorScores[prb] /= (double)nPatches;
                        // accumulation with normalized patch-fusioned scores for descriptor-based score fusion
                        fusionScores[prb] = descriptorScores[prb];
                        std::string probeGT = (probeID[prb] == positivesID[pos] ? "positive" : "negative");
                        logger << "Score for descriptor " << descriptorNames[d] << " (patch-fusion) of probe " << prb
                               << " (ID" << probeID[prb] << ", " << probeGT << "): " << descriptorScores[prb] << std::endl;
                    }
                    logger << "Performance evaluation for patch-based score fusion for '" + descriptorNames[d] + "' descriptor:" << std::endl;
                    eval_PerformanceClassificationScores(descriptorScores, probeGroundTruth[pos]);
                }

                int nCombined = nDescriptors * nPatches;
                for (size_t prb = 0; prb < nProbes; prb++)
                {
                    // average of score accumulation for fusion per descriptor
                    std::string probeGT = (probeID[prb] == positivesID[pos] ? "positive" : "negative");
                    fusionScores[prb] /= (double)nDescriptors;
                    combinedScores[prb] /= (double)nCombined;
                    combinedScoresRaw[prb] /= (double)nCombined;
                    
                    logger << "Score fusion (descriptor,patch) of probe " << prb << " (ID" << probeID[prb] << ", "
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
    const std::string negativesDir = "negatives/";
    const std::string probesFileDir = "data_SAMAN_48x48_HOG-descriptor+9-patches/";

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
    std::vector<int> probesID = { 3, 4, 5, 6, 9, 10, 11, 12, 23, 24 };
    
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
    size_t dimsProbes[2] = { nPatches, 0 };
    xstd::mvector<2, ESVM> esvm(dimsPositives);                             // [target][patch](ESVM)
    xstd::mvector<2, FeatureVector> fvPositiveSamples(dimsPositives);       // [target][patch](FeatureVector)
    xstd::mvector<2, FeatureVector> fvProbeLoadedSamples(dimsProbes);       // [patch][probe](FeatureVector) - reversed indexing for easier access
    std::vector<std::string> probesLoadedID;                                // [probe](string)

    cv::namedWindow(WINDOW_NAME);

    // Load positive stills and extract features
    logger << "Loading positive enroll image stills..." << std::endl;
    for (int pos = 0; pos < nPositives; pos++)
    {
        std::string stillPath = roiChokePointEnrollStillPath + "roi" + buildChokePointIndividualID(positivesID[pos], true) + ".jpg";
        std::vector<cv::Mat> patches = imPreprocess(stillPath, imageSize, patchCounts, WINDOW_NAME, cv::IMREAD_GRAYSCALE);
        fvPositiveSamples.push_back(xstd::mvector<1, FeatureVector>(nPatches));
        for (size_t p = 0; p < nPatches; p++)
            fvPositiveSamples[pos][p] = hog.compute(patches[p]);
    }

    // Load probe images and extract features if required (ESVM_READ_DATA_FILES option 32)
    #if ESVM_READ_DATA_FILES & 0b00100000
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
                            std::vector<cv::Mat> patches = imPreprocess(itDir->path().string(), imageSize,
                                                                        patchCounts, WINDOW_NAME, cv::IMREAD_GRAYSCALE);
                            for (size_t p = 0; p < nPatches; p++)
                                fvProbeLoadedSamples[p].push_back(hog.compute(patches[p]));

                            probesLoadedID.push_back(buildChokePointIndividualID(id, true));
                        } 
                    }
                }
            }                        
        }
    }
    #endif/*ESVM_READ_DATA_FILES & (32)*/
    cv::destroyWindow(WINDOW_NAME);

    // train and test ESVM
    ESVM tmpLoadESVM;   // temporary ESVM only for file loading
    for (int pos = 0; pos < nPositives; pos++)
    {
        std::vector<int> probeGroundTruthsPreGen, probeGroundTruthsLoaded;              // [probe](int)
        std::vector<double> probeFusionScoresPreGen, probeFusionScoresLoaded;           // [probe](double)
        xstd::mvector<2, double> probePatchScoresPreGen, probePatchScoresLoaded;        // [patch][probe](double)
        std::string posID = buildChokePointIndividualID(positivesID[pos], true);
        logger << "Starting ESVM training/testing for '" << posID << "'..." << std::endl;
        for (int p = 0; p < nPatches; p++)
        {
            // load feature vector files
            std::string strPatch = std::to_string(p);
            std::string negativeTrainFile = negativesDir + "negatives-patch" + strPatch + ".data";
            std::vector<FeatureVector> fvNegativePatch;
            tmpLoadESVM.readSampleDataFile(negativeTrainFile, fvNegativePatch, probeGroundTruthsPreGen);

            // train with positive extracted features and negative loaded features
            logger << "Starting ESVM training for '" << posID << "', patch " << strPatch << "..." << std::endl;
            std::vector<FeatureVector> singlePositivePatch = { fvPositiveSamples[pos][p] };
            esvm[pos][p] = ESVM(singlePositivePatch, fvNegativePatch, posID + "-" + strPatch);

            // test against pre-generated probes and loaded probes
            #if ESVM_READ_DATA_FILES & 0b00010000   // (16) use pre-generated probe sample file
            logger << "Starting ESVM testing for '" << posID << "', patch " << strPatch << " (probe pre-generated samples files)..." << std::endl;
            std::string probePreGenTestFile = probesFileDir + "test-target" + posID + "-patch" + strPatch + ".data";
            probePatchScoresPreGen.push_back(esvm[pos][p].predict(probePreGenTestFile));
            logger << "DEBUG -- " << probePatchScoresPreGen.size() << std::endl;
            logger << "DEBUG -- " << probePatchScoresPreGen[0].size() << std::endl;
            #endif/*ESVM_READ_DATA_FILES & (16)*/
            #if ESVM_READ_DATA_FILES & 0b00100000   // (32) use feature extraction on probe images
            logger << "Starting ESVM testing for '" << posID << "', patch " << strPatch << " (probe images and extract feature)..." << std::endl;
            probePatchScoresLoaded.push_back(esvm[pos][p].predict(fvProbeLoadedSamples[p]));
            #endif/*ESVM_READ_DATA_FILES & (32)*/
        }
        logger << "Starting score fusion and normalization for '" << posID << "'..." << std::endl;

        /* -------------------------------------------
           (16) use pre-generated probe sample file 
        ------------------------------------------- */
        #if ESVM_READ_DATA_FILES & 0b00010000 

        // accumulated sum of scores for score fusion
        logger << "DEBUG -- N PROBES" << std::endl;
        int nProbesPreGen = probePatchScoresPreGen[0].size();
        logger << "DEBUG -- N PROBES: " << nProbesPreGen << std::endl;
        probeFusionScoresPreGen = std::vector<double>(nProbesPreGen, 0.0);
        logger << "DEBUG -- N FUSION: " << probeFusionScoresPreGen.size() << std::endl;
        logger << "DEBUG -- N SCORES: " << probePatchScoresPreGen.size() << std::endl;        
        for (int p = 0; p < nPatches; p++)
            for (int prb = 0; prb < nProbesPreGen; prb++)
                probeFusionScoresPreGen[prb] += probePatchScoresPreGen[p][prb];
        
        // average accumulated scores and execute post-fusion normalization
        logger << "DEBUG -- CUMUL" << std::endl;
        for (int prb = 0; prb < nProbesPreGen; prb++)
            probeFusionScoresPreGen[prb] /= (double)nPatches;
        probeFusionScoresPreGen = normalizeMinMaxClassScores(probeFusionScoresPreGen);
        
        // evaluate results with fusioned patch scores
        logger << "Performance evaluation for pre-generated probes (no pre-norm, post-fusion norm) of '" << posID << "':" << std::endl;
        eval_PerformanceClassificationScores(probeFusionScoresPreGen, probeGroundTruthsPreGen);
        
        #endif/*ESVM_READ_DATA_FILES & (16)*/
        
        /* ----------------------------------------------
           (32) use feature extraction on probe images
        ---------------------------------------------- */
        #if ESVM_READ_DATA_FILES & 0b00100000

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
        
        #endif/*ESVM_READ_DATA_FILES & (32)*/       
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

    // Display and output
    cv::namedWindow(WINDOW_NAME);
    logstream logger(LOGGER_FILE);
    size_t nPatches = patchCounts.width * patchCounts.height;
    if (nPatches == 0) nPatches = 1;    
    logger << "Starting single sample per person still-to-video full ChokePoint test..." << std::endl
           << "   useSyntheticPositives: " << useSyntheticPositives << std::endl
           << "   imageSize:             " << imageSize << std::endl
           << "   patchCounts:           " << patchCounts << std::endl;
    
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
            cv::Mat img = imPreprocess(positiveImageStills[pos].Path, imageSize, cv::Size(1,1), WINDOW_NAME, cv::IMREAD_COLOR)[0];
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
            std::vector<cv::Mat> patches = imPreprocess(positiveImageStills[pos].Path, imageSize, patchCounts, WINDOW_NAME, cv::IMREAD_COLOR);
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
                                std::vector<cv::Mat> patches = imPreprocess(itDir->path().string(), imageSize,
                                                                            patchCounts, WINDOW_NAME, cv::IMREAD_GRAYSCALE);
                                for (size_t p = 0; p < nPatches; p++)
                                    matNegativeSamples[neg][p] = patches[p];
    } } } } } } }   // End of negatives loading

    // Load probe samples
    /*
    else if (contains(probesID, strID))
{
    size_t prb = matProbeSamples.size();
    matProbeSamples.push_back(xstd::mvector<1, cv::Mat>(nPatches));
    std::vector<cv::Mat> patches = imPreprocess(itDir->path().string(), imageSize,
        patchCounts, WINDOW_NAME, cv::IMREAD_GRAYSCALE);
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

    std::vector<std::string> positivesID = { "ID0003", "ID0005", "ID0006", "ID0010", "ID0024" };
    size_t nPositives = positivesID.size();
    size_t nPatches = 9;    
    size_t nProbes = 0;     // set when read from testing file
    #if TEST_ESVM_SAMAN == 1
    size_t nFeatures = 128;
    std::string dataFileDir = "data_SAMAN_48x48_HOG-PCA-descriptor+9-patches/";
    #elif TEST_ESVM_SAMAN == 2
    size_t nFeatures = 588;
    std::string dataFileDir = "data_SAMAN_48x48_HOG-descriptor+9-patches/";
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
            std::string strPatch = std::to_string(p);
            logger << "Starting training/testing ESVM evaluation for '" << posID << "', patch " << strPatch << "..." << std::endl;

            // run training / testing from files            
            std::vector<double> probePatchScores;
            std::string trainFile = dataFileDir + "train-target" + posID + "-patch" + strPatch + ".data";
            std::string testFile = dataFileDir + "test-target" + posID + "-patch" + strPatch + ".data";
            esvm[pos][p] = ESVM(trainFile, posID);
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

        ASSERT_LOG(nProbes > 0, "Number of probes should have been initialized and be greater than zero");

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

    return 0;
}

/*
    Evaluates various performance mesures of classification scores according to ground truths
*/
void eval_PerformanceClassificationScores(std::vector<double> normScores, std::vector<int> probeGroundTruths)
{
    ASSERT_LOG(normScores.size() == probeGroundTruths.size(), "Number of classification scores and ground truth must match");

    logstream logger(LOGGER_FILE);

    // Evaluate results
    std::vector<double> TPR, FPR;
    int steps = 100;
    for (int i = 0; i <= steps; i++)
    {
        int FP, FN, TP, TN;
        double T = (double)(steps - i) / (double)steps; // Go in reverse threshold order to respect 'calcAUC' requirement
        countConfusionMatrix(normScores, probeGroundTruths, T, &TP, &TN, &FP, &FN);
        TPR.push_back(calcTPR(TP, FN));
        FPR.push_back(calcFPR(FP, TN));
    }
    double AUC = calcAUC(TPR, FPR);
    double pAUC10 = calcAUC(TPR, FPR, 0.10);
    double pAUC20 = calcAUC(TPR, FPR, 0.20);
    for (size_t j = 0; j < FPR.size(); j++)
        logger << "(FPR,TPR)[" << j << "] = " << FPR[j] << "," << TPR[j] << std::endl;
    logger << "AUC = " << AUC << std::endl              // Area Under ROC Curve
           << "pAUC(10%) = " << pAUC10 << std::endl     // Partial Area Under ROC Curve (FPR=10%)
           << "pAUC(20%) = " << pAUC20 << std::endl;    // Partial Area Under ROC Curve (FPR=20%)
}
