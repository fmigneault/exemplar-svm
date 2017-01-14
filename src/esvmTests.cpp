#include "esvmTests.h"

#include "boost/filesystem.hpp"
namespace bfs = boost::filesystem;

// Builds the string as "P#T_S#_C#", if the individual is non-zero, it adds the sub folder as "P#T_S#_C#\ID"
std::string buildChokePointSequenceString(int portal, PORTAL_TYPE type, int session, int camera, int id)
{
    std::string dir = "P" + std::to_string(portal) + (type == ENTER ? "E" : type == LEAVE ? "L" : "") +
                      "_S" + std::to_string(session) + "_C" + std::to_string(camera);
    return id > 0 ? dir + "/" + buildChokePointIndividualID(id) : dir;
}

std::string buildChokePointIndividualID(int id)
{
    return std::string(id > 9 ? 2 : 3, '0').append(std::to_string(id));
}

int test_imagePatchExtraction(void)
{
    logstream log(LOGGER_FILE);
    log << "Testing image patch extraction..." << std::endl;
    int rawData[24] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 };
    cv::Mat testImg(4, 6, CV_32S, rawData);                                         // 6x4 image with above data filled line by line
    std::vector<cv::Mat> testPatches = imSplitPatches(testImg, cv::Size(3, 2));     // 6 patches of 2x2    
    // check number of patches extracted
    if (testPatches.size() != 6)
    {
        log << "Invalid number of patches extracted (count: " << testPatches.size() << ", expected: 6)" << std::endl;
        return -1;
    }
    // check patch dimensions
    for (int p = 0; p < 6; p++)
    {
        if (testPatches[p].size() != cv::Size(2, 2))
        {
            log << "Invalid image size for patch " << p << " (size: " << testPatches[p].size() << ", expected: (2,2))" << std::endl;
            return -1;
        }
    }    

    // check pixel values of patches
    if (!cv::countNonZero(testPatches[0] != cv::Mat(2, 2, CV_32S, { 1,2,7,8 })))
    {
        log << "Invalid data for patch 0" << std::endl << testPatches[0] << std::endl;
        return -1;
    }
    if (!cv::countNonZero(testPatches[1] != cv::Mat(2, 2, CV_32S, { 3,4,9,10 })))
    {
        log << "Invalid data for patch 1" << std::endl << testPatches[1] << std::endl;
        return -1;
    }
    if (!cv::countNonZero(testPatches[2] != cv::Mat(2, 2, CV_32S, { 5,6,11,12 })))
    {
        log << "Invalid data for patch 2" << std::endl << testPatches[2] << std::endl;
        return -1;
    }
    if (!cv::countNonZero(testPatches[3] != cv::Mat(2, 2, CV_32S, { 13,14,19,20 })))
    {
        log << "Invalid data for patch 3" << std::endl << testPatches[3] << std::endl;
        return -1;
    }
    if (!cv::countNonZero(testPatches[4] != cv::Mat(2, 2, CV_32S, { 15,16,21,22 })))
    {
        log << "Invalid data for patch 4" << std::endl << testPatches[4] << std::endl;
        return -1;
    }
    if (!cv::countNonZero(testPatches[5] != cv::Mat(2, 2, CV_32S, { 17,18,23,24 })))
    {
        log << "Invalid data for patch 5" << std::endl << testPatches[5] << std::endl;
        return -1;
    }
    return 0;
}

#if 0 // TMP
int test_runBasicExemplarSvmFunctionalities(void)
{
    // ------------------------------------------------------------------------------------------------------------------------
    // window to display loaded images and stream for console+file output
    // ------------------------------------------------------------------------------------------------------------------------    
    cv::namedWindow(WINDOW_NAME);
    logstream log(LOGGER_FILE);
    log << "Starting basic Exemplar-SVM functionality test..." << std::endl;

    // ------------------------------------------------------------------------------------------------------------------------
    // C++ parameters
    // ------------------------------------------------------------------------------------------------------------------------
    /* Positive training samples */
    std::string targetName = "person_6";
    const int NB_POSITIVE_IMAGES = 13;
    cv::Mat cvPositiveSamples[NB_POSITIVE_IMAGES];
    log << "Loading positive training samples..." << std::endl;
    cvPositiveSamples[0]  = imReadAndDisplay(roiVideoImagesPath + "person_6/000246.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvPositiveSamples[1]  = imReadAndDisplay(roiVideoImagesPath + "person_6/000247.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvPositiveSamples[2]  = imReadAndDisplay(roiVideoImagesPath + "person_6/000250.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvPositiveSamples[3]  = imReadAndDisplay(roiVideoImagesPath + "person_6/000255.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvPositiveSamples[4]  = imReadAndDisplay(roiVideoImagesPath + "person_6/000260.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvPositiveSamples[5]  = imReadAndDisplay(roiVideoImagesPath + "person_6/000265.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvPositiveSamples[6]  = imReadAndDisplay(roiVideoImagesPath + "person_6/000270.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvPositiveSamples[7]  = imReadAndDisplay(roiVideoImagesPath + "person_6/000280.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvPositiveSamples[8]  = imReadAndDisplay(roiVideoImagesPath + "person_6/000285.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvPositiveSamples[9]  = imReadAndDisplay(roiVideoImagesPath + "person_6/000286.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvPositiveSamples[10] = imReadAndDisplay(roiVideoImagesPath + "person_6/000290.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvPositiveSamples[11] = imReadAndDisplay(roiVideoImagesPath + "person_6/000295.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvPositiveSamples[12] = imReadAndDisplay(roiVideoImagesPath + "person_6/000300.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    /* Negative training samples */
    const int NB_NEGATIVE_IMAGES = 36;
    cv::Mat cvNegativeSamples[NB_NEGATIVE_IMAGES];
    log << "Loading negative training samples..." << std::endl;
    cvNegativeSamples[0]  = imReadAndDisplay(roiVideoImagesPath + "person_16/000350.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[1]  = imReadAndDisplay(roiVideoImagesPath + "person_16/000355.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[2]  = imReadAndDisplay(roiVideoImagesPath + "person_16/000360.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[3]  = imReadAndDisplay(roiVideoImagesPath + "person_16/000361.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[4]  = imReadAndDisplay(roiVideoImagesPath + "person_16/000365.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[5]  = imReadAndDisplay(roiVideoImagesPath + "person_16/000370.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[6]  = imReadAndDisplay(roiVideoImagesPath + "person_20/000410.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[7]  = imReadAndDisplay(roiVideoImagesPath + "person_20/000415.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[8]  = imReadAndDisplay(roiVideoImagesPath + "person_20/000420.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[9]  = imReadAndDisplay(roiVideoImagesPath + "person_20/000425.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[10] = imReadAndDisplay(roiVideoImagesPath + "person_23/000435.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[11] = imReadAndDisplay(roiVideoImagesPath + "person_23/000440.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[12] = imReadAndDisplay(roiVideoImagesPath + "person_23/000445.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[13] = imReadAndDisplay(roiVideoImagesPath + "person_23/000450.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[14] = imReadAndDisplay(roiVideoImagesPath + "person_23/000455.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[15] = imReadAndDisplay(roiVideoImagesPath + "person_23/000460.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[16] = imReadAndDisplay(roiVideoImagesPath + "person_32/000495.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[17] = imReadAndDisplay(roiVideoImagesPath + "person_32/000500.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[18] = imReadAndDisplay(roiVideoImagesPath + "person_32/000505.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[19] = imReadAndDisplay(roiVideoImagesPath + "person_32/000510.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[20] = imReadAndDisplay(roiVideoImagesPath + "person_32/000515.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[21] = imReadAndDisplay(roiVideoImagesPath + "person_32/000520.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[22] = imReadAndDisplay(roiVideoImagesPath + "person_32/000525.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[23] = imReadAndDisplay(roiVideoImagesPath + "person_34/000540.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[24] = imReadAndDisplay(roiVideoImagesPath + "person_34/000545.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[25] = imReadAndDisplay(roiVideoImagesPath + "person_34/000550.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[26] = imReadAndDisplay(roiVideoImagesPath + "person_34/000560.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[27] = imReadAndDisplay(roiVideoImagesPath + "person_34/000570.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[28] = imReadAndDisplay(roiVideoImagesPath + "person_34/000575.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[29] = imReadAndDisplay(roiVideoImagesPath + "person_34/000585.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[30] = imReadAndDisplay(roiVideoImagesPath + "person_40/000670.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[31] = imReadAndDisplay(roiVideoImagesPath + "person_40/000675.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[32] = imReadAndDisplay(roiVideoImagesPath + "person_40/000680.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[33] = imReadAndDisplay(roiVideoImagesPath + "person_40/000685.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[34] = imReadAndDisplay(roiVideoImagesPath + "person_40/000690.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    cvNegativeSamples[35] = imReadAndDisplay(roiVideoImagesPath + "person_40/000700.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);
    /* Probe testing samples */
    const int NB_PROBE_IMAGES = 4;
    cv::Mat cvProbeSamples[NB_PROBE_IMAGES];
    log << "Loading probe testing samples..." << std::endl;
    cvProbeSamples[0] = imReadAndDisplay(roiVideoImagesPath + "person_9/000295.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);    // Negative
    cvProbeSamples[1] = imReadAndDisplay(roiVideoImagesPath + "person_6/000275.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);    // Positive
    cvProbeSamples[2] = imReadAndDisplay(roiVideoImagesPath + "person_37/000541.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);   // Negative
    cvProbeSamples[3] = imReadAndDisplay(roiVideoImagesPath + "person_45/000680.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE);   // Negative

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
    log << "Testing simple image conversion..." << std::endl;
    mwArray mtrx = convertCvToMatlabMat(cvPositiveSamples[0]);
    log << "Testing cell array get first cell..." << std::endl;
    mwArray cell = mwPositiveSamples.Get(1, 1);
    log << "Testing cell array get last cell..." << std::endl;
    mwPositiveSamples.Get(1, NB_POSITIVE_IMAGES);
    log << "Testing cell array set first cell data..." << std::endl;
    cell.Set(mtrx);
    // Full conversion for Exemplar-SVM
    log << "Converting positive training samples..." << std::endl;
    for (int i = 0; i < NB_POSITIVE_IMAGES; i++)
        mwPositiveSamples.Get(1, i + 1).Set(convertCvToMatlabMat(cvPositiveSamples[i]));
    log << "Converting negative training samples..." << std::endl;
    for (int i = 0; i < NB_NEGATIVE_IMAGES; i++)
        mwNegativeSamples.Get(1, i + 1).Set(convertCvToMatlabMat(cvNegativeSamples[i]));
    log << "Converting probe testing samples..." << std::endl;
    for (int i = 0; i < NB_PROBE_IMAGES; i++)
        mwProbeSamples.Get(1, i + 1).Set(convertCvToMatlabMat(cvProbeSamples[i]));

    // ------------------------------------------------------------------------------------------------------------------------
    // Try Exemplar-SVM training and testing
    // ------------------------------------------------------------------------------------------------------------------------
    try
    {        
        log << "Running Exemplar-SVM training..." << std::endl;
        esvm_train_individual(1, models, mwPositiveSamples, mwNegativeSamples, target);
        log << "Running Exemplar-SVM testing..." << std::endl;
        esvm_test_individual(1, scores, models, mwProbeSamples);
        log << "Success" << std::endl;
        return 0;
    }
    catch (const mwException& e)
    {
        log << e.what() << std::endl;
        return -2;
    }
    catch (...)
    {
        log << "Unexpected error thrown" << std::endl;
        return -3;
    }
}
#endif 

int test_runBasicExemplarSvmClassification(void)
{
    // ------------------------------------------------------------------------------------------------------------------------
    // stream for console+file output
    // ------------------------------------------------------------------------------------------------------------------------        
    logstream log(LOGGER_FILE);
    log << "Starting basic Exemplar-SVM classification test..." << std::endl;

    // ------------------------------------------------------------------------------------------------------------------------
    // training ESVM with samples (XOR)
    // ------------------------------------------------------------------------------------------------------------------------ 
    log << "Training Exemplar-SVM with XOR samples..." << std::endl;
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
    log << "Testing Exemplar-SVM classification results..." << std::endl;
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
        log << "  Prediction result for {" << samples[s][0] << "," << samples[s][1] << "}: " << prediction << std::endl;
    }

    /*
    assert(esvm.predict(samples[0]) == -1);
    assert(esvm.predict(samples[1]) == +1);
    assert(esvm.predict(samples[2]) > 0.5);
    assert(esvm.predict(samples[3]) > 0.5);
    assert(esvm.predict(samples[4]) > 0.5);
    assert(esvm.predict(samples[4]) == -1);
    */

    return 0;
}

#if 0
int test_runSingleSamplePerPersonStillToVideo(cv::Size patchCounts)
{
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
    
    // ------------------------------------------------------------------------------------------------------------------------
    // window to display loaded images and stream for console+file output
    // ------------------------------------------------------------------------------------------------------------------------    
    cv::namedWindow(WINDOW_NAME);
    logstream log(LOGGER_FILE);
    log << "Starting single sample per person still-to-video test..." << std::endl;
    int nbPatches = patchCounts.width*patchCounts.height;
    if (nbPatches == 0) nbPatches = 1;

    // ------------------------------------------------------------------------------------------------------------------------
    // C++ parameters
    // ------------------------------------------------------------------------------------------------------------------------
    /* Multiple negative samples as counter-example for each individual to enroll (CANNOT BE A PROBE NOR POSITIVE SAMPLE) */
    const int NB_NEGATIVE_IMAGES = 177;
    std::vector<cv::Mat> cvNegativeSamples[NB_NEGATIVE_IMAGES];
    log << "Loading negative training samples used for all enrollments..." << std::endl;
    /* --- ID0028 --- */
    cvNegativeSamples[0]   = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_0/000190.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[1]   = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_0/000195.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[2]   = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_0/000200.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[3]   = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_0/000205.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[4]   = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_0/000225.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[5]   = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_0/000230.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[6]   = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_0/000235.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[7]   = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_0/000240.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[8]   = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_0/000245.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[9]   = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_0/000250.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    /* --- ID0010 --- */
    cvNegativeSamples[10]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_6/000246.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[11]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_6/000247.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[12]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_6/000250.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[13]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_6/000255.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[14]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_6/000260.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[15]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_6/000265.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[16]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_6/000270.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[17]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_6/000275.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[18]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_6/000280.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[19]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_6/000285.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[20]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_6/000286.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[21]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_6/000290.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[22]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_6/000295.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[23]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_6/000300.png",  WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[24]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_44/000635.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[25]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_44/000640.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[26]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_44/000641.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[27]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_44/000645.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[28]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_44/000650.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[29]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_44/000656.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);    
    /* --- ID0019 --- */
    cvNegativeSamples[30]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000280.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[31]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000285.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[32]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000290.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[33]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000295.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[34]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000300.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[35]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000305.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[36]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000310.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);    
    cvNegativeSamples[37]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000315.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[38]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000320.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[39]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000325.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[40]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000330.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[41]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000335.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[42]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000340.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[43]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000345.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[44]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000350.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[45]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_9/000355.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    /* --- ID0018 --- */
    cvNegativeSamples[46]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_16/000350.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[47]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_16/000355.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[48]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_16/000360.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[49]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_16/000361.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[50]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_16/000365.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[51]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_16/000370.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    /* --- ID0001 --- */
    cvNegativeSamples[52]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000435.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[53]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000440.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[54]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000445.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[55]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000450.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[56]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000455.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[57]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000460.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[58]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000465.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[59]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000470.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[60]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000475.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[61]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000480.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[62]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000485.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[63]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000490.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[64]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000495.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[65]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000500.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[66]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000505.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[67]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_23/000510.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    /* --- ID0030 --- */
    cvNegativeSamples[68]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_28/000465.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[69]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_28/000470.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[70]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_28/000475.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[71]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_28/000480.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[72]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_28/000481.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[73]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_28/000485.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[74]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_28/000490.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[75]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_28/000495.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    /* --- ID0002 --- */
    cvNegativeSamples[76]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_32/000480.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[77]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_32/000485.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[78]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_32/000490.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[79]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_32/000495.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[80]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_32/000497.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[81]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_32/000500.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[82]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_32/000505.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[83]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_32/000510.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[84]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_32/000515.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[85]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_32/000520.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[86]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_32/000525.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    /* --- ID0025 --- */
    cvNegativeSamples[87]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_33/000495.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[88]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_33/000500.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[89]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_33/000510.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[90]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_33/000515.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[91]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_33/000520.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[92]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_33/000525.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[93]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_33/000530.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[94]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_33/000535.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[95]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_33/000540.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    /* --- ID0024 --- */
    cvNegativeSamples[96]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000525.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[97]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000530.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[98]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000535.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[99]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000540.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[100] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000545.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[101] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000550.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[102] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000555.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[103] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000560.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[104] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000565.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[105] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000570.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[106] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000575.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[107] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000580.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[108] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000585.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[109] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000590.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[110] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000595.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[111] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000600.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[112] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000605.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[113] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000610.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[114] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000615.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[115] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000620.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[116] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_34/000625.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);       
    /* --- ID0007 --- */
    cvNegativeSamples[117] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000606.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[118] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000610.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[119] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000615.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[120] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000620.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[121] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000625.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[122] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000630.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[123] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000635.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[124] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000640.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[125] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000645.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[126] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000646.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[127] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000650.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[128] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000651.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[129] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000655.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[130] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000656.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[131] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000660.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[132] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000665.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[133] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000670.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[134] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000675.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[135] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000680.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[136] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000685.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[137] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000690.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[138] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000700.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[139] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000705.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[140] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000710.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[141] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_40/000715.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    /* --- ID0017 --- */
    cvNegativeSamples[142] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_41/000611.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[143] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_41/000615.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[144] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_41/000635.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[145] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_41/000640.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[146] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_41/000645.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[147] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_41/000650.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[148] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_41/000655.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts); 
    cvNegativeSamples[149] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_41/000660.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[150] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_41/000665.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[151] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_41/000670.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[152] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_41/000675.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[153] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_41/000680.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[154] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_41/000685.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    /* --- ID0027 --- */
    cvNegativeSamples[155] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_42/000611.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[156] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_42/000615.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[157] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_46/000641.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[158] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_46/000645.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[159] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_46/000650.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    /* --- ID0006 --- */
    cvNegativeSamples[160] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000650.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[161] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000655.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[162] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000660.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[163] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000665.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[164] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000666.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[165] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000670.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[166] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000675.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[167] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000676.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[168] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000680.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[169] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000681.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[170] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000685.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[171] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000686.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[172] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000690.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[173] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000695.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[174] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000700.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[175] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000705.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvNegativeSamples[176] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_45/000710.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);

    /* Single positive training samples (one per enrollment) */
    const int NB_ENROLLMENT = 5;
    std::string targetName[NB_ENROLLMENT];
    std::vector<cv::Mat> cvPositiveSamples[NB_ENROLLMENT];
    // Positive targets (same as Saman paper)
    targetName[0] = "ID0011";
    targetName[1] = "ID0012";
    targetName[2] = "ID0013";
    targetName[3] = "ID0016";
    targetName[4] = "ID0020";
    log << "Loading single positive training samples..." << std::endl;
    // Deduct full ROI size using the patch size and quantity since positive sample is high quality (different dimension)
    cv::Size imSize = cvNegativeSamples[0][0].size();
    imSize.width *= patchCounts.width;
    imSize.height *= patchCounts.height;
    // Get still reference images (color high quality neutral faces) 
    // filename format: "roi<ID#>.jpg"
    for (int i = 0; i < NB_ENROLLMENT; i++)    
        cvPositiveSamples[i] = imPreprocess(refStillImagesPath + "roi" + targetName[i] + ".JPG", imSize, patchCounts, WINDOW_NAME, cv::IMREAD_COLOR);
    
    /* Testing probe samples */
    const int NB_PROBE_IMAGES = 96;
    std::string probeGroundThruth[NB_PROBE_IMAGES];
    std::vector<cv::Mat> cvProbeSamples[NB_PROBE_IMAGES];
    log << "Loading testing probe samples..." << std::endl;
    /* --- ID0013 --- */
    cvProbeSamples[0]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_7/000255.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[1]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_7/000260.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[2]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_7/000265.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[3]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_7/000267.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[4]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_7/000270.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[5]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_7/000272.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[6]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_7/000275.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    for (int i = 0; i <= 6; i++) probeGroundThruth[i] = "ID0013";
    /* --- ID0012 --- */
    cvProbeSamples[7]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_13/000320.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[8]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_13/000325.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[9]  = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_13/000330.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[10] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_13/000335.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[11] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_13/000340.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[12] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_13/000345.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[13] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_13/000350.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[14] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_13/000355.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[15] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_13/000360.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    for (int i = 7; i <= 15; i++) probeGroundThruth[i] = "ID0012";
    /* --- ID0011 --- */
    cvProbeSamples[16] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_15/000350.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[17] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_15/000355.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[18] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_15/000360.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[19] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_15/000365.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[20] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_15/000370.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[21] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_15/000375.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[22] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_15/000377.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[23] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_15/000380.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[24] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_15/000385.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[25] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_15/000390.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    for (int i = 16; i <= 25; i++) probeGroundThruth[i] = "ID0011";
    /* --- ID0029 --- */
    cvProbeSamples[26] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_18/000365.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[27] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_18/000370.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[28] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_18/000375.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[29] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_18/000380.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[30] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_18/000381.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[31] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_18/000385.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[32] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_18/000390.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[33] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_18/000395.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[34] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_18/000400.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[35] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_18/000401.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[36] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_18/000425.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[37] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_18/000430.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    for (int i = 26; i <= 37; i++) probeGroundThruth[i] = "ID0029";
    /* --- ID0016 --- */
    cvProbeSamples[38] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_19/000400.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[39] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_19/000405.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[40] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_19/000406.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[41] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_19/000410.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[42] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_19/000415.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[43] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_19/000420.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[44] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_19/000425.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[45] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_19/000430.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[46] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_19/000435.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[47] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_19/000440.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[48] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_19/000445.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[49] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_19/000450.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[50] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_19/000455.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[51] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_19/000460.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[52] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_19/000465.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    for (int i = 38; i <= 52; i++) probeGroundThruth[i] = "ID0016";
    /* --- ID0009 --- */
    cvProbeSamples[53] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_20/000410.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[54] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_20/000415.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[55] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_20/000420.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[56] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_20/000425.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[57] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_25/000441.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[58] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_25/000445.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[59] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_25/000450.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    for (int i = 53; i <= 59; i++) probeGroundThruth[i] = "ID0009";
    /* --- ID0004 --- */
    cvProbeSamples[60] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_26/000447.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[61] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_26/000450.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[62] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_26/000455.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[63] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_26/000460.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[64] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_26/000465.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[65] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_26/000470.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[66] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_26/000475.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[67] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_26/000480.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[68] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_26/000485.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    for (int i = 60; i <= 68; i++) probeGroundThruth[i] = "ID0004";
    /* --- ID0020 --- */
    cvProbeSamples[69] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_36/000540.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[70] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_36/000545.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[71] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_36/000550.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[72] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_36/000552.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[73] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_36/000555.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[74] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_36/000556.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[75] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_36/000560.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[76] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_36/000562.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[77] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_36/000565.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[78] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_36/000566.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[79] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_36/000570.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    for (int i = 69; i <= 79; i++) probeGroundThruth[i] = "ID0020";
    /* --- ID0023 --- */
    cvProbeSamples[80] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_37/000541.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[81] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_37/000545.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[82] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_37/000550.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[83] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_37/000555.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[84] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_37/000561.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[85] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_37/000565.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[86] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_37/000570.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    for (int i = 80; i <= 86; i++) probeGroundThruth[i] = "ID0023";
    /* --- ID0026 --- */
    cvProbeSamples[87] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_39/000566.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[88] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_39/000570.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[89] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_39/000575.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[90] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_39/000580.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[91] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_39/000583.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[92] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_39/000585.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[93] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_39/000590.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[94] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_39/000595.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    cvProbeSamples[95] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_39/000600.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    for (int i = 87; i <= 95; i++) probeGroundThruth[i] = "ID0026";

    // Destroy viewing window not required anymore
    cv::destroyWindow(WINDOW_NAME);

    // ------------------------------------------------------------------------------------------------------------------------
    // Transform into MATLAB arrays
    // NB: vector and OpenCV Mat are zero-based, but MATLAB mwArray are one-based
    // ------------------------------------------------------------------------------------------------------------------------    
    std::vector<mwArray> models(nbPatches);
    std::vector<mwArray> mwScores(nbPatches);    
    std::vector<mwArray> mwNegativeSamples(nbPatches);
    std::vector<mwArray> mwProbeSamples(nbPatches);
    // Duplication required for unique positive per individual
    int NB_POSITIVE_DUPLICATION = 1; 
    std::vector< std::vector<mwArray> > mwPositiveSamples(NB_ENROLLMENT);
    // Conversion for Exemplar-SVM
    for (int p = 0; p < nbPatches; p++)
    {
        log << "Converting patches at index " << p << "..." << std::endl;
        mwNegativeSamples[p] = mwArray(NB_NEGATIVE_IMAGES, 1, mxCELL_CLASS);
        mwProbeSamples[p]    = mwArray(NB_PROBE_IMAGES, 1, mxCELL_CLASS);

        log << "Converting positive training samples..." << std::endl;        
        for (int i = 0; i < NB_ENROLLMENT; i++)
        {        
            // Initialize vertor only on first patch for future calls
            if (p == 0) 
                mwPositiveSamples[i] = std::vector<mwArray>(nbPatches);

            // Duplicate unique positive to generate a pool samples           
            mwPositiveSamples[i][p] = mwArray(NB_POSITIVE_DUPLICATION, 1, mxCELL_CLASS);
            mwArray dupPositive = convertCvToMatlabMat(cvPositiveSamples[i][p]);
            for (int j = 0; j < NB_POSITIVE_DUPLICATION; j++)
                mwPositiveSamples[i][p].Get(1, j + 1).Set(dupPositive);
        }
        
        log << "Converting negative training samples..." << std::endl;
        for (int i = 0; i < NB_NEGATIVE_IMAGES; i++)
            mwNegativeSamples[p].Get(1, i + 1).Set(convertCvToMatlabMat(cvNegativeSamples[i][p]));
        
        log << "Converting probe testing samples..." << std::endl;
        for (int i = 0; i < NB_PROBE_IMAGES; i++)
            mwProbeSamples[p].Get(1, i + 1).Set(convertCvToMatlabMat(cvProbeSamples[i][p]));
    }



    //################################################################################ DEBUG
    /*cv::Mat img = imReadAndDisplay(refStillImagesPath + "roi" + targetName[0] + ".JPG", WINDOW_NAME, cv::IMREAD_COLOR);
    cv::cvtColor(img, img, CV_BGR2GRAY);
    cv::resize(img, img, imSize, 0, 0, cv::INTER_CUBIC);    
    cv::imwrite(refStillImagesPath + "roi" + targetName[0] + "_resized.JPG", img);    
    for (int p = 0; p < nbPatches; p++)
    {
        std::string name = refStillImagesPath + "roi" + targetName[0] + "_patch" + std::to_string(p) + ".jpg";        
        cv::imwrite(name, cvPositiveSamples[0][p]);
    }*/
    //################################################################################ DEBUG

    // ------------------------------------------------------------------------------------------------------------------------
    // Try Exemplar-SVM training and testing with single sample per person (SSPP) in still-to-video
    // ------------------------------------------------------------------------------------------------------------------------
    try
    {        
        //################################################################################ DEBUG
        /*
        log << "DEBUG: " << std::endl;
        log << models.size() << std::endl;
        log << mwPositiveSamples.size() << std::endl;
        log << mwNegativeSamples.size() << std::endl;
        log << "Dims Pos sample: " << mwPositiveSamples[0].Get(1, 1).GetDimensions() << std::endl;
        log << "Dims nb positive: " << mwPositiveSamples[0].GetDimensions().ToString() << std::endl;
        log << "Data:" << std::endl;        
        log << mwPositiveSamples[0].ToString() << std::endl;
        for (int i = 0; i < 5; i++)
        {
            log << "Data detail " << i << ":" << std::endl;
            log << mwPositiveSamples[0].Get(1,i+1).ToString() << std::endl;
        }

        cv::Size is = cvNegativeSamples[0][0].size();
        cv::Mat im = imReadAndDisplay(refStillImagesPath + "roi" + targetName[0] + ".JPG", WINDOW_NAME, cv::IMREAD_COLOR);
        log << "Dims resize neg image: " << is << std::endl;
        log << "Dims original image: " << im.size() << std::endl;
        cv::cvtColor(im, im, CV_BGR2GRAY);
        cv::resize(im, im, is, 0, 0, cv::INTER_CUBIC);
        log << "Dims resized pos image: " << im.size() << std::endl;
        auto vIm = imSplitPatches(im, patchCounts);
        log << "Pos nb patches: " << vIm.size() << std::endl;
        log << "Dims patch 0: " << vIm[0].size() << std::endl;


        log << mwPositiveSamples[0].ClassID() << std::endl;     // cell: 1, double: 6, uint8: 9
        log << mwPositiveSamples[0].Get(1, 1).ClassID() << std::endl;
        const int size = 32 * 32;
        UINT8 data[size];
        mwPositiveSamples[0].Get(1, 1).GetData(data, size);
        std::string outData = "";
        for (int i = 0; i < size; i++)
        {
            outData += data[i];
            outData += +" ";
        }
        log << outData << std::endl;
        
        log << "DEBUG INFO" << std::endl;
        log << mwPositiveSamples.size() << std::endl;
        log << mwPositiveSamples[0].size() << std::endl;
        log << mwPositiveSamples[0][0].GetDimensions() << std::endl;
        */
        //################################################################################ DEBUG

        for (int i = 0; i < NB_ENROLLMENT; i++)
        {            
            log << "Starting for individual " << i << ": " + targetName[i] << std::endl;
            double scoreFusion[NB_PROBE_IMAGES] = { 0 };
            for (int p = 0; p < nbPatches; p++)
            {                
                log << "Running Exemplar-SVM training..." << std::endl;
                esvm_train_individual(1, models[p], mwPositiveSamples[i][p], mwNegativeSamples[p], mwArray(targetName[i].c_str()));
                log << "Running Exemplar-SVM testing..." << std::endl;
                esvm_test_individual(1, mwScores[p], models[p], mwProbeSamples[p]);
                double scores[NB_PROBE_IMAGES];
                mwScores[p].GetData(scores, NB_PROBE_IMAGES);
                for (int j = 0; j < NB_PROBE_IMAGES; j++)
                {
                    // score accumulation from patches with normalization
                    double normPatchScore = normalizeClassScoreToSimilarity(scores[j]);
                    scoreFusion[j] += normPatchScore;
                    std::string probeGT = (probeGroundThruth[j] == targetName[i] ? "positive" : "negative");
                    log << "Score for patch " << p << " of probe " << j << " (" << probeGT << "): " << normPatchScore << std::endl;
                }
            }
            for (int j = 0; j < NB_PROBE_IMAGES; j++)
            {
                // average of score accumulation for fusion
                std::string probeGT = (probeGroundThruth[j] == targetName[i] ? "positive" : "negative");
                log << "Score fusion of probe " << j << " (" << probeGT << "): " << scoreFusion[j] / nbPatches << std::endl;               
            }
            log << "Completed for individual " << i << ": " + targetName[i] << std::endl;
        }
        log << "Success" << std::endl;
        return 0;
    }
    catch (const mwException& e)
    {
        log << e.what() << std::endl;
        return -2;
    }
    catch (...)
    {
        log << "Unexpected error thrown" << std::endl;
        return -3;
    }
}
#endif

int test_runSingleSamplePerPersonStillToVideo_FullChokePoint(cv::Size imageSize, cv::Size patchCounts, bool useSyntheticPositives)
{
    /**************************************************************************************************************************
    TEST DEFINITION        
    */
    /* Training Targets:        single high quality still image for enrollment (same as Saman paper) */
    std::vector<std::string> positivesID = { "0011", "0012", "0013", "0016", "0020" };
    /* Training Non-Targets:    as many video negatives as possible */
    std::vector<std::string> negativesID = { "0001", "0002", "0006", "0007", "0010",
                                             "0017", "0018", "0019", "0024", "0025",
                                             "0027", "0028", "0030" };
    /* Testing Probes:          some video positives and negatives */
    std::vector<std::string> probesID = { "0004", "0009", "0011", "0012", "0013",
                                          "0016", "0020", "0023", "0026", "0029" };
    /*************************************************************************************************************************/   

    // Display and output
    cv::namedWindow(WINDOW_NAME);
    logstream log(LOGGER_FILE);
    int nbPatches = patchCounts.width * patchCounts.height;
    if (nbPatches == 0) nbPatches = 1;    
    log << "Starting single sample per person still-to-video full ChokePoint test..." << std::endl
        << "   useSyntheticPositives: " << useSyntheticPositives << std::endl
        << "   imageSize:             " << imageSize << std::endl
        << "   patchCounts:           " << patchCounts << std::endl;
    
    // Samples container with [target][roi/representation][patch] indexes
    int nbPositives = positivesID.size();
    int nbRepresentations = 1;
    std::vector< std::vector< std::vector<cv::Mat> > > cvPositiveSamples(nbPositives);  // [positive][representation][patch](Mat[x,y])
    std::vector< std::string > probeGroundTruthID;                                      // [positive]
    
    // Samples containers with [roi][patch] indexes
    std::vector< std::vector<cv::Mat> > cvNegativeSamples;                              // [negative][patch](Mat[x,y])
    std::vector< std::vector<cv::Mat> > cvProbeSamples;                                 // [probe][patch](Mat[x,y])    

    // Add samples to containers
    log << "Loading positives image for all test sequences..." << std::endl;    
    for (int i = 0; i < nbPositives; i++)
    {        
        // Add additional positive representations as requested
        if (useSyntheticPositives)
        {
            // Get original positive image with preprocessing but without patches splitting
            cv::Mat img = imPreprocess(refStillImagesPath + "roiID" + positivesID[i] + ".jpg",
                                       imageSize, cv::Size(1,1), WINDOW_NAME, cv::IMREAD_COLOR)[0];
            // Get synthetic representations from original and apply patches splitting each one
            std::vector<cv::Mat> representations = imSyntheticGeneration(img);
            nbRepresentations = representations.size();
            cvPositiveSamples[i] = std::vector< std::vector< cv::Mat> >(nbRepresentations);
            /// ############################################# #pragma omp parallel for
            for (int r = 0; r < nbRepresentations; r++)
                cvPositiveSamples[i][r] = imSplitPatches(representations[r], patchCounts);
        }
        // Only original representation otherwise (no synthetic images)
        else
        {
            cvPositiveSamples[i] = std::vector< std::vector< cv::Mat> >(1);
            cvPositiveSamples[i][0] = imPreprocess(refStillImagesPath + "roiID" + positivesID[i] + ".jpg",
                                                   imageSize, patchCounts, WINDOW_NAME, cv::IMREAD_COLOR);
        }
    }


    //################################################################################ DEBUG DISPLAY POSITIVES (+SYNTH)
    /*
    log << "SHOWING DEBUG POSITIVE SAMPLES" << std::endl;
    for (int i = 0; i < nbPositives; i++)
    {
        for (int j = 0; j < cvPositiveSamples[i].size(); j++)
        {
            for (int k = 0; k < cvPositiveSamples[i][j].size(); k++)
            {
                cv::imshow(WINDOW_NAME, cvPositiveSamples[i][j][k]);
                cv::waitKey(500);
            }
        }
    }
    log << "DONE SHOWING DEBUG POSITIVE SAMPLES" << std::endl;
    */
    //################################################################################ DEBUG
    
    // Destroy viewing window not required anymore
    /// ################################################################################ DEBUG
    // cv::destroyWindow(WINDOW_NAME);
    /// ################################################################################ DEBUG
            
    // Containers for feature vectors extracted from samples        
    std::vector< std::vector <FeatureVector > > fvNegativeSamples(nbPatches);                       // [patch][negative][feature]
    std::vector< std::vector <FeatureVector > > fvProbeSamples(nbPatches);                          // [patch][probe][feature]
    std::vector< std::vector < std::vector< FeatureVector > > > fvPositiveSamples(nbPositives);     // [positive][patch][representation][feature]
    std::vector< std::vector< ESVM > > esvmModels(nbPositives);                                     // [positive][patch]

    FeatureExtractorHOG hog;
    cv::Size patchSize = cv::Size(imageSize.width / patchCounts.width, imageSize.height / patchCounts.height);
    cv::Size hogBlock = cv::Size(patchSize.width / 2, patchSize.height / 2);
    cv::Size hogCell = cv::Size(hogBlock.width / 4, hogBlock.height / 4);
    int nbBins = 8;
    hog.initialize(patchSize, hogBlock, hogBlock, hogCell, nbBins);
    log << "HOG feature extraction initialized..." << std::endl
        << "   imageSize: " << imageSize << std::endl
        << "   patchSize: " << patchSize << std::endl
        << "   hogBlock:  " << hogBlock << std::endl
        << "   hogCell:   " << hogCell << std::endl 
        << "   nbBins:    " << nbBins << std::endl;
    
    FeatureExtractorLBP lbp;
    int points = 8;
    int radius = 8;
    MappingType map = LBP_MAPPING_U2;
    lbp.initialize(points, radius, map);
    log << "LBP feature extraction initialized..." << std::endl
        << "   imageSize: " << imageSize << std::endl
        << "   points:    " << points << std::endl
        << "   radius:    " << radius << std::endl
        << "   mapping:   " << lbp::MappingTypeStr[map] << std::endl;

    // Convert unique positive samples (or with synthetic representations)
    log << "Feature extraction of positive images for all test sequences..." << std::endl;
    log << "   nbPositives:       " << nbPositives << std::endl;
    log << "   nbPatches:         " << nbPatches << std::endl;
    log << "   nbRepresentations: " << nbRepresentations << std::endl;
    /// ################################################## #pragma omp parallel for
    for (int i = 0; i < nbPositives; i++)
    {
        // Initialize vector for all positive representations per patch
        fvPositiveSamples[i] = std::vector< std::vector< FeatureVector > >(nbPatches);
        /// ################################################## #pragma omp parallel for
        for (int p = 0; p < nbPatches; p++)
        {            
            fvPositiveSamples[i][p] = std::vector< FeatureVector >(nbRepresentations);
            /// ################################################## #pragma omp parallel for
            for (int r = 0; r < nbRepresentations; r++)
            {
                // switch to (i,p,r) order for patch-based training
                fvPositiveSamples[i][p][r] = lbp.compute(cvPositiveSamples[i][r][p]);
                /// fvPositiveSamples[i][p][r] = hog.compute(cvPositiveSamples[i][r][p]);

                /// ################################################## DEBUG CHECK INPUT IMAGE / VECTORS
                /*
                cv::imshow(WINDOW_NAME, cvPositiveSamples[i][r][p]);
                log << "v (i=" << i << ",r=" << r << ",p=" << p << "): ";
                FeatureVector s = fvPositiveSamples[i][p][r];
                std::string ss = "{";
                for (int f = 0; f < s.size(); f++)
                {
                    if (f != 0) ss += ",";
                    ss += std::to_string(s[f]);
                }
                log << ss << "} | " << 1 << std::endl;
                while (cv::waitKey(10) != 'k');
                */
                /// ################################################## DEBUG 
            }
        }
    }
    log << "Features dimension: " << fvPositiveSamples[0][0][0].size() << std::endl;

    // Tests divided per sequence information according to selected mode
    std::vector<PORTAL_TYPE> types = { ENTER, LEAVE };
    bfs::directory_iterator endDir;
    for (int sn = 1; sn <= SESSION_NUMBER; sn++)
    {
        #if CHOKEPOINT_FULL_TEST_SEQUENCES_MODE == 0
        cv::namedWindow(WINDOW_NAME);
        #endif/*CHOKEPOINT_FULL_TEST_SEQUENCES_MODE*/
        
        for (int pn = 1; pn <= PORTAL_NUMBER; pn++) {
        for (auto it = types.begin(); it != types.end(); ++it) {
        for (int cn = 1; cn <= CAMERA_NUMBER; cn++)
        {     
            #if CHOKEPOINT_FULL_TEST_SEQUENCES_MODE == 1
            cv::namedWindow(WINDOW_NAME);
            // Reset vectors for next test sequences                    
            cvNegativeSamples.clear();
            cvProbeSamples.clear();
            probeGroundTruthID.clear();
            #endif/*CHOKEPOINT_FULL_TEST_SEQUENCES_MODE*/

            std::string seq = buildChokePointSequenceString(pn, *it, sn, cn);
            log << "Loading negative and probe images for sequence " << seq << "..." << std::endl;                    

            // Add ROI to corresponding sample vectors according to individual IDs            
            for (int id = 1; id <= INDIVIDUAL_NUMBER; id++)
            {            
                std::string dirPath = roiChokePointPath + buildChokePointSequenceString(pn, *it, sn, cn, id) + "/";
                if (bfs::is_directory(dirPath))
                {
                    for (bfs::directory_iterator itDir(dirPath); itDir != endDir; ++itDir)
                    {
                        if (bfs::is_regular_file(*itDir) && itDir->path().extension() == ".pgm")
                        {
                            std::string strID = buildChokePointIndividualID(id);
                            if (contains(negativesID, strID))
                            {
                                cvNegativeSamples.push_back(imPreprocess(itDir->path().string(), imageSize, 
                                                            patchCounts, WINDOW_NAME, cv::IMREAD_GRAYSCALE));
                            }
                            else if (contains(probesID, strID))
                            {
                                cvProbeSamples.push_back(imPreprocess(itDir->path().string(), imageSize, 
                                                         patchCounts, WINDOW_NAME, cv::IMREAD_GRAYSCALE));
                                probeGroundTruthID.push_back(strID);
                            }


                            //################################################################################ DEBUG
                            /*
                            auto dbg = imPreprocess(itDir->path().string(), imageSize, cv::Size(1, 1), WINDOW_NAME, cv::IMREAD_GRAYSCALE);
                            cv::imshow(WINDOW_NAME, dbg[0]);
                            cv::waitKey(500);
                            */
                            //################################################################################ DEBUG      
                        }                  
                    }
                }                        
            }

        // Add end of loops if sequences must be combined per session
        #if CHOKEPOINT_FULL_TEST_SEQUENCES_MODE == 0
        } } }
        #endif/*CHOKEPOINT_FULL_TEST_SEQUENCES_MODE*/
        
            // Destroy viewing window not required while training/testing is in progress
            cv::destroyWindow(WINDOW_NAME);
                            
            // Feature extraction of negatives and probes
            int nbProbes = cvProbeSamples.size();
            int nbNegatives = cvNegativeSamples.size();
            log << "Feature extraction of negative and probe samples (total negatives: " << nbNegatives 
                << ", total probes: " << nbProbes << ")..." << std::endl;
            /// ############################################# #pragma omp parallel for
            for (int p = 0; p < nbPatches; p++)
            {                  
                fvNegativeSamples[p] = std::vector< FeatureVector >(nbNegatives);
                fvProbeSamples[p] = std::vector< FeatureVector >(nbProbes);

                // switch to (p,i) order for patch-based training
                /// ############################################# #pragma omp parallel for
                for (int i = 0; i < nbNegatives; i++)
                {
                    fvNegativeSamples[p][i] = lbp.compute(cvNegativeSamples[i][p]);
                    /// fvNegativeSamples[p][i] = hog.compute(cvNegativeSamples[i][p]);
                }
                /// ############################################# #pragma omp parallel for
                for (int i = 0; i < nbProbes; i++)
                {
                    fvProbeSamples[p][i] = lbp.compute(cvProbeSamples[i][p]);
                    /// fvProbeSamples[p][i] = hog.compute(cvProbeSamples[i][p]);
                }
            }

            // Enroll positive individuals with Exemplar-SVMs  
            #if CHOKEPOINT_FULL_TEST_SEQUENCES_MODE == 0
            log << "Starting enrollment for sequence: S" << sn << "..." << std::endl;
            #elif CHOKEPOINT_FULL_TEST_SEQUENCES_MODE == 1
            log << "Starting enrollment for sequence: " << seq << "..." << std::endl;
            #endif/*CHOKEPOINT_FULL_TEST_SEQUENCES_MODE*/



            // ######################################################################################### DEBUG
            /*
            log << "POS: " << nbPositives << std::endl;
            log << "NEG: " << nbNegatives << std::endl;
            log << "PRB: " << nbProbes << std::endl;
            log << "PTC: " << nbPatches << std::endl;
            log << "wP1: " << mwPositiveSamples.size() << std::endl;
            log << "wP2: " << mwPositiveSamples[0].size() << std::endl;
            log << "wP3: " << mwPositiveSamples[0][0].GetDimensions() << std::endl;
            log << "wP4: " << mwPositiveSamples[0][0].Get(1,1).GetDimensions() << std::endl;
            log << "wN1: " << mwNegativeSamples.size() << std::endl;
            log << "wN2: " << mwNegativeSamples[0].GetDimensions() << std::endl;
            log << "wN3: " << mwNegativeSamples[0].Get(1,1).GetDimensions() << std::endl;
            log << "wT1: " << mwProbeSamples.size() << std::endl;
            log << "wT2: " << mwProbeSamples[0].GetDimensions() << std::endl;
            log << "wT3: " << mwProbeSamples[0].Get(1,1).GetDimensions() << std::endl;
            
            for (int p = 0; p < nbPatches; p++)
            {
                log << "Positive i=0 p=" << p << ":" << std::endl << mwPositiveSamples[0][p].Get(1, 1) << std::endl;
                log << "Negative i=0 p=" << p << ":" << std::endl << mwNegativeSamples[p].Get(1, 1) << std::endl;
            }*/
            //############################################################################################## DEBUG

            // Feature normalization
            log << "Running feature normalization..." << std::endl;
            FeatureVector minFeatures, maxFeatures;
            std::vector< FeatureVector > allFeatureVectors;
            /// ############################################# #pragma omp parallel for
            for (int p = 0; p < nbPatches; p++)
            {
                for (int i = 0; i < nbPositives; i++)
                    for (int r = 0; r < nbRepresentations; r++)
                        allFeatureVectors.push_back(fvPositiveSamples[i][p][r]);
                for (int i = 0; i < nbNegatives; i++)
                    allFeatureVectors.push_back(fvNegativeSamples[p][i]);
                for (int i = 0; i < nbProbes; i++)
                    allFeatureVectors.push_back(fvProbeSamples[p][i]);
            }
            findMinMaxFeatures(allFeatureVectors, &minFeatures, &maxFeatures);
            for (int p = 0; p < nbPatches; p++)
            {
                for (int i = 0; i < nbPositives; i++)                  
                    for (int r = 0; r < nbRepresentations; r++)
                    {
                        fvPositiveSamples[i][p][r] = normalizeFeatures(fvPositiveSamples[i][p][r], minFeatures, maxFeatures);
                        log << "POS: " << featuresToString(fvPositiveSamples[i][p][r]) << std::endl;    /// ################################ DEBUG
                    }
                for (int i = 0; i < nbNegatives; i++)
                {
                    fvNegativeSamples[p][i] = normalizeFeatures(fvNegativeSamples[p][i], minFeatures, maxFeatures);
                    log << "NEG: " << featuresToString(fvNegativeSamples[p][i]) << std::endl;           /// ################################ DEBUG
                }
                for (int i = 0; i < nbProbes; i++)
                {
                    fvProbeSamples[p][i] = normalizeFeatures(fvProbeSamples[p][i], minFeatures, maxFeatures);
                    log << "PRB: " << featuresToString(fvProbeSamples[p][i]) << std::endl;              /// ################################ DEBUG
                }
            }

            // Classifiers training and testing
            for (int i = 0; i < nbPositives; i++)
            {                        
                log << "Starting for individual " << i << ": " + positivesID[i] << std::endl;
                std::vector<double> fusionScores(nbProbes, 0.0);
                for (int p = 0; p < nbPatches; p++)
                {                    
                    esvmModels[i] = std::vector<ESVM>(nbPatches);
                    try
                    {
                        log << "Running Exemplar-SVM training..." << std::endl;
                        esvmModels[i][p] = ESVM(fvPositiveSamples[i][p], fvNegativeSamples[p], positivesID[i]);                        
                    }
                    catch (const std::exception& e)
                    {
                        log << e.what() << std::endl;
                        return -2;
                    }
                    catch (...)
                    {
                        log << "Unexpected error thrown" << std::endl;
                        return -3;
                    }

                    log << "Running Exemplar-SVM testing..." << std::endl;
                    std::vector<double> patchScores(nbProbes, 0.0);
                    /// ############################################# #pragma omp parallel for
                    for (int j = 0; j < nbProbes; j++)
                    {
                        patchScores[j] = esvmModels[i][p].predict(fvProbeSamples[p][j]);
                    }
                    
                    for (int j = 0; j < nbProbes; j++)
                    {
                        // score accumulation from patches with normalization
                        double normPatchScore = patchScores[j]; /// normalizeClassScoreToSimilarity(patchScores[j]);
                        fusionScores[j] += normPatchScore;
                        std::string probeGT = (probeGroundTruthID[j] == positivesID[i] ? "positive" : "negative");
                        log << "Score for patch " << p << " of probe " << j << " (ID" << probeGroundTruthID[j] << ", " 
                            << probeGT << "): " << normPatchScore << std::endl;
                    }
                }
                for (int j = 0; j < nbProbes; j++)
                {
                    // average of score accumulation for fusion
                    std::string probeGT = (probeGroundTruthID[j] == positivesID[i] ? "positive" : "negative");
                    fusionScores[j] = fusionScores[j] / nbPatches;
                    log << "Score fusion of probe " << j << " (ID" << probeGroundTruthID[j] << ", "
                        << probeGT << "): " << fusionScores[j] << std::endl;
                }
                log << "Completed for individual " << i << ": " + positivesID[i] << std::endl;
            }

            #if CHOKEPOINT_FULL_TEST_SEQUENCES_MODE == 0
            log << "Completed for sequence: S" << sn << "..." << std::endl;
            #elif CHOKEPOINT_FULL_TEST_SEQUENCES_MODE == 1
            log << "Completed for sequence: " << seq << std::endl;
            #endif/*CHOKEPOINT_FULL_TEST_SEQUENCES_MODE*/            
        
        // Add end of loops if sequences must be separated per scene
        #if CHOKEPOINT_FULL_TEST_SEQUENCES_MODE == 1
        } } }
        #endif/*CHOKEPOINT_FULL_TEST_SEQUENCES_MODE*/
            
    } // End session loop 

    log << "Test complete" << std::endl;
    return 0;
}