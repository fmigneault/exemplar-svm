#include "esvmTypesDef.h"
#include "helperFunctions.h"
#include "esvmTests.h"

/*
int main(void)
{
    cv::Size imgSize = cv::Size(96, 96);
    cv::Size patchSize = cv::Size(imgSize.width / 3, imgSize.height / 3);
    cv::Size patchCell = cv::Size(imgSize.width / 2, imgSize.height / 2);

    cv::Mat img;
    cv::resize(img, img, imgSize, 0, 0, CV_INTER_CUBIC);

    FeatureExtractorHOG hog;
    hog.initialize(img.size(), patchSize, patchSize, patchCell, 8);
    
        
    ESVM esvm(hog.compute(img), )


    return 0;
}
*/


int main(int argc, char* argv[])
{
    logstream log(LOGGER_FILE);
    log << "=================================================================" << std::endl
        << "Starting new Exemplar-SVM test execution " << currentTimeStamp() << std::endl;

    //################################################################################ DEBUG
    /*
    int err = test_imagePatchExtraction();
    if (err)
    {
    log << "Test 'imagePatchExtraction' failed." << std::endl;
    return err;
    }

    err = test_runBasicExemplarSvmFunctionalities();
    if (err)
    {
    log << "Test 'runBasicExemplarSvmFunctionalities' failed." << std::endl;
    return err;
    }

    // Number of patches to use in each direction, must fit within the ROIs (ex: 4x4 patches & ROI 128x128 -> 16 patches of 32x32)
    // Specifying Size(0,0) or Size(1,1) will result in not applying patches (use whole ROI)
    cv::Size patchCounts = cv::Size(4, 4);
    err = test_runSingleSamplePerPersonStillToVideo(patchCounts);
    if (err)
    {
    log << "Test 'runSingleSamplePerPersonStillToVideo' failed." << std::endl;
    return err;
    }
    */
    //################################################################################ DEBUG


    //################################################################################ NO PATCH
    /* NO PATCHES TEST */
    cv::Size patchCounts = cv::Size(1, 1);
    cv::Size imageSize = cv::Size(96, 96);
    bool useSyntheticPositives = true;
    int err = test_runSingleSamplePerPersonStillToVideo_FullChokePoint(imageSize, patchCounts, useSyntheticPositives);
    if (err)
    {
        log << "Test 'runSingleSamplePerPersonStillToVideo_FullChokePoint' failed." << std::endl;
        return err;
    }
    //################################################################################ NO PATCH

    /*------------------------------------------------------------------------------------------------
    NOTE:    IMPORTANT TO AVOID EMPTY CELL ARRAY ERROR GENERATED BY ZERO LENGHT EXTRACTED HOG FEATURES

    Since the HOG 'sbin' value is specified as 8 in the MATLAB code (standard value), the image
    size must absolutely be greater or equal to 20x20. The number of image blocks is calculated
    within 'pedro_features.mex' using '(int)round(imSize/sbin)-2'. For example, a 16x16 image
    results in 0x0 blocks within the image to extract features.

    If patches are employed, each one must then be greater or equal to 20x20 because each one
    is processed as an independant image for HOG feature extraction.
    ------------------------------------------------------------------------------------------------*/
    /*
    cv::Size patchCountsCP = cv::Size(4, 4);
    cv::Size imageSizeCP = cv::Size(96, 96);
    bool useSyntheticPositives = true;
    int err = test_runSingleSamplePerPersonStillToVideo_FullChokePoint(imageSizeCP, patchCountsCP, useSyntheticPositives);
    if (err)
    {
    log << "Test 'runSingleSamplePerPersonStillToVideo_FullChokePoint' failed." << std::endl;
    return err;
    }
    */

    return 0;
}
