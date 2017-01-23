#include "logger.h"
#include "esvmOptions.h"
#include "esvmTests.h"

int main(int argc, char* argv[])
{
    logstream logger(LOGGER_FILE);
    logger << "=================================================================" << std::endl
           << "Starting new Exemplar-SVM test execution " << currentTimeStamp() << std::endl;
    int err;
    
    #if TEST_IMAGE_PROCESSING
    err = test_imagePatchExtraction();
    if (err)
    {
        logger << "Test 'imagePatchExtraction' failed." << std::endl;
        return err;
    }
    #endif/*TEST_IMAGE_PROCESSING*/

    #if TEST_ESVM_BASIC_FUNCTIONALITY
    err = test_runBasicExemplarSvmFunctionalities();
    if (err)
    {
        logger << "Test 'runBasicExemplarSvmFunctionalities' failed." << std::endl;
        return err;
    }
    
    err = test_runBasicExemplarSvmClassification();
    if (err)
    {
        logger << "Test 'test_runBasicExemplarSvmClassification' failed." << std::endl;
        return err;
    }
    #endif/*TEST_ESVM_BASIC_FUNCTIONALITY*/
    
    // Number of patches to use in each direction, must fit within the ROIs (ex: 4x4 patches & ROI 128x128 -> 16 patches of 32x32)
    // Specifying Size(0,0) or Size(1,1) will result in not applying patches (use whole ROI)
    #if TEST_ESVM_BASIC_STILL2VIDEO
    cv::Size patchCounts = cv::Size(4, 4);
    err = test_runSingleSamplePerPersonStillToVideo(patchCounts);
    if (err)
    {
        logger << "Test 'runSingleSamplePerPersonStillToVideo' failed." << std::endl;
        return err;
    }
    #endif/*TEST_ESVM_BASIC_STILL2VIDEO*/

    #if ESVM_READ_DATA_FILES & 0b0001       // (1) Run ESVM training/testing using images and feature extraction on whole image
    // Specifying Size(0,0) or Size(1,1) will result in not applying patches (use whole ROI)
    cv::Size patchCounts = cv::Size(1, 1);
    cv::Size imageSize = cv::Size(64, 64);
    #elif ESVM_READ_DATA_FILES & 0b0010     // (2) Run ESVM training/testing using images and patch-based feature extraction
    // Number of patches to use in each direction, must fit within the ROIs (ex: 4x4 patches & ROI 128x128 -> 16 patches of 32x32)
    cv::Size patchCounts = cv::Size(4, 4);
    cv::Size imageSize = cv::Size(64, 64);
    #endif/* (1) or (2) params */
    #if ESVM_READ_DATA_FILES & (0b0001 | 0b0010)
    bool useSyntheticPositives = true;
    err = test_runSingleSamplePerPersonStillToVideo_FullChokePoint(imageSize, patchCounts, useSyntheticPositives);
    if (err)
    {
        logger << "Test 'test_runSingleSamplePerPersonStillToVideo_FullChokePoint' failed." << std::endl;
        return err;
    }
    #endif/* (1) or (2) test */
    #if ESVM_READ_DATA_FILES & 0b0100       // (4) Run ESVM training/testing using pre-generated whole image samples files
    err = test_runSingleSamplePerPersonStillToVideo_DataFiles_WholeImage();
    if (err)
    {        
        logger << "Test 'test_runSingleSamplePerPersonStillToVideo_DataFiles_WholeImage' failed." << std::endl;
        return err;
    }
    #endif/* (4) */
    #if ESVM_READ_DATA_FILES & 0b1000       // (8) Run ESVM training/testing using pre-generated (feature+patch)-based samples files
    int nPatches = 16;
    err = test_runSingleSamplePerPersonStillToVideo_DataFiles_FeatureAndPatchBased(nPatches);
    if (err)
    {        
        logger << "Test 'test_runSingleSamplePerPersonStillToVideo_DataFiles_FeatureAndPatchBased' failed." << std::endl;
        return err;
    }
    #endif/* (8) */

    logger << "All tests completed " << currentTimeStamp() << std::endl;
    return 0;
}
