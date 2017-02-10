#include "logger.h"
#include "generic.h"
#include "esvmOptions.h"
#include "esvmTests.h"

#include "boost/filesystem.hpp"
namespace bfs = boost::filesystem;

int main(int argc, char* argv[])
{
    int err;
    logstream logger(LOGGER_FILE);
    logger << "=================================================================" << std::endl
           << "Starting new Exemplar-SVM test execution " << currentTimeStamp() << std::endl;
    test_outputOptions();
    try
    {

        // Check paths for tests
        #if TEST_IMAGE_PATHS
        test_imagePaths();
        #endif/*TEST_IMAGE_PATHS*/
        #if ESVM_READ_DATA_FILES != 0b0000 || ESVM_WRITE_DATA_FILES != 0
        ASSERT_LOG(checkPathEndSlash(dataFilePath), "Data file path doesn't end with slash character");
        #endif/*ESVM_READ_DATA_FILES | ESVM_WRITE_DATA_FILES*/
    
        #if TEST_IMAGE_PROCESSING
        err = test_imagePatchExtraction();
        if (err)
        {
            logger << "Test 'test_imagePatchExtraction' failed." << std::endl;
            return err;
        }
        logger << "Test 'test_imagePatchExtraction' completed." << std::endl;
        #endif/*TEST_IMAGE_PROCESSING*/

        #if TEST_MULTI_LEVEL_VECTORS
        err = test_multiLevelVectors();
        if (err)
        {
            logger << "Test 'test_multiLevelVectors' failed." << std::endl;
            return err;
        }
        logger << "Test 'test_multiLevelVectors' completed." << std::endl;
        #endif/*TEST_MULTI_LEVEL_VECTORS*/

        #if TEST_NORMALIZATION
        err = test_normalizationFunctions();
        if (err)
        {
            logger << "Test 'test_normalizationFunctions' failed." << std::endl;
            return err;
        }
        logger << "Test 'test_normalizationFunctions' completed." << std::endl;
        #endif/*TEST_NORMALIZATION*/

        #if TEST_ESVM_BASIC_FUNCTIONALITY
        err = test_runBasicExemplarSvmFunctionalities();
        if (err)
        {
            logger << "Test 'test_runBasicExemplarSvmFunctionalities' failed." << std::endl;
            return err;
        }
        logger << "Test 'test_runBasicExemplarSvmFunctionalities' completed." << std::endl;
    
        err = test_runBasicExemplarSvmClassification();
        if (err)
        {
            logger << "Test 'test_runBasicExemplarSvmClassification' failed." << std::endl;
            return err;
        }
        logger << "Test 'test_runBasicExemplarSvmClassification' completed." << std::endl;
        #endif/*TEST_ESVM_BASIC_FUNCTIONALITY*/
    
        // Number of patches to use in each direction, must fit within the ROIs (ex: 4x4 patches & ROI 128x128 -> 16 patches of 32x32)
        // Specifying Size(0,0) or Size(1,1) will result in not applying patches (use whole ROI)
        #if TEST_ESVM_BASIC_STILL2VIDEO
        cv::Size patchCounts = cv::Size(4, 4);
        err = test_runSingleSamplePerPersonStillToVideo(patchCounts);
        if (err)
        {
            logger << "Test 'test_runSingleSamplePerPersonStillToVideo' failed." << std::endl;
            return err;
        }
        logger << "Test 'test_runSingleSamplePerPersonStillToVideo' completed." << std::endl;
        #endif/*TEST_ESVM_BASIC_STILL2VIDEO*/

        #if ESVM_READ_DATA_FILES & 0b0001       // (1) Run ESVM training/testing using images and feature extraction on whole image
        // Specifying Size(0,0) or Size(1,1) will result in not applying patches (use whole ROI)
        cv::Size patchCounts = cv::Size(1, 1);
        cv::Size imageSize = cv::Size(64, 64);
        #elif ESVM_READ_DATA_FILES & 0b0010     // (2) Run ESVM training/testing using images and patch-based feature extraction
        // Number of patches to use in each direction, must fit within the ROIs (ex: 4x4 patches & ROI 128x128 -> 16 patches of 32x32)
        cv::Size patchCounts = cv::Size(3, 3);
        cv::Size imageSize = cv::Size(48, 48);
        #endif/* (1) or (2) params */
        #if ESVM_READ_DATA_FILES & (0b0001 | 0b0010)
        err = test_runSingleSamplePerPersonStillToVideo_FullChokePoint(imageSize, patchCounts);
        if (err)
        {
            logger << "Test 'test_runSingleSamplePerPersonStillToVideo_FullChokePoint' failed." << std::endl;
            return err;
        }
        logger << "Test 'test_runSingleSamplePerPersonStillToVideo_FullChokePoint' completed." << std::endl;
        #endif/* (1) or (2) test */
        #if ESVM_READ_DATA_FILES & 0b0100       // (4) Run ESVM training/testing using pre-generated whole image samples files
        err = test_runSingleSamplePerPersonStillToVideo_DataFiles_WholeImage();
        if (err)
        {        
            logger << "Test 'test_runSingleSamplePerPersonStillToVideo_DataFiles_WholeImage' failed." << std::endl;
            return err;
        }
        logger << "Test 'test_runSingleSamplePerPersonStillToVideo_DataFiles_WholeImage' completed." << std::endl;
        #endif/* (4) */
        #if ESVM_READ_DATA_FILES & 0b1000       // (8) Run ESVM training/testing using pre-generated (feature+patch)-based samples files
        int nPatches = patchCounts.width * patchCounts.height;
        err = test_runSingleSamplePerPersonStillToVideo_DataFiles_DescriptorAndPatchBased(nPatches);
        if (err)
        {        
            logger << "Test 'test_runSingleSamplePerPersonStillToVideo_DataFiles_DescriptorAndPatchBased' failed." << std::endl;
            return err;
        }
        logger << "Test 'test_runSingleSamplePerPersonStillToVideo_DataFiles_DescriptorAndPatchBased' completed." << std::endl;
        #endif/* (8) */

        #if TEST_ESVM_TITAN
        cv::Size patchCounts = cv::Size(3, 3);
        cv::Size imageSize = cv::Size(48, 48);
        bool useSyntheticPositives = true;
        err = test_runSingleSamplePerPersonStillToVideo_TITAN(imageSize, patchCounts, useSyntheticPositives);
        if (err)
        {
            logger << "Test 'test_runSingleSamplePerPersonStillToVideo_TITAN' failed." << std::endl;
            return err;
        }
        logger << "Test 'test_runSingleSamplePerPersonStillToVideo_TITAN' completed." << std::endl;
        #endif/*TEST_ESVM_TITAN*/

        #if TEST_ESVM_SAMAN
        err = test_runSingleSamplePerPersonStillToVideo_DataFiles_SAMAN();
        if (err)
        {
            logger << "Test 'test_runSingleSamplePerPersonStillToVideo_DataFiles_SAMAN' failed." << std::endl;
            return err;
        }
        logger << "Test 'test_runSingleSamplePerPersonStillToVideo_DataFiles_SAMAN' completed." << std::endl;
        #endif/*TEST_ESVM_SAMAN*/

    }
    catch(std::exception& ex)
    {
        logger << "Unhandled exception occurred: [" << ex.what() << "]" << std::endl;
    }

    logger << "All tests completed. " << currentTimeStamp() << std::endl;
    return 0;
}
