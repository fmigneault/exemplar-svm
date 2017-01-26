#include "logger.h"
#include "generic.h"
#include "esvmOptions.h"
#include "esvmTests.h"

#include "boost/filesystem.hpp"
namespace bfs = boost::filesystem;

int main(int argc, char* argv[])
{
    logstream logger(LOGGER_FILE);
    logger << "=================================================================" << std::endl
           << "Starting new Exemplar-SVM test execution " << currentTimeStamp() << std::endl;
    
    // Check image paths for tests
    #if TEST_IMAGE_PATHS
    // Local
    ASSERT_LOG(bfs::is_directory(roiVideoImagesPath), "Cannot find ROI directory");
    ASSERT_LOG(bfs::is_directory(refStillImagesPath), "Cannot find REF directory");
    // ChokePoint
    ASSERT_LOG(bfs::is_directory(rootChokePointPath), "Cannot find ChokePoint root directory");    
    ASSERT_LOG(bfs::is_directory(roiChokePointCroppedFacePath), "Cannot find ChokePoint cropped faces root directory");
    ASSERT_LOG(bfs::is_directory(roiChokePointFastDTTrackPath), "Cannot find ChokePoint FAST-DT tracks root directory");
    ASSERT_LOG(bfs::is_directory(roiChokePointEnrollStillPath), "Cannot find ChokePoint enroll stills root directory");
    // TITAN Unit
    ASSERT_LOG(bfs::is_directory(rootTitanUnitPath), "Cannot find TITAN Unit root directory");
    ASSERT_LOG(bfs::is_directory(roiTitanUnitFastDTTrackPath), "Cannot find TITAN Unit FAST-DT tracks root directory");
    ASSERT_LOG(bfs::is_directory(roiTitanUnitEnrollStillPath), "Cannot find TITAN Unit enroll stills root directory");  
    // COX-S2V
    ASSERT_LOG(bfs::is_directory(rootCOXS2VPath), "Cannot find COX-S2V root directory");    
    ASSERT_LOG(bfs::is_directory(roiCOXS2VTestingVideoPath), "Cannot find COX-S2V testing video root directory");
    ASSERT_LOG(bfs::is_directory(roiCOXS2VEnrollStillsPath), "Cannot find COX-S2V enroll stills root directory");
    ASSERT_LOG(bfs::is_directory(roiCOXS2VAllImgsStillPath), "Cannot find COX-S2V all image stills root directory");
    ASSERT_LOG(bfs::is_directory(roiCOXS2VEyeLocaltionPath), "Cannot find COX-S2V eye location root directory");
    #endif/*TEST_IMAGE_PATHS*/

    int err;
    
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
    bool useSyntheticPositives = true;
    err = test_runSingleSamplePerPersonStillToVideo_FullChokePoint(imageSize, patchCounts, useSyntheticPositives);
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

    logger << "All tests completed. " << currentTimeStamp() << std::endl;
    return 0;
}
