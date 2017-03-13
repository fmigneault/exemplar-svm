#include "logger.h"
#include "generic.h"
#include "esvmOptions.h"
#include "esvmTests.h"
#include "createSampleFiles.h"

#include "boost/filesystem.hpp"
namespace bfs = boost::filesystem;

int main(int argc, char* argv[])
{
    int err = 0;
    logstream logger(LOGGER_FILE);
    std::string header = "Starting new Exemplar-SVM test execution " + currentTimeStamp();
    logger << std::string(header.size(), '=') << std::endl << header << std::endl;
    test_outputOptions();
    try
    {
        // Check paths for tests
        #if TEST_IMAGE_PATHS
        test_imagePaths();
        #endif/*TEST_IMAGE_PATHS*/
    
        #if TEST_IMAGE_PATCH_EXTRACTION
        err = test_imagePatchExtraction();
        if (err)
        {
            logger << "Test 'test_imagePatchExtraction' failed (" << std::to_string(err) << ")." << std::endl;
            return err;
        }
        logger << "Test 'test_imagePatchExtraction' completed." << std::endl;
        #endif/*TEST_IMAGE_PATCH_EXTRACTION*/

        #if TEST_IMAGE_PREPROCESSING
        err = test_imagePreprocessing();
        if (err)
        {
            logger << "Test 'test_imagePreprocessing' failed (" << std::to_string(err) << ")." << std::endl;
            return err;
        }
        logger << "Test 'test_imagePreprocessing' completed." << std::endl;
        #endif/*TEST_IMAGE_PREPROCESSING*/

        #if TEST_MULTI_LEVEL_VECTORS
        err = test_multiLevelVectors();
        if (err)
        {
            logger << "Test 'test_multiLevelVectors' failed (" << std::to_string(err) << ")." << std::endl;
            return err;
        }
        logger << "Test 'test_multiLevelVectors' completed." << std::endl;
        #endif/*TEST_MULTI_LEVEL_VECTORS*/

        #if TEST_NORMALIZATION
        err = test_normalizationFunctions();
        if (err)
        {
            logger << "Test 'test_normalizationFunctions' failed (" << std::to_string(err) << ")." << std::endl;
            return err;
        }
        logger << "Test 'test_normalizationFunctions' completed." << std::endl;
        #endif/*TEST_NORMALIZATION*/

        #if TEST_PERF_EVAL_FUNCTIONS
        err = test_performanceEvaluationFunctions();
        if (err)
        {
            logger << "Test 'test_performanceEvaluationFunctions' failed (" << std::to_string(err) << ")." << std::endl;
            return err;
        }
        logger << "Test 'test_performanceEvaluationFunctions' completed." << std::endl;
        #endif/*TEST_PERF_EVAL_FUNCTIONS*/        

        #if TEST_ESVM_BASIC_FUNCTIONALITY
        err = test_ESVM_BasicFunctionalities();
        if (err)
        {
            logger << "Test 'test_ESVM_BasicFunctionalities' failed (" << std::to_string(err) << ")." << std::endl;
            return err;
        }
        logger << "Test 'test_ESVM_BasicFunctionalities' completed." << std::endl;
    
        err = test_ESVM_BasicClassification();
        if (err)
        {
            logger << "Test 'test_ESVM_BasicClassification' failed (" << std::to_string(err) << ")." << std::endl;
            return err;
        }
        logger << "Test 'test_ESVM_BasicClassification' completed." << std::endl;
        #endif/*TEST_ESVM_BASIC_FUNCTIONALITY*/
    
        // Number of patches to use in each direction, must fit within the ROIs (ex: 4x4 patches & ROI 128x128 -> 16 patches of 32x32)
        // Specifying Size(0,0) or Size(1,1) will result in not applying patches (use whole ROI)
        #if TEST_ESVM_BASIC_STILL2VIDEO
        cv::Size patchCounts = cv::Size(4, 4);
        err = test_runSingleSamplePerPersonStillToVideo(patchCounts);
        if (err)
        {
            logger << "Test 'test_runSingleSamplePerPersonStillToVideo' failed (" << std::to_string(err) << ")." << std::endl;
            return err;
        }
        logger << "Test 'test_runSingleSamplePerPersonStillToVideo' completed." << std::endl;
        #endif/*TEST_ESVM_BASIC_STILL2VIDEO*/

        #if TEST_ESVM_READ_SAMPLES_FILE_PARSER
        err = test_ESVM_ReadSampleFile_libsvm();
        if (err)
        {
            logger << "Test 'test_ESVM_ReadSampleFile_libsvm' failed (" << std::to_string(err) << ")." << std::endl;
            return err;
        }
        logger << "Test 'test_ESVM_ReadSampleFile_libsvm' completed." << std::endl;
        err = test_ESVM_ReadSampleFile_binary();
        if (err)
        {
            logger << "Test 'test_ESVM_ReadSampleFile_binary' failed (" << std::to_string(err) << ")." << std::endl;
            return err;
        }
        logger << "Test 'test_ESVM_ReadSampleFile_binary' completed." << std::endl;
        #endif/*TEST_ESVM_READ_SAMPLES_FILE_PARSER*/

        #if TEST_ESVM_READ_SAMPLES_FILE_TIMING
        size_t nSamplesRead = 2000;
        size_t nFeaturesRead = 500;
        err = test_ESVM_ReadSampleFile_timing(nSamplesRead, nFeaturesRead);
        if (err)
        {
            logger << "Test 'test_ESVM_ReadSampleFile_timing' failed (" << std::to_string(err) << ")." << std::endl;
            return err;
        }
        logger << "Test 'test_ESVM_ReadSampleFile_timing' completed." << std::endl;
        #endif/*TEST_ESVM_READ_SAMPLES_FILE_TIMING*/
        
        #if TEST_ESVM_READ_SAMPLES_FILE_FORMAT_COMPARE
        err = test_ESVM_ReadSampleFile_compare();
        if (err)
        {
            logger << "Test 'test_ESVM_ReadSampleFile_compare' failed (" << std::to_string(err) << ")." << std::endl;
            return err;
        }
        logger << "Test 'test_ESVM_ReadSampleFile_compare' completed." << std::endl;
        #endif/*TEST_ESVM_READ_SAMPLES_FILE_FORMAT_COMPARE*/
        
        #if TEST_ESVM_WRITE_SAMPLES_FILE_TIMING
        size_t nSamplesWrite = 2000;
        size_t nFeaturesWrite = 500;
        err = test_ESVM_WriteSampleFile_timing(nSamplesWrite, nFeaturesWrite);
        if (err)
        {
            logger << "Test 'test_ESVM_WriteSampleFile_timing' failed (" << std::to_string(err) << ")." << std::endl;
            return err;
        }
        logger << "Test 'test_ESVM_WriteSampleFile_timing' completed." << std::endl;
        #endif/*TEST_ESVM_WRITE_SAMPLES_FILE_TIMING*/

        #if TEST_ESVM_SAVE_LOAD_MODEL_FILE_PARSER
        err = test_ESVM_SaveLoadModelFile_libsvm();
        if (err)
        {
            logger << "Test 'test_ESVM_SaveLoadModelFile_libsvm' failed (" << std::to_string(err) << ")." << std::endl;
            return err;
        }
        logger << "Test 'test_ESVM_SaveLoadModelFile_libsvm' completed." << std::endl;
        err = test_ESVM_SaveLoadModelFile_binary();
        if (err)
        {
            logger << "Test 'test_ESVM_SaveLoadModelFile_binary' failed (" << std::to_string(err) << ")." << std::endl;
            return err;
        }
        logger << "Test 'test_ESVM_SaveLoadModelFile_binary' completed." << std::endl;
        #endif/*TEST_ESVM_SAVE_LOAD_MODEL_FILE_PARSER*/

        #if TEST_ESVM_SAVE_LOAD_MODEL_FILE_FORMAT_COMPARE
        err = test_ESVM_SaveLoadModelFile_compare();
        if (err)
        {
            logger << "Test 'test_ESVM_SaveLoadModelFile_compare' failed (" << std::to_string(err) << ")." << std::endl;
            return err;
        }
        logger << "Test 'test_ESVM_SaveLoadModelFile_compare' completed." << std::endl;
        #endif/*TEST_ESVM_SAVE_LOAD_MODEL_FILE_FORMAT_COMPARE*/

        #if TEST_READ_DATA_FILES & 0b00000001   // (1) Run ESVM training/testing using images and feature extraction on whole image
        // Specifying Size(0,0) or Size(1,1) will result in not applying patches (use whole ROI)
        cv::Size patchCounts = cv::Size(1, 1);
        cv::Size imageSize = cv::Size(64, 64);
        #elif TEST_READ_DATA_FILES & 0b00000010 // (2) Run ESVM training/testing using images and patch-based feature extraction
        // Number of patches to use in each direction, must fit within the ROIs (ex: 4x4 patches & ROI 128x128 -> 16 patches of 32x32)
        cv::Size patchCounts = cv::Size(3, 3);
        cv::Size imageSize = cv::Size(48, 48);
        #endif/* (1|2) params */
        #if TEST_READ_DATA_FILES & 0b00000011
        err = test_runSingleSamplePerPersonStillToVideo_FullChokePoint(imageSize, patchCounts);
        if (err)
        {
            logger << "Test 'test_runSingleSamplePerPersonStillToVideo_FullChokePoint' failed (" << std::to_string(err) << ")." << std::endl;
            return err;
        }
        logger << "Test 'test_runSingleSamplePerPersonStillToVideo_FullChokePoint' completed." << std::endl;
        #endif/* (1|2) test */
        #if TEST_READ_DATA_FILES & 0b00000100   // (4) Run ESVM training/testing using pre-generated whole image samples files
        err = test_runSingleSamplePerPersonStillToVideo_DataFiles_WholeImage();
        if (err)
        {        
            logger << "Test 'test_runSingleSamplePerPersonStillToVideo_DataFiles_WholeImage' failed (" << std::to_string(err) << ")." << std::endl;
            return err;
        }
        logger << "Test 'test_runSingleSamplePerPersonStillToVideo_DataFiles_WholeImage' completed." << std::endl;
        #endif/* (4) */
        #if TEST_READ_DATA_FILES & 0b00001000   // (8) Run ESVM training/testing using pre-generated (feature+patch)-based samples files
        int nPatches = patchCounts.width * patchCounts.height;
        err = test_runSingleSamplePerPersonStillToVideo_DataFiles_DescriptorAndPatchBased(nPatches);
        if (err)
        {        
            logger << "Test 'test_runSingleSamplePerPersonStillToVideo_DataFiles_DescriptorAndPatchBased' failed (" 
                   << std::to_string(err) << ")." << std::endl;
            return err;
        }
        logger << "Test 'test_runSingleSamplePerPersonStillToVideo_DataFiles_DescriptorAndPatchBased' completed." << std::endl;
        #endif/* (8) */
        #if TEST_READ_DATA_FILES & 0b11110000   // (16|32|64|128) Run ESVM training/testing using pre-generated patch-based negatives samples files
        err = test_runSingleSamplePerPersonStillToVideo_NegativesDataFiles_PositivesExtraction_PatchBased();
        if (err)
        {        
            logger << "Test 'test_runSingleSamplePerPersonStillToVideo_NegativesDataFiles_PositivesExtraction_PatchBased' failed (" 
                   << std::to_string(err) << ")." << std::endl;
            return err;
        }
        logger << "Test 'test_runSingleSamplePerPersonStillToVideo_NegativesDataFiles_PositivesExtraction_PatchBased' completed." << std::endl;
        #endif/* (16|32|64|128) */

        #if TEST_ESVM_TITAN
        cv::Size patchCounts = cv::Size(3, 3);
        cv::Size imageSize = cv::Size(48, 48);
        bool useSyntheticPositives = true;
        err = test_runSingleSamplePerPersonStillToVideo_TITAN(imageSize, patchCounts, useSyntheticPositives);
        if (err)
        {
            logger << "Test 'test_runSingleSamplePerPersonStillToVideo_TITAN' failed (" << std::to_string(err) << ")." << std::endl;
            return err;
        }
        logger << "Test 'test_runSingleSamplePerPersonStillToVideo_TITAN' completed." << std::endl;
        #endif/*TEST_ESVM_TITAN*/

        #if TEST_ESVM_SAMAN
        err = test_runSingleSamplePerPersonStillToVideo_DataFiles_SAMAN();
        if (err)
        {
            logger << "Test 'test_runSingleSamplePerPersonStillToVideo_DataFiles_SAMAN' failed (" << std::to_string(err) << ")." << std::endl;
            return err;
        }
        logger << "Test 'test_runSingleSamplePerPersonStillToVideo_DataFiles_SAMAN' completed." << std::endl;
        #endif/*TEST_ESVM_SAMAN*/

        #if TEST_ESVM_WORKING_PROCEDURE
        err = test_runSingleSamplePerPersonStillToVideo_DataFiles_SimplifiedWorkingProcedure();
        if (err)
        {
            logger << "Test 'test_runSingleSamplePerPersonStillToVideo_DataFiles_SimplifiedWorkingProcedure' failed (" 
                   << std::to_string(err) << ")." << std::endl;
            return err;
        }
        logger << "Test 'test_runSingleSamplePerPersonStillToVideo_DataFiles_SimplifiedWorkingProcedure' completed." << std::endl;
        #endif/*TEST_ESVM_WORKING_PROCEDURE*/

        #if PROC_ESVM_GENERATE_SAMPLE_FILES
        logger << "Starting process 'create_negatives'." << std::endl;
        err = create_negatives();
        if (err)
        {
            logger << "Process 'create_negatives' failed ("
                   << std::to_string(err) << ")." << std::endl;
            return err;
        }
        logger << "Process 'create_negatives' completed." << std::endl;
        #endif/*PROC_ESVM_GENERATE_SAMPLE_FILES*/
    }
    catch(std::exception& ex)
    {
        logger << "Unhandled exception occurred: [" << ex.what() << "]" << std::endl;
    }

    if (!err)
        logger << "All tests completed. " << currentTimeStamp() << std::endl;
    return 0;
}
