#include "logger.h"
#include "generic.h"
#include "esvmOptions.h"
#include "esvmTests.h"
#include "createSampleFiles.h"

#include "boost/filesystem.hpp"
namespace bfs = boost::filesystem;

int main(int argc, char* argv[])
{
    logstream logger(LOGGER_FILE);
    displayHeader();
    displayOptions();
    try
    {
        /* ---------------
            unit tests
        --------------- */

        RETURN_ERROR(test_imagePaths());
        RETURN_ERROR(test_imagePatchExtraction());
        RETURN_ERROR(test_imagePreprocessing());
        RETURN_ERROR(test_multiLevelVectors());
        RETURN_ERROR(test_normalizationFunctions());
        RETURN_ERROR(test_performanceEvaluationFunctions());
        RETURN_ERROR(test_ESVM_BasicFunctionalities());
        RETURN_ERROR(test_ESVM_BasicClassification());
        RETURN_ERROR(test_runSingleSamplePerPersonStillToVideo(cv::Size(4, 4)));
        RETURN_ERROR(test_ESVM_ReadSampleFile_binary());
        RETURN_ERROR(test_ESVM_ReadSampleFile_libsvm());
        RETURN_ERROR(test_ESVM_ReadSampleFile_timing(2000, 500));
        RETURN_ERROR(test_ESVM_ReadSampleFile_compare());
        RETURN_ERROR(test_ESVM_WriteSampleFile_timing(2000, 500));
        RETURN_ERROR(test_ESVM_SaveLoadModelFile_binary());
        RETURN_ERROR(test_ESVM_SaveLoadModelFile_libsvm());
        RETURN_ERROR(test_ESVM_SaveLoadModelFile_compare());
        RETURN_ERROR(test_ESVM_ModelMemoryDealloc());
        RETURN_ERROR(test_ESVM_ModelMemoryReset());

        /* ----------------
          procedure tests
        ---------------- */

        RETURN_ERROR(proc_ReadDataFiles());
        RETURN_ERROR(proc_runSingleSamplePerPersonStillToVideo_TITAN(cv::Size(48, 48), cv::Size(3, 3), true));
        RETURN_ERROR(proc_runSingleSamplePerPersonStillToVideo_DataFiles_SAMAN());
        RETURN_ERROR(proc_runSingleSamplePerPersonStillToVideo_DataFiles_SimplifiedWorking());
        RETURN_ERROR(create_negatives());
    }
    catch(std::exception& ex)
    {
        logger << "Unhandled exception occurred: [" << ex.what() << "]" << std::endl;
    }

    logger << "All tests completed. " << currentTimeStamp() << std::endl;
    return 0;
}
