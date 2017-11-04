#ifdef ESVM_HAS_TESTS

#include "esvmOptions.h"
#include "esvmPaths.h"
#include "esvmTests.h"
#include "esvmTypes.h"
#include "esvmUtils.h"
#include "esvm.h"

#include "feHOG.h"
#if ESVM_HAS_FELBP
#include "feLBP.h"
#endif/*ESVM_HAS_FELBP*/

#include "CommonCpp.h"

#include "boost/filesystem.hpp"
namespace bfs = boost::filesystem;

namespace esvm {
namespace test {

/* ======================
    UTILITY FUNCTIONS
====================== */

// parameters employed by 'dummy' svm_model builder/destructor for testing purposes
#define DUMMY_SVM_MODEL_NSV 5
#define DUMMY_SVM_MODEL_NCLASS 2
#define DUMMY_SVM_MODEL_NFEATURES 3

// builds a dummy model with all valid parameters to test read/write procedures
svm_model* buildDummyExemplarSvmModel(FreeModelState free_sv)
{
    svm_model *model = ESVM::makeEmptyModel();
    try
    {
        model->param.kernel_type = LINEAR;
        model->param.svm_type = C_SVC;
        model->param.C = 1;
        model->free_sv = free_sv;
        model->nr_class = DUMMY_SVM_MODEL_NCLASS;
        model->l = DUMMY_SVM_MODEL_NSV;
        model->label = Malloc(int, model->nr_class);
        model->label[0] = ESVM_POSITIVE_CLASS;
        model->label[1] = ESVM_NEGATIVE_CLASS;
        if (free_sv == FreeModelState::PARAM || free_sv == FreeModelState::MULTI) {
            model->param.nr_weight = model->nr_class;
            model->param.weight = Malloc(double, model->nr_class);
            model->param.weight[0] = model->l - 1.0;
            model->param.weight[1] = 1.0;
            model->param.weight_label = Malloc(int, model->nr_class);
            model->param.weight_label[0] = model->label[0];
            model->param.weight_label[1] = model->label[1];
        }
        if (free_sv == FreeModelState::MODEL || free_sv == FreeModelState::MULTI) {
            model->rho = Malloc(double, 1);
            model->rho[0] = 2.5;
            model->sv_indices = Malloc(int, model->l);
            for (int i = 0; i < model->l; ++i)
                model->sv_indices[i] = i;
            model->sv_coef = (double**)malloc(sizeof(double*)*(model->nr_class - 1));
            std::vector<double> sv{ 3.5, -0.1, -0.2, -0.1, -0.2 };
            for (int i = 0; i < model->nr_class - 1; ++i) {
                model->sv_coef[i] = Malloc(double, model->l);
                for (int j = 0; j < model->l; ++j)
                    model->sv_coef[i][j] = sv[j];
            }
            model->nSV = Malloc(int, model->nr_class);
            model->nSV[0] = 1;
            model->nSV[1] = model->l - 1;
            model->SV = Malloc(svm_node*, model->l);
            for (int sv = 0; sv < model->l; ++sv) {
                model->SV[sv] = Malloc(svm_node, DUMMY_SVM_MODEL_NFEATURES + 1);
                for (int f = 0; f < DUMMY_SVM_MODEL_NFEATURES + 1; ++f)
                    model->SV[sv][f].index = (f == DUMMY_SVM_MODEL_NFEATURES) ? -1 : f;
            }
            model->SV[0][0].value = 0.50;   model->SV[0][1].value = 0.75;   model->SV[0][2].value = 0.25;
            model->SV[1][0].value = 0.20;   model->SV[1][1].value = 0.75;   model->SV[1][2].value = 0.10;
            model->SV[2][0].value = 0.30;   model->SV[2][1].value = 0.75;   model->SV[2][2].value = 0.05;
            model->SV[3][0].value = 0.25;   model->SV[3][1].value = 0.75;   model->SV[3][2].value = 0.00;
            model->SV[4][0].value = 0.15;   model->SV[4][1].value = 0.75;   model->SV[4][2].value = 0.15;
        }
        #if ESVM_USE_PREDICT_PROBABILITY
        model->param.probability = 1;
        model->probA = Malloc(double, 1);
        model->probA[0] = 1.2;
        model->probB = Malloc(double, 1);
        model->probB[0] = 0.8;
        #else
        model->param.probability = 0;
        model->probA = nullptr;
        model->probB = nullptr;
        #endif/*ESVM_USE_PREDICT_PROBABILITY*/

        ASSERT_THROW(ESVM::checkModelParameters(model), "Dummy 'svm_model' generation did not respect ESVM requirements");
    }
    catch (std::exception& ex)
    {
        logstream logger(LOGGER_FILE);
        logger << "Error occurred when building the dummy ESVM model for testing" << std::endl
               << "Exception: [" << ex.what() << "]" << std::endl;
        ESVM::destroyModel(&model);
        throw ex;  // re-throw
    }
    return model;
}

// destroys all the contained memory references inside an 'svm_model' created with 'buildDummyExemplarSvmModel'
void destroyDummyExemplarSvmModelContent(svm_model *model, FreeModelState free_sv)
{
    ///logstream logger(LOGGER_FILE);///TODO REMOVE
    if (!model) return;
    ///logger << "CLEANUP - MODEL !null" << std::endl;///TODO REMOVE
    try
    {
        bool destroyParam = free_sv == FreeModelState::PARAM || FreeModelState::MULTI;
        bool destroyModel = free_sv == FreeModelState::MODEL || FreeModelState::MULTI;

        if (destroyParam) {
            ///logger << "CLEANUP - DEL weight" << std::endl;///TODO REMOVE
            delete[] model->param.weight;
            ///logger << "CLEANUP - DEL weight_label" << std::endl;///TODO REMOVE
            delete[] model->param.weight_label;
        }
        ///logger << "CLEANUP - DEL label" << std::endl;///TODO REMOVE
        delete[] model->label;
        ///logger << "CLEANUP - DEL probA" << std::endl;///TODO REMOVE

        delete[] model->probA;
        ///logger << "CLEANUP - DEL probB" << std::endl;///TODO REMOVE
        delete[] model->probB;
        ///logger << "CLEANUP - DEL rho" << std::endl;///TODO REMOVE
        delete[] model->rho;
        ///logger << "CLEANUP - DEL nSV" << std::endl;///TODO REMOVE
        delete[] model->nSV;
        ///logger << "CLEANUP - DEL sv_coef" << std::endl;///TODO REMOVE
        if (model->sv_coef)
            for (int c = 0; c < DUMMY_SVM_MODEL_NCLASS - 1; ++c)
                delete[] model->sv_coef[c];
        ///logger << "CLEANUP - DEL sv_coef*" << std::endl;///TODO REMOVE
        delete[] model->sv_coef;
        ///logger << "CLEANUP - DEL SV" << std::endl;///TODO REMOVE
        if (model->SV)
            for (int sv = 0; sv < DUMMY_SVM_MODEL_NSV; ++sv)
                delete[] model->SV[sv];
        ///logger << "CLEANUP - DEL SV*" << std::endl;///TODO REMOVE
        delete[] model->SV;
    }
    catch (std::exception& ex)
    {
        logstream logger(LOGGER_FILE);
        logger << "Error occurred when destroying the dummy ESVM model for testing" << std::endl
               << "Exception: [" << ex.what() << "]" << std::endl;
        throw ex;  // re-throw
    }
}

void generateDummySamples(std::vector<FeatureVector>& samples, std::vector<int>& targetOutputs, size_t nSamples, size_t nFeatures)
{
    std::srand(0);
    samples = std::vector<FeatureVector>(nSamples);
    targetOutputs = std::vector<int>(nSamples, ESVM_NEGATIVE_CLASS);
    for (size_t s = 0; s < nSamples; ++s)
    {
        samples[s] = FeatureVector(nFeatures);
        for (size_t f = 0; f < nFeatures; ++f)
            samples[s][f] = ((double)std::rand() / (double)RAND_MAX);
    }
}

bool generateDummySampleFile_libsvm(std::string filePath, size_t nSamples, size_t nFeatures)
{
    std::ofstream sampleFile(filePath);
    if (!sampleFile) return false;
    std::srand(0);
    for (size_t s = 0; s < nSamples; ++s)
    {
        sampleFile << std::to_string(ESVM_NEGATIVE_CLASS);
        for (size_t f = 0; f < nFeatures; ++f)
            sampleFile << " " << f + 1 << ":" << ((double)std::rand() / (double)RAND_MAX);
        sampleFile << " -1:0" << std::endl; // can be omitted or not (parser should handle both cases)
    }
    if (sampleFile.is_open()) sampleFile.close();
    return true;
}

bool generateDummySampleFile_binary(std::string filePath, size_t nSamples, size_t nFeatures)
{
    std::vector<FeatureVector> samples;
    std::vector<int> outputs;
    generateDummySamples(samples, outputs, nSamples, nFeatures);
    ESVM::writeSampleDataFile(filePath, samples, outputs, BINARY);
    return bfs::is_regular_file(filePath);
}

void displayHeader()
{
    logstream logger(LOGGER_FILE);
    std::string header = "Starting new Exemplar-SVM test execution " + currentTimeStamp();
    logger << std::string(header.size(), '=') << std::endl << header << std::endl;
}

void displayOptions()
{
    logstream logger(LOGGER_FILE);
    std::string tab = "   ";

    logger << "Options:" << std::endl
           << tab << "ESVM:" << std::endl
           << tab << tab << "ESVM_USE_HOG:                                    " << ESVM_USE_HOG << std::endl
           << tab << tab << "ESVM_USE_LBP:                                    " << ESVM_USE_LBP << std::endl
           << tab << tab << "ESVM_USE_HIST_EQUAL:                             " << ESVM_USE_HIST_EQUAL << std::endl
           << tab << tab << "ESVM_USE_PREDICT_PROBABILITY:                    " << ESVM_USE_PREDICT_PROBABILITY << std::endl
           << tab << tab << "ESVM_POSITIVE_CLASS:                             " << ESVM_POSITIVE_CLASS << std::endl
           << tab << tab << "ESVM_NEGATIVE_CLASS:                             " << ESVM_NEGATIVE_CLASS << std::endl
           #if   ESVM_USE_LIBSVM
           << tab << tab << "ESVM_BINARY_HEADER_MODEL:                        " << ESVM_BINARY_HEADER_MODEL_LIBSVM << std::endl
           #elif ESVM_USE_LIBLINEAR
           << tab << tab << "ESVM_BINARY_HEADER_MODEL:                        " << ESVM_BINARY_HEADER_MODEL_LIBLINEAR << std::endl
           #endif/*esvm impl*/
           << tab << tab << "ESVM_BINARY_HEADER_SAMPLES:                      " << ESVM_BINARY_HEADER_SAMPLES << std::endl
           << tab << tab << "ESVM_ROI_CROP_RATIO:                             " << ESVM_ROI_CROP_RATIO << std::endl
           << tab << tab << "ESVM_ROI_PREPROCESS_MODE:                        " << ESVM_ROI_PREPROCESS_MODE << std::endl
           << tab << tab << "ESVM_WEIGHTS_MODE:                               " << ESVM_WEIGHTS_MODE << std::endl
           << tab << tab << "ESVM_FEATURE_NORM_MODE:                          " << ESVM_FEATURE_NORM_MODE << std::endl
           << tab << tab << "ESVM_FEATURE_NORM_CLIP:                          " << ESVM_FEATURE_NORM_CLIP << std::endl
           << tab << tab << "ESVM_SCORE_NORM_MODE:                            " << ESVM_SCORE_NORM_MODE << std::endl
           << tab << tab << "ESVM_SCORE_NORM_CLIP:                            " << ESVM_SCORE_NORM_CLIP << std::endl
           << tab << tab << "ESVM_READ_LIBSVM_PARSER_MODE:                    " << ESVM_READ_LIBSVM_PARSER_MODE << std::endl
           << tab << "TEST:" << std::endl
           << tab << tab << "TEST_CHOKEPOINT_SEQUENCES_MODE:                  " << TEST_CHOKEPOINT_SEQUENCES_MODE << std::endl
           << tab << tab << "TEST_USE_SYNTHETIC_GENERATION:                   " << TEST_USE_SYNTHETIC_GENERATION << std::endl
           << tab << tab << "TEST_DUPLICATE_COUNT:                            " << TEST_DUPLICATE_COUNT << std::endl
           << tab << tab << "TEST_USE_OTHER_POSITIVES_AS_NEGATIVES:           " << TEST_USE_OTHER_POSITIVES_AS_NEGATIVES << std::endl
           << tab << tab << "TEST_FEATURES_NORM_MODE:                " << TEST_FEATURES_NORM_MODE << std::endl
           << tab << tab << "TEST_PATHS:                                      " << TEST_PATHS << std::endl
           << tab << tab << "TEST_IMAGE_PATCH_EXTRACTION:                     " << TEST_IMAGE_PATCH_EXTRACTION << std::endl
           << tab << tab << "TEST_IMAGE_PREPROCESSING:                        " << TEST_IMAGE_PREPROCESSING << std::endl
           << tab << tab << "TEST_MULTI_LEVEL_VECTORS:                        " << TEST_MULTI_LEVEL_VECTORS << std::endl
           << tab << tab << "TEST_NORMALIZATION:                              " << TEST_NORMALIZATION << std::endl
           << tab << tab << "TEST_PERF_EVAL_FUNCTIONS:                        " << TEST_PERF_EVAL_FUNCTIONS << std::endl
           << tab << tab << "TEST_ESVM_BASIC_FUNCTIONALITY:                   " << TEST_ESVM_BASIC_FUNCTIONALITY << std::endl
           << tab << tab << "TEST_ESVM_BASIC_CLASSIFICATION:                  " << TEST_ESVM_BASIC_CLASSIFICATION << std::endl
           << tab << tab << "TEST_ESVM_READ_SAMPLES_FILE_TIMING:              " << TEST_ESVM_READ_SAMPLES_FILE_TIMING << std::endl
           << tab << tab << "TEST_ESVM_READ_SAMPLES_FILE_PARSER_BINARY:       " << TEST_ESVM_READ_SAMPLES_FILE_PARSER_BINARY << std::endl
           << tab << tab << "TEST_ESVM_READ_SAMPLES_FILE_PARSER_LIBSVM:       " << TEST_ESVM_READ_SAMPLES_FILE_PARSER_LIBSVM << std::endl
           << tab << tab << "TEST_ESVM_READ_SAMPLES_FILE_FORMAT_COMPARE:      " << TEST_ESVM_READ_SAMPLES_FILE_FORMAT_COMPARE << std::endl
           << tab << tab << "TEST_ESVM_WRITE_SAMPLES_FILE_TIMING:             " << TEST_ESVM_WRITE_SAMPLES_FILE_TIMING << std::endl
           << tab << tab << "TEST_ESVM_SAVE_LOAD_MODEL_FILE_PARSER_BINARY:    " << TEST_ESVM_SAVE_LOAD_MODEL_FILE_PARSER_BINARY << std::endl
           << tab << tab << "TEST_ESVM_SAVE_LOAD_MODEL_FILE_PARSER_LIBSVM:    " << TEST_ESVM_SAVE_LOAD_MODEL_FILE_PARSER_LIBSVM << std::endl
           << tab << tab << "TEST_ESVM_SAVE_LOAD_MODEL_FILE_FORMAT_COMPARE:   " << TEST_ESVM_SAVE_LOAD_MODEL_FILE_FORMAT_COMPARE << std::endl
           << tab << tab << "TEST_ESVM_MODEL_STRUCT_SVM_PARAMS:               " << TEST_ESVM_MODEL_STRUCT_SVM_PARAMS << std::endl
           << tab << tab << "TEST_ESVM_MODEL_MEMORY_OPERATIONS:               " << TEST_ESVM_MODEL_MEMORY_OPERATIONS << std::endl
           << tab << tab << "TEST_ESVM_MODEL_MEMORY_PARAM_CHECK:              " << TEST_ESVM_MODEL_MEMORY_PARAM_CHECK << std::endl
           << tab << "PROCEDURES:" << std::endl
           << tab << tab << "PROC_READ_DATA_FILES:                            " << displayAsBinary<8>(PROC_READ_DATA_FILES, true) << std::endl
           << tab << tab << "PROC_WRITE_DATA_FILES:                           " << PROC_WRITE_DATA_FILES << std::endl
           << tab << tab << "PROC_ESVM_BASIC_STILL2VIDEO:                     " << PROC_ESVM_BASIC_STILL2VIDEO << std::endl
           << tab << tab << "PROC_ESVM_TITAN:                                 " << PROC_ESVM_TITAN << std::endl
           << tab << tab << "PROC_ESVM_SAMAN:                                 " << PROC_ESVM_SAMAN << std::endl
           << tab << tab << "PROC_ESVM_SIMPLIFIED_WORKING:                    " << PROC_ESVM_SIMPLIFIED_WORKING << std::endl
           << tab << tab << "PROC_ESVM_FULL_GENERATION_TESTING:               " << PROC_ESVM_FULL_GENERATION_TESTING << std::endl
           << tab << tab << "PROC_ESVM_GENERATE_CONVERTED_IMAGES:             " << PROC_ESVM_GENERATE_CONVERTED_IMAGES << std::endl
           << tab << tab << "PROC_ESVM_GENERATE_SAMPLE_FILES:                 " << PROC_ESVM_GENERATE_SAMPLE_FILES << std::endl
           << tab << tab << "PROC_ESVM_GENERATE_SAMPLE_FILES_BINARY:          " << PROC_ESVM_GENERATE_SAMPLE_FILES_BINARY << std::endl
           << tab << tab << "PROC_ESVM_GENERATE_SAMPLE_FILES_LIBSVM:          " << PROC_ESVM_GENERATE_SAMPLE_FILES_LIBSVM << std::endl
           << tab << tab << "PROC_ESVM_GENERATE_SAMPLE_FILES_SESSION:         " << PROC_ESVM_GENERATE_SAMPLE_FILES_SESSION << std::endl
           << tab << tab << "PROC_ESVM_GENERATE_SAMPLE_FILES_REPLICATION:     " << PROC_ESVM_GENERATE_SAMPLE_FILES_REPLICATION << std::endl
           << tab << tab << "PROC_ESVM_GENERATE_SAMPLE_FILES_NEGATIVE_COUNT:  " << PROC_ESVM_GENERATE_SAMPLE_FILES_NEGATIVE_COUNT << std::endl;
}

/* ==========
    TESTS
========== */

int test_paths()
{
    #if TEST_PATHS
    logstream logger(LOGGER_FILE);
    logger << "Running '" << __func__ << "' test..." << std::endl;
    using namespace esvm::path;

    try{
    logger << "FORCING A 'THROW'" << std::endl;

    logger << "OK1" << std::endl;
    logger << "OK2" << std::endl;
    //ASSERT_LOG(bfs::is_directory(p), "NOPE1");
    logger << "OK3" << std::endl;
    logger << roiVideoImagesPath << std::endl;
    logger << "-----" << std::endl;
    bfs::path p(roiVideoImagesPath.c_str());
    logger << "....." << std::endl;
    logger << p.string() << std::endl;
    logger << "=====" << std::endl;
    ASSERT_LOG(bfs::is_directory(p), "NOPE2");
    logger << "OK4" << std::endl;
    ASSERT_LOG(1, "11");
    ASSERT_LOG(bfs::exists(roiVideoImagesPath), "Cannot find ROI directory");
    logger << "OKO" << std::endl;
    boost::system::error_code ec;
    ASSERT_LOG(bfs::is_directory(roiVideoImagesPath, ec), "Cannot find ROI directory");
    ASSERT_LOG(0, "13");
    ASSERT_LOG(bfs::is_directory(roiVideoImagesPath), "Cannot find ROI directory");
    THROW("22");
    logger << "OK" << std::endl;
    }
    catch (std::exception& ex)
    {
        logger << "THE HELL?? [" << ex.what() << "]" << std::endl;
    }

    // ESVM build/test paths
    ASSERT_LOG(bfs::is_directory(roiVideoImagesPath), "Cannot find ROI directory");
    ASSERT_LOG(bfs::is_directory(refStillImagesPath), "Cannot find REF directory");
    ASSERT_LOG(bfs::is_directory(negativeSamplesDir), "Cannot find negative samples directory");
    ASSERT_LOG(bfs::is_directory(testingSamplesDir),  "Cannot find testing probe samples directory");
    ASSERT_LOG(checkPathEndSlash(roiVideoImagesPath), "ROI directory doesn't end with slash character");
    ASSERT_LOG(checkPathEndSlash(refStillImagesPath), "REF directory doesn't end with slash character");
    ASSERT_LOG(checkPathEndSlash(negativeSamplesDir), "Negative samples directory doesn't end with slash character");
    ASSERT_LOG(checkPathEndSlash(testingSamplesDir),  "Testing probe samples directory doesn't end with slash character");
    // OpenCV
    ASSERT_LOG(bfs::is_directory(sourcesOpenCV), "Cannot find OpenCV's root sources directory");
    ASSERT_LOG(checkPathEndSlash(sourcesOpenCV), "OpenCV's root sources directory doesn't end with slash character");
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

    #else/*TEST_PATHS*/
    return passThroughDisplayTestStatus(__func__, SKIPPED);
    #endif/*TEST_PATHS*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

int test_imagePatchExtraction(void)
{
    #if TEST_IMAGE_PATCH_EXTRACTION
    logstream logger(LOGGER_FILE);
    logger << "Running '" << __func__ << "' test..." << std::endl;

    int rawData[24] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 };
    cv::Mat testImg(4, 6, CV_32S, rawData);                                         // 6x4 image with above data filled line by line
    std::vector<cv::Mat> testPatches = imSplitPatches(testImg, cv::Size(3, 2));     // 6 patches of 2x2
    // check number of patches extracted
    if (testPatches.size() != 6)
    {
        logger << "Invalid number of patches extracted (count: " << testPatches.size() << ", expected: 6)" << std::endl;
        return passThroughDisplayTestStatus(__func__, -1);
    }
    // check patch dimensions
    for (int p = 0; p < 6; ++p)
    {
        if (testPatches[p].size() != cv::Size(2, 2))
        {
            logger << "Invalid image size for patch " << p << " (size: " << testPatches[p].size() << ", expected: (2,2))" << std::endl;
            return passThroughDisplayTestStatus(__func__, -2);
        }
    }

    // check pixel values of patches
    if (!cv::countNonZero(testPatches[0] != cv::Mat(2, 2, CV_32S, { 1,2,7,8 })))
    {
        logger << "Invalid data for patch 0" << std::endl << testPatches[0] << std::endl;
        return passThroughDisplayTestStatus(__func__, -3);
    }
    if (!cv::countNonZero(testPatches[1] != cv::Mat(2, 2, CV_32S, { 3,4,9,10 })))
    {
        logger << "Invalid data for patch 1" << std::endl << testPatches[1] << std::endl;
        return passThroughDisplayTestStatus(__func__, -4);
    }
    if (!cv::countNonZero(testPatches[2] != cv::Mat(2, 2, CV_32S, { 5,6,11,12 })))
    {
        logger << "Invalid data for patch 2" << std::endl << testPatches[2] << std::endl;
        return passThroughDisplayTestStatus(__func__, -5);
    }
    if (!cv::countNonZero(testPatches[3] != cv::Mat(2, 2, CV_32S, { 13,14,19,20 })))
    {
        logger << "Invalid data for patch 3" << std::endl << testPatches[3] << std::endl;
        return passThroughDisplayTestStatus(__func__, -6);
    }
    if (!cv::countNonZero(testPatches[4] != cv::Mat(2, 2, CV_32S, { 15,16,21,22 })))
    {
        logger << "Invalid data for patch 4" << std::endl << testPatches[4] << std::endl;
        return passThroughDisplayTestStatus(__func__, -7);
    }
    if (!cv::countNonZero(testPatches[5] != cv::Mat(2, 2, CV_32S, { 17,18,23,24 })))
    {
        logger << "Invalid data for patch 5" << std::endl << testPatches[5] << std::endl;
        return passThroughDisplayTestStatus(__func__, -8);
    }

    #else/*TEST_IMAGE_PATCH_EXTRACTION*/
    return passThroughDisplayTestStatus(__func__, SKIPPED);
    #endif/*TEST_IMAGE_PATCH_EXTRACTION*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

int test_imagePreprocessing()
{
    #if TEST_IMAGE_PREPROCESSING
    logstream logger(LOGGER_FILE);
    logger << "Running '" << __func__ << "' test..." << std::endl;

    std::string refImgName = "roiID0003.tif";
    std::string refImgPath = esvm::path::refStillImagesPath + refImgName;
    ASSERT_LOG(bfs::is_regular_file(refImgPath), "Reference image employed for preprocessing test was not found");

    cv::Mat refImg = cv::imread(refImgPath, cv::IMREAD_GRAYSCALE);
    ASSERT_LOG(refImg.size() == cv::Size(96, 96), "Expected reference image should be of dimension 96x96");
    int refImgSide = 48;
    cv::resize(refImg, refImg, cv::Size(refImgSide, refImgSide), 0, 0, cv::INTER_CUBIC);

    cv::Size patchCount(3, 3);
    int nPatches = patchCount.area();
    std::vector<cv::Mat> refImgPatches = imSplitPatches(refImg, patchCount);
    ASSERT_LOG(refImgPatches.size() == nPatches, "Reference image should have been split to expected number of patches");
    for (size_t p = 0; p < nPatches; ++p)
        ASSERT_LOG(refImgPatches[p].size() == cv::Size(16,16), "Reference image should have been split to expected patches dimensions");

    std::vector<FeatureVector> hogPatches(nPatches);
    FeatureExtractorHOG hog(refImgPatches[0].size(), cv::Size(2, 2), cv::Size(2, 2), cv::Size(2, 2), 3);
    for (size_t p = 0; p < nPatches; ++p)
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
    size_t nFeatures = hogPatches[0].size();
    double *features = new double[nFeatures];
    for (size_t p = 0; p < nPatches; ++p)
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
    for (size_t p = 0; p < nPatches; ++p)
    {
        for (int y = 0; y < patchSide; y++)
            for (int x = 0; x < patchSide; x++)
                patchData[x + y * patchSide] = (double)refImgPatches[p].at<uchar>(x, y);
        HOG(patchData, hogParams, patchSize, features, 1);
        hogPatchesDbl2[p] = std::vector<double>(features, features + nFeatures);
        logger << refImgName << " hog588-impl-DBL2-patch" << std::to_string(p) << ": " << featuresToVectorString(hogPatchesDbl2[p]) << std::endl;
    }

    // access
    std::vector<FeatureVector> hogPatchesAccess(nPatches);
    for (size_t p = 0; p < nPatches; ++p)
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
    for (size_t p = 0; p < nPatches; ++p)
    {
        hogPatchesData[p] = hog.compute(refImgDataPatches[p]);
        logger << "roiID0003.txt hog588-impl-data-patch" << std::to_string(p) << ": " << featuresToVectorString(hogPatchesData[p]) << std::endl;
    }

    // employ 'text' + enforce 'double'
    std::vector<FeatureVector> hogPatchesDblData(nPatches);
    int patchRow = 0;
    int patchCol = 0;
    for (size_t p = 0; p < nPatches; ++p)
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
    for (size_t p = 0; p < nPatches; ++p)
    {
        hogPatchesDataPermute[p] = hog.compute(refImgDataPermutePatches[p]);
        logger << "roiID0003.txt hog588-impl-data-permute-patch" << std::to_string(p) << ": " << featuresToVectorString(hogPatchesDataPermute[p]) << std::endl;
    }

    // employ permuted 'text' + enforce 'double'
    std::vector<FeatureVector> hogPatchesDblDataPermute(nPatches);
    patchRow = 0;
    patchCol = 0;
    for (size_t p = 0; p < nPatches; ++p)
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

    #else/*TEST_IMAGE_PREPROCESSING*/
    return passThroughDisplayTestStatus(__func__, SKIPPED);
    #endif/*TEST_IMAGE_PREPROCESSING*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

int test_multiLevelVectors()
{
    #if TEST_MULTI_LEVEL_VECTORS
    logstream logger(LOGGER_FILE);
    logger << "Running '" << __func__ << "' test..." << std::endl;

    size_t vdim = 6;
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
    for (size_t f = 0; f < vdim; f++)
    {
        ASSERT_LOG(mv_index[0][f] == f + 1, "Multi-level vector assigned by index should have corresponding feature value");
        ASSERT_LOG(mv_index[1][f] == f + 7, "Multi-level vector assigned by index should have corresponding feature value");
    }

    // Not initialized 1D-vector of FV, added by push_back
    xstd::mvector<1, FeatureVector> mv_push;
    mv_push.push_back(v1);
    mv_push.push_back(v2);
    for (size_t f = 0; f < vdim; f++)
    {
        ASSERT_LOG(mv_push[0][f] == f + 1, "Multi-level vector assigned by index should have corresponding feature value");
        ASSERT_LOG(mv_push[1][f] == f + 7, "Multi-level vector assigned by index should have corresponding feature value");
    }

    // Pre-initialized 2D-vector of FV using single size
    size_t singleSize = 5;
    xstd::mvector<2, FeatureVector> mvv_singleSize(singleSize);
    ASSERT_LOG(mvv_singleSize.size() == singleSize, "Multi-level vector assigned with single size should have same dimension on each level");
    for (size_t L1 = 0; L1 < singleSize; L1++)
    {
        ASSERT_LOG(mvv_singleSize[L1].size() == singleSize, "Multi-level vector assigned with single size should have same dimension on each level");
        for (size_t L2 = 0; L2 < singleSize; L2++)
            ASSERT_LOG(mvv_singleSize[L1][L2].size() == 0, "FeatureVector as lowest level object of multi-level vector should not be initialized");
    }

    // Pre-initialized 1D-vector of FV using sizes per level, assigned by index
    size_t dims[2] = { 3, 2 };
    xstd::mvector<2, FeatureVector> mvv_index(dims);
    ASSERT_LOG(mvv_index.size() == dims[0], "Multi-level vector should be initialized with specified 1st level dimension");
    for (size_t L1 = 0; L1 < dims[0]; L1++)
        ASSERT_LOG(mvv_index[L1].size() == dims[1], "Each multi-level vector on 2nd level should be initialized with specified dimension");
    mvv_index[0][0] = v1;
    mvv_index[0][1] = v2;
    mvv_index[1][0] = v3;
    mvv_index[1][1] = v4;
    mvv_index[2][0] = v5;
    mvv_index[2][1] = v6;
    for (size_t L1 = 0; L1 < dims[0]; L1++)
        for (size_t L2 = 0; L2 < dims[1]; L2++)
        {
            ASSERT_LOG(mvv_index[L1][L2].size() == vdim, "Lowest level feature vector should have original dimension");
            for (size_t f = 0; f < vdim; f++)
            {
                double fval = (double)((L1 * dims[1] + L2) * vdim + f) + 1.0;
                ASSERT_LOG(mvv_index[L1][L2][f] == fval, "Multi-level vector assigned by index should have corresponding feature value");
            }
        }

    // Not initialized 2D-vector of FV, added by push_back of sub multi-level vectors
    xstd::mvector<2, FeatureVector> mvv_push;
    for (size_t L1 = 0; L1 < dims[0]; L1++)
        mvv_push.push_back(xstd::mvector<1, FeatureVector>());
    ASSERT_LOG(mvv_push.size() == dims[0], "Multi-level vector should be initialized with number of pushed vectors");
    for (size_t L1 = 0; L1 < dims[0]; L1++)
        ASSERT_LOG(mvv_push[L1].size() == 0, "Second level vector should still be empty");
    mvv_push[0].push_back(v1);
    mvv_push[0].push_back(v2);
    mvv_push[1].push_back(v3);
    mvv_push[1].push_back(v4);
    mvv_push[2].push_back(v5);
    mvv_push[2].push_back(v6);
    for (size_t L1 = 0; L1 < dims[0]; L1++)
    {
        ASSERT_LOG(mvv_push[L1].size() == dims[1], "Each multi-level vector on 2nd level should be initialized with number of pushed vectors");
        for (size_t L2 = 0; L2 < dims[1]; L2++)
        {
            ASSERT_LOG(mvv_push[L1][L2].size() == vdim, "Lowest level feature vector should have original dimension");
            for (size_t f = 0; f < vdim; f++)
            {
                double fval = (double)((L1 * dims[1] + L2) * vdim + f) + 1.0;
                ASSERT_LOG(mvv_push[L1][L2][f] == fval, "Multi-level vector assigned with push back should have corresponding feature value");
            }
        }
    }
    mvv_push[1].push_back(v1);
    ASSERT_LOG(mvv_push.size() == dims[0], "Top level dimension of multi-level vector shouldn't be affected by lower level push");
    ASSERT_LOG(mvv_push[0].size() == dims[1], "Other lower level index then vector that got another push shouldn't be affected in size");
    ASSERT_LOG(mvv_push[1].size() == dims[1]+1, "Lower level vector with additional pushed feature vector should be expanded by one");
    ASSERT_LOG(mvv_push[2].size() == dims[1], "Other lower level index then vector that got another push shouldn't be affected in size");

    #else/*TEST_MULTI_LEVEL_VECTORS*/
    return passThroughDisplayTestStatus(__func__, SKIPPED);
    #endif/*TEST_MULTI_LEVEL_VECTORS*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

int test_normalizationFunctions()
{
    #if TEST_NORMALIZATION
    logstream logger(LOGGER_FILE);
    logger << "Running '" << __func__ << "' test..." << std::endl;

    // testing values
    FeatureVector v1 { -1, 2, 3, 4, 5, 14 };
    FeatureVector v2 { 8, 4, 6, 0.5, 1, 5 };
    FeatureVector v1_norm01 = { 0.0, 0.2, 4.0/15.0, 1.0/3.0, 0.4, 1.0 };
    FeatureVector v2_min =  { 2, 0, 4, 0, 0, 5 };
    FeatureVector v2_max =  { 8, 8, 8, 4, 4, 8 };
    FeatureVector v2_norm = { 1, 0.5, 0.5, 0.125, 0.25, 0 };
    std::vector<FeatureVector> v = { v1, v2 };

    /* ---------------------
        normal operation
    --------------------- */

    ASSERT_LOG(normalize(MIN_MAX, 0.50, -1.0, 1.0) == 0.75,  "Value should have been normalized with min-max rule");
    ASSERT_LOG(normalize(MIN_MAX, 1.00, -1.0, 1.0) == 1.00,  "Value should have been normalized with min-max rule");
    ASSERT_LOG(normalize(MIN_MAX, 0.25, -1.0, 1.0) == 0.625, "Value should have been normalized with min-max rule");
    ASSERT_LOG(normalize(MIN_MAX, 1.00, -2.0, 0.5) == 1.20,  "Value should have been normalized with min-max rule");
    ASSERT_LOG(normalize(MIN_MAX, -3.0, -2.0, 0.5) == -0.4,  "Value should have been normalized with min-max rule");
    ASSERT_LOG(normalize(MIN_MAX, 1.00, -2.0, 0.5, true) == 1.0, "Value should be clipped from normalization with min-max rule");
    ASSERT_LOG(normalize(MIN_MAX, -3.0, -2.0, 0.5, true) == 0.0, "Value should be clipped from normalization with min-max rule");
    ASSERT_LOG(normalize(Z_SCORE, 1.0, 0.0, 1.0) == 2.0/3.0, "Value should have been normalized with z-score rule");
    ASSERT_LOG(normalize(Z_SCORE, 3.0, 0.0, 1.0) == 1.0,     "Value should have been normalized with z-score rule");
    ASSERT_LOG(normalize(Z_SCORE, -0.75, 0.0, 1.0) == 0.375, "Value should have been normalized with z-score rule");
    ASSERT_LOG(normalize(Z_SCORE, 0.0, 0.0, 1.0) == 0.5,     "Value should have been normalized with z-score rule");
    ASSERT_LOG(normalize(Z_SCORE, -12, 0.0, 1.0) == -1.5,    "Value should have been normalized with z-score rule");
    ASSERT_LOG(normalize(Z_SCORE, 6.0, 0.0, 1.0, true) == 1.0, "Value should be clipped from normalization with z-score rule");
    ASSERT_LOG(normalize(Z_SCORE, -12, 0.0, 1.0, true) == 0.0, "Value should be clipped from normalization with z-score rule");

    double min1 = -1, max1 = -1, min2 = -1, max2 = -1;
    int posMin1 = -1, posMax1 = -1, posMin2 = -1, posMax2 = -1;
    findNormParamsAcrossFeatures(MIN_MAX, v1, min1, max1, &posMin1, &posMax1);
    findNormParamsAcrossFeatures(MIN_MAX, v2, min2, max2, &posMin2, &posMax2);
    ASSERT_LOG(min1 == -1,   "Minimum value of vector should be assigned to variable by reference");
    ASSERT_LOG(max1 == 14,   "Maximum value of vector should be assigned to variable by reference");
    ASSERT_LOG(posMin1 == 0, "Index position of minimum value of vector should be assigned to variable by reference");
    ASSERT_LOG(posMax1 == 5, "Index position of maximum value of vector should be assigned to variable by reference");
    ASSERT_LOG(min2 == 0.5,  "Minimum value of vector should be assigned to variable by reference");
    ASSERT_LOG(max2 == 8,    "Maximum value of vector should be assigned to variable by reference");
    ASSERT_LOG(posMin2 == 3, "Index position of minimum value of vector should be assigned to variable by reference");
    ASSERT_LOG(posMax2 == 0, "Index position of maximum value of vector should be assigned to variable by reference");

    FeatureVector vmin, vmax;
    findNormParamsPerFeature(MIN_MAX, v, vmin, vmax);
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
    findNormParamsOverAll(MIN_MAX, v, minAll, maxAll);
    ASSERT_LOG(minAll == -1, "Minimum value of all features of whole list should be found");
    ASSERT_LOG(maxAll == 14, "Maximum value of all features of whole list should be found");

    FeatureVector normAll = normalizeOverAll(MIN_MAX, v1, -1, 14);     // min/max of v1 are -1,14, makes (max-min)=15
    for (size_t f = 0; f < normAll.size(); f++)
        ASSERT_LOG(normAll[f] == v1_norm01[f], "Feature should be normalized with specified min/max values");

    FeatureVector normAllMore = normalizeOverAll(MIN_MAX, v1, -1, 29); // using max == 29 makes (max-min)=30, 1/2 norm values
    for (size_t f = 0; f < normAllMore.size(); f++)
        ASSERT_LOG(normAllMore[f] == v1_norm01[f] / 2.0, "Feature normalization should be enforced with specified min/max values");

    FeatureVector normAllAuto = normalizeOverAll(MIN_MAX, v1);         // min/max not specified, find them
    for (size_t f = 0; f < normAllAuto.size(); f++)
        ASSERT_LOG(normAllAuto[f] == v1_norm01[f], "Feature should be normalized with min/max found within the specified vector");

    std::vector<double> scores = normalizeClassScores(MIN_MAX, v1);        // same as 'normalizeOverAll<MinMax>'
    for (size_t f = 0; f < scores.size(); f++)
        ASSERT_LOG(scores[f] == v1_norm01[f], "Score should be normalized with min/max of all scores");

    FeatureVector v2_normPerFeat = normalizePerFeature(MIN_MAX, v2, v2_min, v2_max);
    for (size_t f = 0; f < v2_normPerFeat.size(); f++)
        ASSERT_LOG(v2_normPerFeat[f] == v2_norm[f], "Feature should be normalized with corresponding min/max features");

    double s1pos = normalizeClassScoreToSimilarity(+1);
    double s1neg = normalizeClassScoreToSimilarity(-1);
    double s0mid = normalizeClassScoreToSimilarity(0);
    double sprob = normalizeClassScoreToSimilarity(0.5);
    ASSERT_LOG(s1pos == 1, "Positive class score should be normalized as maximum similarity");
    ASSERT_LOG(s1neg == 0, "Negative class score should be normalized as minimum similarity");
    ASSERT_LOG(s0mid == 0.5, "Indifferent class score should be normalized as middle similarity");
    ASSERT_LOG(sprob == 0.75, "Half-probable positive class score should be normalized as 3/4 simiarity");

    /* --------------------
        exception cases
    -------------------- */

    double dummyValue;
    FeatureVector vEmpty;
    try {
        normalize(MIN_MAX, 1.0, 1.0, -1.0);
        logger << "Minimum value greater than maximum value should have raised an exception" << std::endl;
        return passThroughDisplayTestStatus(__func__, -1);
    } catch (...) {}    // expceted exception
    try {
        normalize(Z_SCORE, 1.0, 1.0, 0.0);
        logger << "Zero value standard deviation should have raised an exception" << std::endl;
        return passThroughDisplayTestStatus(__func__, -2);
    } catch (...) {}    // expceted exception
    // Does not apply with switch passing by pointer to by reference
    /*
    try {
        findNormParamsAcrossFeatures(MIN_MAX, v1, nullptr, &dummyValue);
        logger << "Null reference for minimum value should have raised an exception" << std::endl;
        return passThroughDisplayTestStatus(__func__, -3);
    } catch (...) {}    // expceted exception
    try {
        findNormParamsAcrossFeatures(MIN_MAX, v1, &dummyValue, nullptr);
        logger << "Null reference for maximum value should have raised an exception" << std::endl;
        return passThroughDisplayTestStatus(__func__, -4);
    } catch (...) {}    // expceted exception
    */
    try {
        findNormParamsAcrossFeatures(MIN_MAX, vEmpty, dummyValue, dummyValue);
        logger << "Empty feature vector should have raised an exception" << std::endl;
        return passThroughDisplayTestStatus(__func__, -5);
    } catch (...) {}    // expceted exception
    // Does not apply with switch passing by pointer to by reference
    /*
    try {
        findNormParamsPerFeature(MIN_MAX, v, nullptr, &vEmpty);
        logger << "Null reference for minimum features should have raised an exception" << std::endl;
        return passThroughDisplayTestStatus(__func__, -6);
    } catch (...) {}    // expceted exception
    try {
        findNormParamsPerFeature(MIN_MAX, v, &vEmpty, nullptr);
        logger << "Null reference for maximum features should have raised an exception" << std::endl;
        return passThroughDisplayTestStatus(__func__, -7);
    } catch (...) {}    // expceted exception
    */
    try {
        normalizePerFeature(MIN_MAX, v1, vEmpty, v1);
        logger << "Inconsistent size for minimum features should have raised an exception" << std::endl;
        return passThroughDisplayTestStatus(__func__, -8);
    } catch (...) {}    // expceted exception
    try {
        normalizePerFeature(MIN_MAX, v1, v1, vEmpty);
        logger << "Inconsistent size for maximum features should have raised an exception" << std::endl;
        return passThroughDisplayTestStatus(__func__, -9);
    } catch (...) {}    // expceted exception

    #else/*TEST_NORMALIZATION*/
    return passThroughDisplayTestStatus(__func__, SKIPPED);
    #endif/*TEST_NORMALIZATION*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

int test_performanceEvaluationFunctions()
{
    #if TEST_PERF_EVAL_FUNCTIONS
    logstream logger(LOGGER_FILE);
    logger << "Running '" << __func__ << "' test..." << std::endl;

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
    double pAUC20_valid = 0.18875/0.20; // pAUC(20%) of values, lands perfectly on an existing FPR
    double pAUC35_valid = 0.33650/0.35; // pAUC(35%) of values, lands between existing FPRs, interpolation applied
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
    xstd::mvector<2, double> scores;
    xstd::mvector<2, int> groundTruths;
    std::vector<std::string> targets = { "TEST" };
    std::vector<double> targetScores{ 0.9, 0.85, 0.92, 0.89, 0.87, 0.63, 0.42, 0.56 };
    std::vector<int> targetOutputs{ 1, 1, 1, 1, 1, -1, -1, -1 };
    scores.push_back(targetScores);
    groundTruths.push_back(targetOutputs);
    logger << "Displaying results table from dummy classification scores:" << std::endl;
    eval_PerformanceClassificationSummary(targets, scores, groundTruths);

    #else/*TEST_PERF_EVAL_FUNCTIONS*/
    return passThroughDisplayTestStatus(__func__, SKIPPED);
    #endif/*TEST_PERF_EVAL_FUNCTIONS*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

int test_ESVM_BasicFunctionalities(void)
{
    #if TEST_ESVM_BASIC_FUNCTIONALITY
    logstream logger(LOGGER_FILE);
    logger << "Running '" << __func__ << "' test..." << std::endl;

    cv::namedWindow(WINDOW_NAME);

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
    for (int i = 0; i < NB_POSITIVE_IMAGES; ++i)
        mwPositiveSamples.Get(1, i + 1).Set(convertCvToMatlabMat(matPositiveSamples[i]));
    logger << "Converting negative training samples..." << std::endl;
    for (int i = 0; i < NB_NEGATIVE_IMAGES; ++i)
        mwNegativeSamples.Get(1, i + 1).Set(convertCvToMatlabMat(matNegativeSamples[i]));
    logger << "Converting probe testing samples..." << std::endl;
    for (int i = 0; i < NB_PROBE_IMAGES; ++i)
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
    }
    catch (const mwException& e)
    {
        logger << e.what() << std::endl;
        return passThroughDisplayTestStatus(__func__, -2);
    }
    catch (...)
    {
        logger << "Unexpected error thrown" << std::endl;
        return passThroughDisplayTestStatus(__func__, -3);
    }

    #else/*TEST_ESVM_BASIC_FUNCTIONALITY*/
    return passThroughDisplayTestStatus(__func__, OBSOLETE);   // DISABLE - USING OBSOLETE MATLAB PROCEDURE
    #endif/*TEST_ESVM_BASIC_FUNCTIONALITY*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

int test_ESVM_BasicClassification(void)
{
    #if TEST_ESVM_BASIC_CLASSIFICATION
    logstream logger(LOGGER_FILE);
    logger << "Running '" << __func__ << "' test..." << std::endl;

    // ------------------------------------------------------------------------------------------------------------------------
    // training ESVM with samples (XOR)
    // ------------------------------------------------------------------------------------------------------------------------
    logger << "Training Exemplar-SVM with XOR samples..." << std::endl;
    std::vector<FeatureVector> positives(20);
    std::vector<FeatureVector> negatives(20);
    std::srand(std::time(0));
    for (int i = 0; i < 10; ++i)
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

    #else/*TEST_ESVM_BASIC_CLASSIFICATION*/
    return passThroughDisplayTestStatus(__func__, SKIPPED);
    #endif/*TEST_ESVM_BASIC_CLASSIFICATION*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

// // Test ESVM especially in the case where RSM are enabled
// int test_ESVM_BasicTrainTestRSM(void)
// {
//     #if TEST_ESVM_BASIC_TRAIN_TEST_RSM
//     logstream logger(LOGGER_FILE);
//     logger << "Running '" << __func__ << "' test..." << std::endl;
//     #if ESVM_RANDOM_SUBSPACE_METHOD <= 0
//     logger << "Warning: '" << __func__ << "' disabled because 'ESVM_RANDOM_SUBSPACE_METHOD' is not enabled..." << std::endl;
//     #else/*ESVM_RANDOM_SUBSPACE_METHOD > 0*/

//     size_t nSamples = 10;
//     size_t nRSM = ESVM_RANDOM_SUBSPACE_METHOD;
//     size_t nPatches = 2;
//     size_t dimsSamplesRaw[2] { nSamples, nPatches };
//     size_t dimsSamplesRSM[2] { nSamples, nRSM };
//     xstd::mvector<2, FeatureVector> trainSamples(dimsSamplesRaw);

//     #endif/*ESVM_RANDOM_SUBSPACE_METHOD <= 0*/
//     #else/*TEST_ESVM_BASIC_TRAIN_TEST_RSM*/
//     return passThroughDisplayTestStatus(__func__, SKIPPED);
//     #endif/*TEST_ESVM_BASIC_TRAIN_TEST_RSM*/
//     return passThroughDisplayTestStatus(__func__, PASSED);
// }

// Tests LIBSVM format sample file reading functionality of ESVM (index and value parsing)
int test_ESVM_ReadSampleFile_libsvm()
{
    #if TEST_ESVM_READ_SAMPLES_FILE_PARSER_LIBSVM
    logstream logger(LOGGER_FILE);
    logger << "Running '" << __func__ << "' test..." << std::endl;

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
    validSampleFile4 << std::to_string(ESVM_POSITIVE_CLASS) << " 1:90.111 2:80.222 3:70.333 4:60.444 5:50.555" << std::endl;;      // no '-1:?' ok
    validSampleFile4 << std::to_string(ESVM_NEGATIVE_CLASS) << " 1:10.999 2:20.888 3:30.777 4:40.666 5:50.000" << std::endl;;      // no '-1:?' ok
    wrongSampleFile1 << std::to_string(ESVM_NEGATIVE_CLASS) << " 1:10.999 2:20.111 5:30.555 4:40.333 3:50.777 -1:0" << std::endl;  // 5->4->3
    wrongSampleFile2 << std::to_string(ESVM_NEGATIVE_CLASS) << " 1:10.999 2:20.111 3:30.555 5:40.333 5:50.777 -1:0" << std::endl;  // 3->5->5
    wrongSampleFile3 << std::to_string(ESVM_NEGATIVE_CLASS) << " 1:10.999 2:20.111 3:30.555 4:40.333 5:50.777 -1:0" << std::endl;  // != size
    wrongSampleFile3 << std::to_string(ESVM_NEGATIVE_CLASS) << " 1:10.999 2:20.111 3:30.555 -1:0" << std::endl;                    // != size
    wrongSampleFile4 << std::to_string(ESVM_NEGATIVE_CLASS) << " 10.999 20.111 30.555 40.666 50.777 60.888 70.999" << std::endl;   // missing ':'
    wrongSampleFile5 << "1:10.999 2:20.111 3:30.555 4:40.666 5:50.777 6:60.888 7:70.999" << std::endl;              // missing target output class

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
    std::vector<FeatureVector> samples;
    std::vector<int> targetOutputs;
    try
    {
        // test valid normal indexed samples (exception not expected)
        ESVM::readSampleDataFile(validSampleFileName1, samples, targetOutputs);
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
               << "Exception: [" << ex.what() << "]" << std::endl;
        bfs::remove_all(testDir);
        return passThroughDisplayTestStatus(__func__, -1);
    }
    try
    {
        // test valid sparse indexed samples (exception not expected)
        ESVM::readSampleDataFile(validSampleFileName2, samples, targetOutputs);
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
               << "Exception: [" << ex.what() << "]" << std::endl;
        bfs::remove_all(testDir);
        return passThroughDisplayTestStatus(__func__, -2);
    }
    try
    {
        // test valid limited final index (-1) samples (exception not expected)
        ESVM::readSampleDataFile(validSampleFileName3, samples, targetOutputs);
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
               << "Exception: [" << ex.what() << "]" << std::endl;
        bfs::remove_all(testDir);
        return passThroughDisplayTestStatus(__func__, -3);
    }
    try
    {
        // test valid omitted final index (-1) samples (exception not expected)
        ESVM::readSampleDataFile(validSampleFileName4, samples, targetOutputs);
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
               << "Exception: [" << ex.what() << "]" << std::endl;
        bfs::remove_all(testDir);
        return passThroughDisplayTestStatus(__func__, -4);
    }
    try
    {
        // test wrong non ascending index samples (exception expected)
        ESVM::readSampleDataFile(wrongSampleFileName1, samples, targetOutputs);
        logger << "Error: Indexes not specified in ascending order should have raised an exception." << std::endl;
        bfs::remove_all(testDir);
        return passThroughDisplayTestStatus(__func__, -5);
    }
    catch (...) {}
    try
    {
        // test wrong repeated index samples (exception expected)
        ESVM::readSampleDataFile(wrongSampleFileName2, samples, targetOutputs);
        logger << "Error: Repeating indexes should have raised an exception." << std::endl;
        bfs::remove_all(testDir);
        return passThroughDisplayTestStatus(__func__, -6);
    }
    catch (...) {}
    try
    {
        // test wrong non matching size samples (exception expected)
        ESVM::readSampleDataFile(wrongSampleFileName3, samples, targetOutputs);
        logger << "Error: Non matching sample sizes should have raised an exception." << std::endl;
        bfs::remove_all(testDir);
        return passThroughDisplayTestStatus(__func__, -7);
    }
    catch (...) {}
    try
    {
        // test wrong missing index:value separator (exception expected)
        ESVM::readSampleDataFile(wrongSampleFileName4, samples, targetOutputs);
        logger << "Error: Not found 'index:value' seperator should have raised an exception." << std::endl;
        bfs::remove_all(testDir);
        return passThroughDisplayTestStatus(__func__, -8);
    }
    catch (...) {}
    try
    {
        // test wrong missing target output class (exception expected)
        ESVM::readSampleDataFile(wrongSampleFileName5, samples, targetOutputs);
        logger << "Error: Missing target output class value should have raised an exception." << std::endl;
        bfs::remove_all(testDir);
        return passThroughDisplayTestStatus(__func__, -9);
    }
    catch (...) {}

    // delete test directory and sample files
    bfs::remove_all(testDir);

    #else/*TEST_ESVM_READ_SAMPLES_FILE_PARSER_LIBSVM*/
    return passThroughDisplayTestStatus(__func__, SKIPPED);
    #endif/*TEST_ESVM_READ_SAMPLES_FILE_PARSER_LIBSVM*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

// Tests BINARY sample file reading functionality of ESVM
int test_ESVM_ReadSampleFile_binary()
{
    #if TEST_ESVM_READ_SAMPLES_FILE_PARSER_BINARY
    logstream logger(LOGGER_FILE);
    logger << "Running '" << __func__ << "' test..." << std::endl;

    // create test sample files inside test directory
    std::string testDir = "test_sample-read-binary-file/";
    bfs::create_directory(testDir);
    std::string validSampleFileName1 = testDir + "test_valid-samples1.data";  // for testing valid BINARY formatted file
    std::string wrongSampleFileName1 = testDir + "test_wrong-samples1.data";  // for testing missing header
    std::string wrongSampleFileName2 = testDir + "test_wrong-samples2.data";  // for testing invalid header
    std::string wrongSampleFileName3 = testDir + "test_wrong-samples3.data";  // for testing invalid number of samples
    std::string wrongSampleFileName4 = testDir + "test_wrong-samples4.data";  // for testing invalid number of features
    std::string wrongSampleFileName5 = testDir + "test_wrong-samples5.data";  // for testing missing target output class value

    // fill test sample files
    FeatureVector v1 = { 0.999, 1.888, 2.777, 3.666, 4.555 }, v2 = { 5.444, 6.333, 7.222, 8.111, 9.000 };
    std::vector<FeatureVector> validSamples = { v1, v2 };
    std::vector<int> validTargetOutputs = { ESVM_POSITIVE_CLASS, ESVM_NEGATIVE_CLASS };
    ESVM::writeSampleDataFile(validSampleFileName1, validSamples, validTargetOutputs, BINARY);

    /*=====================
    TODO OTHER / MORE EXTENSIVE TESTS
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
        ESVM::readSampleDataFile(validSampleFileName1, readSamples, readTargetOutputs, BINARY);
        ASSERT_LOG(readSamples.size() == validSamples.size(), "Number of samples read from BINARY file should match original");
        ASSERT_LOG(readSamples[0].size() == validSamples[0].size(), "Number of features read from BINARY file should match original");
        ASSERT_LOG(readTargetOutputs.size() == validTargetOutputs.size(), "Number of target outputs read from BINARY file should match original");
        ASSERT_LOG(readSamples[0][0] == validSamples[0][0], "Sample feature value from BINARY file should match the original one");
        ASSERT_LOG(readSamples[0][1] == validSamples[0][1], "Sample feature value from BINARY file should match the original one");
        ASSERT_LOG(readSamples[0][2] == validSamples[0][2], "Sample feature value from BINARY file should match the original one");
        ASSERT_LOG(readSamples[0][3] == validSamples[0][3], "Sample feature value from BINARY file should match the original one");
        ASSERT_LOG(readSamples[0][4] == validSamples[0][4], "Sample feature value from BINARY file should match the original one");
        ASSERT_LOG(readSamples[1][0] == validSamples[1][0], "Sample feature value from BINARY file should match the original one");
        ASSERT_LOG(readSamples[1][1] == validSamples[1][1], "Sample feature value from BINARY file should match the original one");
        ASSERT_LOG(readSamples[1][2] == validSamples[1][2], "Sample feature value from BINARY file should match the original one");
        ASSERT_LOG(readSamples[1][3] == validSamples[1][3], "Sample feature value from BINARY file should match the original one");
        ASSERT_LOG(readSamples[1][4] == validSamples[1][4], "Sample feature value from BINARY file should match the original one");
        ASSERT_LOG(readTargetOutputs[0] == validTargetOutputs[0], "Target output value from BINARY file should match the original one");
        ASSERT_LOG(readTargetOutputs[1] == validTargetOutputs[1], "Target output value from BINARY file should match the original one");
    }
    catch (std::exception& ex)
    {
        logger << "Error: Valid BINARY samples file reading should not have generated an exception." << std::endl
               << "Exception: [" << ex.what() << "]" << std::endl;
        bfs::remove_all(testDir);
        return passThroughDisplayTestStatus(__func__, -1);
    }
    try
    {
        // test wrong reading file format
        ESVM::readSampleDataFile(wrongSampleFileName5, readSamples, readTargetOutputs, LIBSVM);
        logger << "Error: Reading BINARY formatted file as LIBSVM format should result in parsing failure." << std::endl;
        return passThroughDisplayTestStatus(__func__, -2);
    }
    catch (...) {}

    bfs::remove_all(testDir);

    #else/*TEST_ESVM_READ_SAMPLES_FILE_PARSER_BINARY*/
    return passThroughDisplayTestStatus(__func__, SKIPPED);
    #endif/*TEST_ESVM_READ_SAMPLES_FILE_PARSER_BINARY*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

// Validation of identical sample features from BINARY/LIBSVM formatted files
int test_ESVM_ReadSampleFile_compare()
{
    #if TEST_ESVM_READ_SAMPLES_FILE_FORMAT_COMPARE
    logstream logger(LOGGER_FILE);
    logger << "Running '" << __func__ << "' test..." << std::endl;

    std::vector<std::string> positivesID = { "ID0003", "ID0005", "ID0006", "ID0010", "ID0024" };
    size_t nPatches = 9;
    size_t nPositives = positivesID.size();

    for (size_t p = 0; p < nPatches; ++p)
    {
        std::vector<FeatureVector> samples_libsvm, samples_binary;
        std::vector<int> targetOutputs_libsvm, targetOutputs_binary;
        std::string strPatch = std::to_string(p);

        // load negatives samples and target ouputs from BINARY/LIBSVM formatted files
        std::string negativeSamplesPatchFile = negativeSamplesDir + "negatives-hog-patch" + strPatch;
        std::string currentNegativePatch = " (negatives, patch " + strPatch + ")";
        logger << "Loading samples files (BINARY/LIBSVM) for comparison" << currentNegativePatch << "..." << std::endl;
        ESVM::readSampleDataFile(negativeSamplesPatchFile + ".data", samples_libsvm, targetOutputs_libsvm, LIBSVM);
        ESVM::readSampleDataFile(negativeSamplesPatchFile + ".bin",  samples_binary, targetOutputs_binary, BINARY);

        // compare negative patch samples and target outputs
        logger << "Comparing samples files (BINARY/LIBSVM) data" << currentNegativePatch << "..." << std::endl;
        size_t nNegatives = samples_libsvm.size();
        size_t nNegativeFeatures = samples_libsvm[0].size();
        ASSERT_LOG(nNegatives == samples_binary.size(), "Inconsistent LIBSVM and BINARY samples count" + currentNegativePatch);
        ASSERT_LOG(nNegatives == targetOutputs_libsvm.size(), "Inconsistent LIBSVM target outputs count" + currentNegativePatch);
        ASSERT_LOG(nNegatives == targetOutputs_binary.size(), "Inconsistent BINARY target outputs count" + currentNegativePatch);
        for (size_t neg = 0; neg < nNegatives; ++neg)
        {
            currentNegativePatch = " (negative " + std::to_string(neg) + ", patch " + strPatch + ")";
            ASSERT_LOG(nNegativeFeatures == samples_binary[neg].size(), "Inconsistent LIBSVM and BINARY features count" + currentNegativePatch);
            ASSERT_LOG(targetOutputs_libsvm[neg] == targetOutputs_binary[neg], "Target outputs should match" + currentNegativePatch);
            for (size_t f = 0; f < nNegativeFeatures; f++)
            {
                currentNegativePatch = " (negative " + std::to_string(neg) + ", patch " + strPatch + ", feature " + std::to_string(f) + ")";
                ASSERT_LOG(samples_libsvm[neg][f] == samples_binary[neg][f], "Sample features should match" + currentNegativePatch);
            }
        }

        for (size_t pos = 0; pos < nPositives; ++pos)
        {
            // load probe samples and target ouputs from BINARY/LIBSVM formatted files
            std::string probeSamplesPatchFile = testingSamplesDir + positivesID[pos] + "-probes-hog-patch" + strPatch;
            std::string currentProbePatch = " (positive " + positivesID[pos] + ", probes, patch " + strPatch + ")";
            logger << "Loading samples files (BINARY/LIBSVM) for comparison" << currentProbePatch << "..." << std::endl;
            ESVM::readSampleDataFile(probeSamplesPatchFile + ".data", samples_libsvm, targetOutputs_libsvm, LIBSVM);
            ESVM::readSampleDataFile(probeSamplesPatchFile + ".bin",  samples_binary, targetOutputs_binary, BINARY);

            // compare probe patch samples and target outputs
            logger << "Comparing samples files (BINARY/LIBSVM) data" << currentProbePatch << "..." << std::endl;
            size_t nProbes = samples_libsvm.size();
            size_t nProbeFeatures = samples_libsvm[0].size();
            ASSERT_LOG(nProbeFeatures == nNegativeFeatures, "Inconsistent negative and probe sample feature count" + currentProbePatch);
            ASSERT_LOG(nProbes == samples_binary.size(), "Inconsistent LIBSVM and BINARY samples count" + currentProbePatch);
            ASSERT_LOG(nProbes == targetOutputs_libsvm.size(), "Inconsistent LIBSVM target outputs count" + currentProbePatch);
            ASSERT_LOG(nProbes == targetOutputs_binary.size(), "Inconsistent BINARY target outputs count" + currentProbePatch);
            for (size_t prb = 0; prb < nProbes; ++prb)
            {
                currentProbePatch = " (positive " + positivesID[pos] + ", probe " + std::to_string(prb) + ", patch " + strPatch + ")";
                ASSERT_LOG(nProbeFeatures == samples_binary[prb].size(), "Inconsistent LIBSVM and BINARY features count" + currentProbePatch);
                ASSERT_LOG(targetOutputs_libsvm[prb] == targetOutputs_binary[prb], "Target outputs should match" + currentProbePatch);
                for (size_t f = 0; f < nProbeFeatures; f++)
                {
                    currentProbePatch = " (positive " + positivesID[pos] + ", probe " + std::to_string(prb) +
                                        ", patch " + strPatch + ", feature " + std::to_string(f) + ", [libsvm: " +
                                        std::to_string(samples_libsvm[prb][f]) + " != binary: " + std::to_string(samples_binary[prb][f]) + "])";
                    ASSERT_LOG(samples_libsvm[prb][f] == samples_binary[prb][f], "Sample features should match" + currentProbePatch);
                }
            }
        }
    }

    #else/*TEST_ESVM_READ_SAMPLES_FILE_FORMAT_COMPARE*/
    return passThroughDisplayTestStatus(__func__, SKIPPED);
    #endif/*TEST_ESVM_READ_SAMPLES_FILE_FORMAT_COMPARE*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

int test_ESVM_ReadSampleFile_timing(size_t nSamples, size_t nFeatures)
{
    #if TEST_ESVM_READ_SAMPLES_FILE_TIMING
    logstream logger(LOGGER_FILE);
    logger << "Running '" << __func__ << "' test..." << std::endl;

    ASSERT_LOG(nSamples > 0, "Number of samples must be greater than zero");
    ASSERT_LOG(nFeatures > 0, "Number of features must be greater than zero");

    // generate test samples file
    logger << "Generating dummy test samples file for timing evaluation..." << std::endl;
    std::string timingSampleFileName_libsvm = "test_timing-read-samples.data";
    std::string timingSampleFileName_binary = "test_timing-read-samples.bin";
    ASSERT_LOG(generateDummySampleFile_libsvm(timingSampleFileName_libsvm, nSamples, nFeatures), "Failed to generate dummy LIBSVM sample file");
    ASSERT_LOG(generateDummySampleFile_binary(timingSampleFileName_binary, nSamples, nFeatures), "Failed to generate dummy BINARY sample file");

    try
    {
        // start reading to evaluate timing
        std::vector<FeatureVector> samples;
        std::vector<int> targetOutputs;
        TP t0 = getTimeNowPrecise();
        ESVM::readSampleDataFile(timingSampleFileName_libsvm, samples, targetOutputs, LIBSVM);
        double dt = getDeltaTimePrecise(t0, MILLISECONDS);
        logger << "Elapsed time to read file with " << nSamples << " samples of " << nFeatures << " features (LIBSVM): "
               << std::setprecision(12) << dt << "ms" << std::endl;
        TP t1 = getTimeNowPrecise();
        ESVM::readSampleDataFile(timingSampleFileName_binary, samples, targetOutputs, BINARY);
        dt = getDeltaTimePrecise(t1, MILLISECONDS);
        logger << "Elapsed time to read file with " << nSamples << " samples of " << nFeatures << " features (BINARY): "
               << std::setprecision(12) << dt << "ms" << std::endl;
        bfs::remove(timingSampleFileName_libsvm);
        bfs::remove(timingSampleFileName_binary);
    }
    catch (std::exception& ex)
    {
        bfs::remove(timingSampleFileName_libsvm);
        bfs::remove(timingSampleFileName_binary);
        return passThroughDisplayTestStatus(__func__, -1);
    }

    #else/*TEST_ESVM_READ_SAMPLES_FILE_TIMING*/
    return passThroughDisplayTestStatus(__func__, SKIPPED);
    #endif/*TEST_ESVM_READ_SAMPLES_FILE_TIMING*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

int test_ESVM_WriteSampleFile_timing(size_t nSamples, size_t nFeatures)
{
    #if TEST_ESVM_WRITE_SAMPLES_FILE_TIMING
    logstream logger(LOGGER_FILE);
    logger << "Running '" << __func__ << "' test..." << std::endl;

    ASSERT_LOG(nSamples > 0, "Number of samples must be greater than zero");
    ASSERT_LOG(nFeatures > 0, "Number of features must be greater than zero");

    std::string timingSampleFileName_libsvm = "test_timing-write-samples.data";
    std::string timingSampleFileName_binary = "test_timing-write-samples.bin";

    try
    {
        // generate dummy data
        std::vector<FeatureVector> samples;
        std::vector<int> targetOutputs;
        logger << "Generating dummy test samples file for timing evaluation..." << std::endl;
        generateDummySamples(samples, targetOutputs, nSamples, nFeatures);

        // start writing to evaluate timing
        TP t0 = getTimeNowPrecise();
        ESVM::writeSampleDataFile(timingSampleFileName_libsvm, samples, targetOutputs, LIBSVM);
        double dt = getDeltaTimePrecise(t0, MILLISECONDS);
        logger << "Elapsed time to write file with " << nSamples << " samples of " << nFeatures << " features (LIBSVM): "
               << std::setprecision(12) << dt << "ms" << std::endl;
        TP t1 = getTimeNowPrecise();
        ESVM::writeSampleDataFile(timingSampleFileName_binary, samples, targetOutputs, BINARY);
        dt = getDeltaTimePrecise(t1, MILLISECONDS);
        logger << "Elapsed time to write file with " << nSamples << " samples of " << nFeatures << " features (BINARY): "
               << std::setprecision(12) << dt << "ms" << std::endl;
        bfs::remove(timingSampleFileName_libsvm);
        bfs::remove(timingSampleFileName_binary);
    }
    catch (std::exception& ex)
    {
        bfs::remove(timingSampleFileName_libsvm);
        bfs::remove(timingSampleFileName_binary);
        return passThroughDisplayTestStatus(__func__, -1);
    }

    #else/*TEST_ESVM_WRITE_SAMPLES_FILE_TIMING*/
    return passThroughDisplayTestStatus(__func__, SKIPPED);
    #endif/*TEST_ESVM_WRITE_SAMPLES_FILE_TIMING*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

// Test functionality of BINARY model file saving/loading and parsing of parameters allowing valid use afterwards
int test_ESVM_SaveLoadModelFile_libsvm()
{
    #if TEST_ESVM_SAVE_LOAD_MODEL_FILE_PARSER_LIBSVM
    logstream logger(LOGGER_FILE);
    logger << "Running '" << __func__ << "' test..." << std::endl;

    // create test model files inside test directory
    std::string testDir = "test_model-read-libsvm-file/";
    bfs::create_directory(testDir);
    std::string validModelFileName = testDir + "test_valid-model-libsvm.model";
    svm_model* validModel;
    FeatureVector validSample{ 0.55, 0.70, 0.22 };

    // check for generated model files
    try
    {
        logger << "Generating dummy test model file (LIBSVM) for functionality evaluation..." << std::endl;
        validModel = buildDummyExemplarSvmModel();
        ASSERT_LOG(svm_save_model(validModelFileName.c_str(), validModel) == 0, "Failed to create dummy model file: '" + validModelFileName + "'");
        ASSERT_LOG(bfs::is_regular_file(validModelFileName), "Couldn't find dummy model file: '" + validModelFileName + "'");
    }
    catch (std::exception& ex)
    {
        logger << "Model pre-generation for following ESVM tests should not have raised an exception." << std::endl
               << "Exception: [" << ex.what() << "]" << std::endl;
        bfs::remove_all(testDir);
        return passThroughDisplayTestStatus(__func__, -1);
    }

    // test file loading
    try
    {
        ESVM esvm;
        esvm.loadModelFile(validModelFileName, LIBSVM, "TEST");
        ASSERT_LOG(esvm.isModelTrained(), "Model should be trained after loading LIBSVM formatted model file");
        ASSERT_LOG(esvm.targetID == "TEST", "Target ID should have been properly set from model file loading function");
        esvm.predict(validSample);  // call test to ensure file loading provided a working model
    }
    catch (std::exception& ex)
    {
        logger << "Valid LIBSVM formatted model file should not have raised an exception." << std::endl
               << "Exception: [" << ex.what() << "]" << std::endl;
        bfs::remove_all(testDir);
        return passThroughDisplayTestStatus(__func__, -2);
    }

    ESVM::destroyModel(&validModel);
    bfs::remove_all(testDir);

    #else/*TEST_ESVM_SAVE_LOAD_MODEL_FILE_PARSER*/
    return passThroughDisplayTestStatus(__func__, SKIPPED);
    #endif/*TEST_ESVM_SAVE_LOAD_MODEL_FILE_PARSER*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

// Test functionality of LIBSVM model file saving/loading and parsing of parameters allowing valid use afterwards
int test_ESVM_SaveLoadModelFile_binary()
{
    #if TEST_ESVM_SAVE_LOAD_MODEL_FILE_PARSER_BINARY
    logstream logger(LOGGER_FILE);
    logger << "Running '" << __func__ << "' test..." << std::endl;

    // create test model files inside test directory
    std::string testDir = "test_model-read-binary-file/";
    bfs::create_directory(testDir);
    std::string validModelFileName = testDir + "test_valid-model-binary.model";
    std::string wrongModelFileName = testDir + "test_wrong-model-binary.model";
    svm_model* validModel;
    FeatureVector validSample({ 0.55, 0.70, 0.22 });

    try
    {
        logger << "Generating dummy test model file (BINARY) for functionality evaluation..." << std::endl;
        validModel = buildDummyExemplarSvmModel();
        ESVM esvmValid(validModel, "TEST-VALID");
        ASSERT_LOG(esvmValid.saveModelFile(validModelFileName, BINARY),
                   "Valid BINARY pre-trained model file loading should not have returned a failure status");
    }
    catch (std::exception& ex)
    {
        logger << "Valid BINARY pre-trained model file loading should not have raised an exception." << std::endl
               << "Exception: [" << ex.what() << "]" << std::endl;
        bfs::remove_all(testDir);
        return passThroughDisplayTestStatus(__func__, -1);
    }
    ESVM::destroyModel(&validModel);

    // ensure that trying to save the model from not initialized (not trained) ESVM fails
    try
    {
        ESVM esvmWrong;
        ASSERT_LOG(!esvmWrong.isModelTrained(), "ESVM should not be trained to test this functionality");
        ASSERT_LOG(!esvmWrong.saveModelFile(wrongModelFileName, BINARY),
                   "Invalid BINARY model file saving from untrained ESVM should not have returned a success status");
    }
    catch (std::exception& ex)
    {
        logger << "Invalid BINARY model file saving from untrained ESVM should not have raised an exception." << std::endl
               << "Exception: [" << ex.what() << "]" << std::endl;
        bfs::remove_all(testDir);
        return passThroughDisplayTestStatus(__func__, -2);
    }

    ESVM esvmLoaded;
    try
    {
        esvmLoaded.loadModelFile(validModelFileName, LIBSVM);
        logger << "Trying to load a BINARY model file in LIBSVM format should have raised an exception" << std::endl;
        bfs::remove_all(testDir);
        return passThroughDisplayTestStatus(__func__, -3);
    }
    catch (...) {}  // expected exception

    try
    {
        ASSERT_LOG(esvmLoaded.loadModelFile(validModelFileName, BINARY), "Loading valid BINARY model file should have returned a success");
        ASSERT_LOG(esvmLoaded.targetID == validModelFileName, "Target ID should equal file name when not specified upon model file loading");
        ASSERT_LOG(esvmLoaded.isModelTrained(), "Model should be trained after loading BINARY formatted model file");
        ASSERT_LOG(esvmLoaded.loadModelFile(validModelFileName, BINARY, "TEST-LOAD"), "Loading valid BINARY model file should return successfully");
        ASSERT_LOG(esvmLoaded.targetID == "TEST-LOAD", "Target ID should have been set to specified value upon model file loading");
        ASSERT_LOG(esvmLoaded.isModelTrained(), "Model should be trained after loading BINARY formatted model file");
    }
    catch (std::exception& ex)
    {
        bfs::remove_all(testDir);
        return passThroughDisplayTestStatus(__func__, -4);
    }

    try
    {
        esvmLoaded.predict(validSample);  // call test to ensure file loading provided a working model
    }
    catch (std::exception& ex)
    {
        logger << "Valid BINARY formatted model file should not have raised an exception." << std::endl
               << "Exception: [" << ex.what() << "]" << std::endl;
        bfs::remove_all(testDir);
        return passThroughDisplayTestStatus(__func__, -5);
    }

    bfs::remove_all(testDir);

    #else/*TEST_ESVM_SAVE_LOAD_MODEL_FILE_PARSER*/
    return passThroughDisplayTestStatus(__func__, SKIPPED);
    #endif/*TEST_ESVM_SAVE_LOAD_MODEL_FILE_PARSER*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

// Test functionality of model file saving/loading from (LIBSVM/binary,pre-trained/from samples) format comparison
int test_ESVM_SaveLoadModelFile_compare()
{
    #if TEST_ESVM_SAVE_LOAD_MODEL_FILE_FORMAT_COMPARE
    logstream logger(LOGGER_FILE);
    logger << "Running '" << __func__ << "' test..." << std::endl;

    // create test model files inside test directory and create testing data
    std::string testDir = "test_model-read-compare-file/";
    bfs::create_directory(testDir);
    std::string validModelFileName_libsvm = testDir + "test_valid-model-libsvm.model";
    std::string validModelFileName_binary = testDir + "test_valid-model-binary.model";
    FeatureVector probe{ 0.45, 0.20, 0.75, 0.05, 0.80 };
    std::vector<FeatureVector> trainingSamples
    {
        FeatureVector{ 0.50, 0.25, 0.75, 0.10, 0.90 },
        FeatureVector{ 0.10, 0.20, 0.75, 0.10, 0.10 },
        FeatureVector{ 0.20, 0.20, 0.75, 0.20, 0.80 },
        FeatureVector{ 0.25, 0.30, 0.75, 0.20, 0.70 },
        FeatureVector{ 0.15, 0.30, 0.75, 0.10, 0.20 }
    };
    std::vector<int> trainingLabels{ ESVM_POSITIVE_CLASS, ESVM_NEGATIVE_CLASS, ESVM_NEGATIVE_CLASS, ESVM_NEGATIVE_CLASS, ESVM_NEGATIVE_CLASS };

    try
    {
        // train a model from testing data, save to pre-trained model files
        ESVM esvmRef(trainingSamples, trainingLabels);
        double scoreRef = esvmRef.predict(probe);
        esvmRef.saveModelFile(validModelFileName_libsvm, LIBSVM);
        esvmRef.saveModelFile(validModelFileName_binary, BINARY);
        ASSERT_LOG(!doubleAlmostEquals(scoreRef, 0.0),
                   "Reference score should be different than zero to ensure validation of model file loading/saving procedures");

        // load files and compare results
        ESVM esvmLoad_libsvm, esvmLoad_binary;
        esvmLoad_libsvm.loadModelFile(validModelFileName_libsvm, LIBSVM);
        double score_libsvm = esvmLoad_libsvm.predict(probe);
        ASSERT_LOG(doubleAlmostEquals(scoreRef, score_libsvm, 0.000001),
                   "Loaded LIBSVM format model file should result in same score as reference model trained from samples (scoreRef: " +
                   std::to_string(scoreRef) + ", score_libsvm: " + std::to_string(score_libsvm) + ")");
        esvmLoad_binary.loadModelFile(validModelFileName_binary, BINARY);
        double score_binary = esvmLoad_binary.predict(probe);
        ASSERT_LOG(doubleAlmostEquals(scoreRef, score_binary, 0.000001),
                   "Loaded BINARY format model file should result in same score as reference model trained from samples (scoreRef: " +
                   std::to_string(scoreRef) + ", score_libsvm: " + std::to_string(score_binary) + ")");
        // re-load already trained model with swapped formats and compare results
        esvmLoad_libsvm.loadModelFile(validModelFileName_binary, BINARY);
        double score_libsvm_swap = esvmLoad_libsvm.predict(probe);
        ASSERT_LOG(doubleAlmostEquals(scoreRef, score_libsvm_swap, 0.000001),
                   "Re-loaded model file from different BINARY format should still result in same reference score (scoreRef: " +
                   std::to_string(scoreRef) + ", score_libsvm_swap: " + std::to_string(score_libsvm_swap) + ")");
        esvmLoad_binary.loadModelFile(validModelFileName_libsvm, LIBSVM);
        double score_binary_swap = esvmLoad_binary.predict(probe);
        ASSERT_LOG(doubleAlmostEquals(scoreRef, score_binary_swap, 0.000001),
                   "Re-loaded model file from different LIBSVM format should still result in same reference score (scoreRef: " +
                   std::to_string(scoreRef) + ", score_binary_swap: " + std::to_string(score_binary_swap) + ")");
    }
    catch (std::exception& ex)
    {
        logstream logger(LOGGER_FILE);
        logger << "Valid test procedures should not have raised an exception." << std::endl
               << "Exception: [" << ex.what() << "]" << std::endl;
        bfs::remove_all(testDir);
        return passThroughDisplayTestStatus(__func__, -1);
    }

    bfs::remove_all(testDir);

    #else/*TEST_ESVM_SAVE_LOAD_MODEL_FILE_FORMAT_COMPARE*/
    return passThroughDisplayTestStatus(__func__, SKIPPED);
    #endif/*TEST_ESVM_SAVE_LOAD_MODEL_FILE_FORMAT_COMPARE*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

int test_ESVM_ModelFromStructSVM()
{
    #if TEST_ESVM_MODEL_STRUCT_SVM_PARAMS
    logstream logger(LOGGER_FILE);
    logger << "Running '" << __func__ << "' test..." << std::endl;

    logger << "Testing ESVM model resetting evaluation using predefined 'svm_model' struct..." << std::endl;
    try
    {
        // verify valid model setting
        logger << "Generating dummy 'svm_model' for setting functionality evaluation..." << std::endl;
        svm_model* validModel = buildDummyExemplarSvmModel();
        ASSERT_LOG(ESVM::checkModelParameters(validModel), "Valid SVM model parameters check should haved returned 'true' status");

        logger << "Testing ESVM model setting operation with dummy 'svm_model'..." << std::endl;
        ESVM esvm(validModel, "TEST");
    }
    catch (std::exception& ex)
    {
        logger << "Valid test procedures should not have raised an exception." << std::endl
               << "Exception: [" << ex.what() << "]" << std::endl;
        return passThroughDisplayTestStatus(__func__, -1);
    }

    size_t nSVM_preTrained = 12, nSVM_notTrained = 10;
    std::vector<svm_model*>invalidModels_preTrained(nSVM_preTrained);
    std::vector<svm_model*>invalidModels_notTrained(nSVM_notTrained);
    try
    {
        // verify invalid model parameters
        ASSERT_LOG(!ESVM::checkModelParameters(nullptr), "No reference to SVM model paramter check should have returned 'false' status");

        logger << "Generating dummy 'svm_model' with invalid parameters..." << std::endl;
        for (size_t svm = 0; svm < nSVM_notTrained; ++svm)
            invalidModels_notTrained[svm] = buildDummyExemplarSvmModel(FreeModelState::PARAM);
        for (size_t svm = 0; svm < nSVM_preTrained; ++svm)
            invalidModels_preTrained[svm] = buildDummyExemplarSvmModel(FreeModelState::MODEL);

        // params required for both 'free_sv' = 0 | 1
        invalidModels_preTrained[0]->param.kernel_type = RBF;           // not 'LINEAR' when 'free_sv' = 0
        invalidModels_preTrained[1]->param.svm_type = ONE_CLASS;        // not 'C_SVC'
        invalidModels_preTrained[2]->label[1] = 2;                      // negative class label as '2' instead of expected 'ESVM_NEGATIVE_CLASS'
        invalidModels_preTrained[3]->nr_class = 3;                      // not 2 class
        invalidModels_preTrained[3]->sv_coef = Malloc(double*, 2);      // create the <nr_class-1> sv_coef that correspond to 'nr_class=3'
        invalidModels_preTrained[3]->sv_coef[0] = Malloc(double, 5);    //     otherwise, we get invalid memory access that is not the current test
        invalidModels_preTrained[3]->sv_coef[1] = Malloc(double, 5);
        std::vector<double> svc{ 3.5, -0.1, -0.2, -0.1, -0.2 };
        for (int sv = 0; sv < 5; ++sv) {
            invalidModels_preTrained[3]->sv_coef[0][sv] = svc[sv];
            invalidModels_preTrained[3]->sv_coef[1][sv] = svc[sv];
        }
        invalidModels_preTrained[4]->l = 1;                             // >= 2 samples (1 positive + 1 negative minimum)
        invalidModels_preTrained[5]->probA = Malloc(double, 2);         // probability estimates not matching
        invalidModels_preTrained[5]->probA[0] = 1;
        invalidModels_preTrained[5]->probA[1] = 1;
        invalidModels_preTrained[5]->probB = nullptr;
        invalidModels_notTrained[0]->param.kernel_type = POLY;          // not 'LINEAR' when 'free_sv' = 0
        invalidModels_notTrained[1]->param.svm_type = NU_SVC;           // not 'C_SVC'
        invalidModels_notTrained[2]->label[1] = 1;                      // negative class label as '1' (duplicate of 'ESVM_POSITIVE_CLASS')
        invalidModels_notTrained[3]->nr_class = 1;                      // not 2 class
        invalidModels_notTrained[4]->l = 0;                             // >= 2 samples (1 positive + 1 negative minimum)
        invalidModels_notTrained[5]->probA = nullptr;                   // probability estimates not matching
        invalidModels_notTrained[5]->probB = Malloc(double, 2);
        invalidModels_notTrained[5]->probB[0] = 1;
        invalidModels_notTrained[5]->probB[1] = 1;

        // params required only when 'free_sv' = 0
        invalidModels_notTrained[6]->param.C = -1;                      // C <= 0
        invalidModels_notTrained[7]->param.nr_weight = 1;               // not 0 or 2
        invalidModels_notTrained[8]->param.weight_label[1] = 2;         // negative class weight label as '2' instead of 'ESVM_NEGATIVE_CLASS'
        invalidModels_notTrained[9]->param.weight[1] = -1;              // negative class weight not > 0

        // params required only when 'free_sv' = 1
        invalidModels_preTrained[6]->nSV[0] = 0;                        // no positive samples
        FreeNull(invalidModels_preTrained[7]->rho);                     // missing rho
        FreeNull(invalidModels_preTrained[8]->sv_coef[0]);              // missing SV coefficient for decision function
        FreeNull(invalidModels_preTrained[9]->sv_coef[0]);              // missing SV coefficient container (only 1D for ESVM containing 2 classes)
        FreeNull(invalidModels_preTrained[9]->sv_coef);
        FreeNull(invalidModels_preTrained[10]->SV[2]);                  // missing any of the SV features (not zero to validate whole set check)
        for (int sv = 0; sv < invalidModels_preTrained[11]->l; ++sv)    // missing the SV container
            free(invalidModels_preTrained[11]->SV[sv]);
        FreeNull(invalidModels_preTrained[11]->SV);

        for (size_t svm = 0; svm < nSVM_notTrained; ++svm)
            ASSERT_LOG(!ESVM::checkModelParameters(invalidModels_notTrained[svm]), "Invalid SVM not trained model parameters (svm=" +
                       std::to_string(svm) + ") check should have returned 'false' status");
        for (size_t svm = 0; svm < nSVM_preTrained; ++svm)
            ASSERT_LOG(!ESVM::checkModelParameters(invalidModels_preTrained[svm]), "Invalid SVM pre-trained model parameters (svm=" +
                       std::to_string(svm) + ") check should have returned 'false' status");
        logger << "Generation of dummy 'svm_model' for test validated." << std::endl;
    }
    catch (std::exception& ex)
    {
        logger << "Valid test preparation of test classes should not have raised an exception." << std::endl
               << "Exception: [" << ex.what() << "]" << std::endl;
        for (size_t svm = 0; svm < nSVM_notTrained; ++svm)
        {
            svm_destroy_param(&invalidModels_notTrained[svm]->param);
            ESVM::destroyModel(&invalidModels_notTrained[svm]);
        }
        for (size_t svm = 0; svm < nSVM_preTrained; ++svm)
        {
            svm_destroy_param(&invalidModels_preTrained[svm]->param);
            ESVM::destroyModel(&invalidModels_preTrained[svm]);
        }
        return passThroughDisplayTestStatus(__func__, -2);
    }

    // verify invalid model parameters not initialized (not trained models)
    logger << "Testing invalid 'svm_model' parameters for not trained models." << std::endl;
    for (size_t svm = 0; svm < nSVM_notTrained; ++svm)
    {
        ESVM esvm;
        try
        {
            esvm = ESVM(invalidModels_notTrained[svm], "INVALID " + std::to_string(svm));
            logger << "Invalid parameters specified in SVM model to reset should have raised an exception." << std::endl;
            return passThroughDisplayTestStatus(__func__, -3);
        }
        catch (...) {} // expected exception
        ASSERT_LOG(!esvm.isModelTrained(), "Invalid parameters should not have allowed ESVM initialization with model considered as trained");
        ASSERT_LOG(!esvm.isModelSet(), "Invalid parameters should not have allowed ESVM resetting with invalid model");
        ESVM::destroyModel(&invalidModels_notTrained[svm]);
    }

    // verify invalid model parameters not initialized (pre-trained models)
    logger << "Testing invalid 'svm_model' parameters for pre-trained models." << std::endl;
    for (size_t svm = 0; svm < nSVM_preTrained; ++svm)
    {
        ESVM esvm;
        try
        {
            esvm = ESVM(invalidModels_preTrained[svm], "INVALID " + std::to_string(svm));
            logger << "Invalid parameters specified in SVM model to reset should have raised an exception." << std::endl;
            return passThroughDisplayTestStatus(__func__, -4);
        }
        catch (...) {} // expected exception
        ASSERT_LOG(!esvm.isModelTrained(), "Invalid parameters should not have allowed ESVM initialization with model considered as trained");
        ASSERT_LOG(!esvm.isModelSet(), "Invalid parameters should not have allowed ESVM resetting with invalid model");
        ESVM::destroyModel(&invalidModels_preTrained[svm]);
    }

    #else/*TEST_ESVM_MODEL_STRUCT_SVM_PARAMS*/
    return passThroughDisplayTestStatus(__func__, SKIPPED);
    #endif/*TEST_ESVM_MODEL_STRUCT_SVM_PARAMS*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

int test_ESVM_ModelMemoryOperations()
{
    #if TEST_ESVM_MODEL_MEMORY_OPERATIONS
    logstream logger(LOGGER_FILE);
    logger << "Running '" << __func__ << "' test..." << std::endl;

    logger << "Testing ESVM model memory operations evaluation (ctor, copy, move, operator=, reset, dtor)..." << std::endl;
    std::string modelFileName = "test_model-memory-operations.model";
    svm_model* model;
    try {   // scope to call ESVM destructors before actual end of test
        try
        {
            // test deallocated memory when destructor is automatically called from out of scope
            model = buildDummyExemplarSvmModel();
            {
                ESVM esvm(model, "TEST-DESTRUCTOR");
                ASSERT_LOG(esvm.isModelTrained(), "ESVM model should have been trained and properly set to evaluate following functionality");
            }   // out of score will call destructor

            // verify results of model destructor
            /* --- CANNOT BE CHECKED AS SVM_MODEL IS NOW DEEPCOPIED
            ASSERT_LOG(model->label == nullptr, "Model 'label' should have been deallocated and its reference be set to 'null'");
            ASSERT_LOG(model->nSV == nullptr, "Model 'nSV' should have been deallocated and its reference be set to 'null'");
            ASSERT_LOG(model->probA == nullptr, "Model 'probA' should have been deallocated and its reference be set to 'null'");
            ASSERT_LOG(model->probB == nullptr, "Model 'probB' should have been deallocated and its reference be set to 'null'");
            ASSERT_LOG(model->rho == nullptr, "Model 'rho' should have been deallocated and its reference be set to 'null'");
            ASSERT_LOG(model->sv_coef == nullptr, "Model 'coef' container should have been deallocated and its reference be set to 'null'");
            ASSERT_LOG(model->sv_indices == nullptr, "Model 'sv_indices' should have been deallocated and its reference be set to 'null'");
            ASSERT_LOG(model->SV == nullptr, "Model 'SV' reference container should have been deallocated and its reference be set to 'null'");
            */
            //destroyDummyExemplarSvmModelContent(model); /*IGNORE FOR NOW - ONLY VALIDATE VALUES - MEMORY LEAK OF THE TEST NOT A MAJOR CONCERN*/
        }
        catch (std::exception& ex)
        {
            logger << "Valid test procedure destroying the model should not have raised an exception." << std::endl
                   << "Exception: [" << ex.what() << "]" << std::endl;
            return passThroughDisplayTestStatus(__func__, -1);
        }
        ESVM::destroyModel(&model);

        // test deallocated memory when reset of model is called
        ESVM esvm;
        try
        {
            // prepare test model data (pre-trained model)
            model = buildDummyExemplarSvmModel(FreeModelState::MODEL);
            esvm = ESVM(model, "TEST-RESET");
            ASSERT_LOG(esvm.isModelSet(), "ESVM pre-trained model should have been properly set to evaluate following functionality");
        }
        catch (std::exception& ex)
        {
            logger << "Valid test model preparation for reset evaluation should not have raised an exception." << std::endl
                   << "Exception: [" << ex.what() << "]" << std::endl;
            bfs::remove_all(modelFileName);
            return passThroughDisplayTestStatus(__func__, -2);
        }
        logger << "Model for reset memory evaluation properly generated, preparing to reset..." << std::endl;
        try
        {
            // save to file then reload to induce a 'reset' (replace old model by loaded one)
            esvm.saveModelFile(modelFileName, LIBSVM);
            esvm.loadModelFile(modelFileName, LIBSVM);
            ASSERT_LOG(esvm.isModelSet(), "ESVM pre-trained model should have been set from reset operation");
        }
        catch (std::exception& ex)
        {
            logger << "Valid test resetting operation for model reset evaluation should not have raised an exception." << std::endl
                   << "Exception: [" << ex.what() << "]" << std::endl;
            bfs::remove_all(modelFileName);
            return passThroughDisplayTestStatus(__func__, -3);
        }
        /* --- CANNOT BE CHECKED AS SVM_MODEL IS NOW DEEPCOPIED
        logger << "Model for reset memory evaluation properly reset, validating reset parameters..." << std::endl;
        try
        {
            // verify results of model reset
            ASSERT_LOG(model->label == nullptr, "Model 'label' should have been deallocated and its reference be set to 'null'");
            ASSERT_LOG(model->nSV == nullptr, "Model 'nSV' should have been deallocated and its reference be set to 'null'");
            ASSERT_LOG(model->probA == nullptr, "Model 'probA' should have been deallocated and its reference be set to 'null'");
            ASSERT_LOG(model->probB == nullptr, "Model 'probB' should have been deallocated and its reference be set to 'null'");
            ASSERT_LOG(model->rho == nullptr, "Model 'rho' should have been deallocated and its reference be set to 'null'");
            ASSERT_LOG(model->sv_coef == nullptr, "Model 'sv_coef' container should have been deallocated and its reference be set to 'null'");
            ASSERT_LOG(model->sv_indices == nullptr, "Model 'sv_indices' should have been deallocated and its reference be set to 'null'");
            ASSERT_LOG(model->SV == nullptr, "Model 'SV' reference container should have been deallocated and its reference be set to 'null'");
        }
        catch (std::exception& ex)
        {
            logger << "Valid test parameter validation for model reset evaluation should not have raised an exception." << std::endl
                   << "Exception: [" << ex.what() << "]" << std::endl;
            bfs::remove_all(modelFileName);
            return passThroughDisplayTestStatus(__func__, -4);
        }
        */

        ESVM::destroyModel(&model);

        /// TODO
        //OPERATION THAT CHECKS COPY-CTOR TO SELF! (not dealloc)
        //OPERATION THAT CHECKS MOVE-CTOR + to self
        //OPERATION THAT CHECKS =()-CTOR, + to self
        //OPERATION THAT CHECKS DTOR direct call


    } // end scope for ESVM destructor calls
    catch (...) {}

    bfs::remove_all(modelFileName);

    #else/*TEST_ESVM_MODEL_MEMORY_OPERATIONS*/
    return passThroughDisplayTestStatus(__func__, SKIPPED);
    #endif/*TEST_ESVM_MODEL_MEMORY_OPERATIONS*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

int test_ESVM_ModelMemoryParamCheck()
{
    #if TEST_ESVM_MODEL_MEMORY_PARAM_CHECK
    logstream logger(LOGGER_FILE);
    logger << "Running '" << __func__ << "' test..." << std::endl;

    logger << "Testing ESVM model resetting evaluation and validation of updated parameters..." << std::endl;
    std::string modelFileName = "test_model2-memory-param-reset.model";
    try {   // scope to call ESVM destructors before actual end of test

        svm_model *model1, *model2;
        ESVM esvm1, esvm2;
        std::vector<FeatureVector> testFeatureVectors;  // feature vectors for testing 'model2'
        std::vector<double> expectedPredictions;        // values expected with corresponding feature vectors with 'model2'
        try
        {
            // initial model to test that parameters are reset
            model1 = buildDummyExemplarSvmModel(FreeModelState::MODEL);
            ///logger << "esvm1 = ESVM(&model1, 'RESET-PARAM-1');" << std::endl;  ///TODO REMOVE
            std::string modelID1 = "RESET-PARAM-1";
            esvm1 = ESVM(model1, modelID1);
            ///logger << "INFOS::" << std::endl;  ///TODO REMOVE
            ///esvm1.logModelParameters(true);  ///TODO REMOVE
            ASSERT_LOG(esvm1.isModelTrained(), "Model for parameter reset validation should be set and trained for following evaluations");
            ASSERT_LOG(esvm1.targetID == modelID1, "Model for parameter reset validation should be set with expected target ID");
        }
        catch (std::exception& ex)
        {
            logger << "Generation of first test model for reset parameter evaluation should not have raised an exception." << std::endl
                   << "Exception: [" << ex.what() << "]" << std::endl;
            return passThroughDisplayTestStatus(__func__, -1);
        }
        try
        {
            std::vector<double> sv{ 2.4, -0.8, -0.4 };

            // create minimal model with parameters different than 'buildDummyExemplarSvmModel' to test against after reset
            model2 = esvm2.makeEmptyModel();
            model2->param.kernel_type = LINEAR;
            model2->param.svm_type = C_SVC;
            model2->free_sv = 1;
            model2->nr_class = 2;
            model2->l = sv.size();
            model2->param.weight = nullptr;
            model2->param.weight_label = nullptr;
            model2->rho = Malloc(double, 1);
            model2->rho[0] = 4.8;
            model2->sv_indices = Malloc(int, model2->l);
            model2->sv_indices[0] = 10;
            model2->sv_indices[1] = 9;
            model2->sv_indices[2] = 8;
            model2->sv_coef = Malloc(double*, model2->nr_class - 1);
            model2->sv_coef[0] = Malloc(double, model2->l);
            for (size_t i = 0; i < sv.size(); ++i)
                model2->sv_coef[0][i] = sv[i];
            model2->label = Malloc(int, model2->nr_class);
            model2->label[0] = ESVM_POSITIVE_CLASS;
            model2->label[1] = ESVM_NEGATIVE_CLASS;
            model2->nSV = Malloc(int, model2->nr_class);
            model2->nSV[0] = 1;
            model2->nSV[1] = model2->l - 1;
            model2->SV = Malloc(svm_node*, model2->l);
            int nFeatures = 2;                          // number of samples features different to induce error on failed reset of parameters
            for (int sv = 0; sv < model2->l; ++sv) {
                model2->SV[sv] = Malloc(svm_node, nFeatures + 1);
                for (int f = 0; f < nFeatures + 1; ++f)
                    model2->SV[sv][f].index = (f == nFeatures) ? -1 : f;
            }
            model2->SV[0][0].value = -2.8;   model2->SV[0][1].value = 1.25;
            model2->SV[1][0].value = 3.25;   model2->SV[1][1].value = 0.25;
            model2->SV[2][0].value = 1.75;   model2->SV[2][1].value = -1.5;
            model2->param.probability = 0;
            model2->probA = nullptr;
            model2->probB = nullptr;

            // set some testing feature vectors, verify that expected prediction values are obtained from proper parameter update
            testFeatureVectors.push_back(FeatureVector{ -2.8, 1.25 });    expectedPredictions.push_back(-14.32);
            testFeatureVectors.push_back(FeatureVector{ 3.25, 0.25 });    expectedPredictions.push_back(6.25);
            testFeatureVectors.push_back(FeatureVector{ -1.8, 1.10 });    expectedPredictions.push_back(-10.92);

            std::string modelID2 = "RESET-PARAM-2";
            esvm2 = ESVM(model2, modelID2);
            ASSERT_LOG(esvm2.isModelTrained(), "Model for parameter reset validation should be set and trained for following evaluations");
            ASSERT_LOG(esvm2.targetID == modelID2, "Model for parameter reset validation should be set with expected target ID");


            //////// TODO /////////
            /*TEST MODEL PARAMETERS UPDATED*/
        }
        catch (std::exception& ex)
        {
            logger << "Generation of second test model for reset parameter evaluation should not have raised an exception." << std::endl
                   << "Exception: [" << ex.what() << "]" << std::endl;
            return passThroughDisplayTestStatus(__func__, -2);
        }
        try
        {
            esvm2.saveModelFile(modelFileName, LIBSVM);
            ASSERT_LOG(bfs::is_regular_file(modelFileName), "Second models file should have been created");
        }
        catch (std::exception& ex)
        {
            logger << "Second model saving to file for reset parameter evaluation should not have raised an exception." << std::endl
                   << "Exception: [" << ex.what() << "]" << std::endl;
            return passThroughDisplayTestStatus(__func__, -3);
        }
        try
        {
            // induce reset of 'model1' by loading 'model2' from file
            ASSERT_LOG(esvm1.loadModelFile(modelFileName, LIBSVM), "Loading of second model into first ESVM should be successful");
            for (size_t fv = 0; fv < testFeatureVectors.size(); ++fv)
            {
                double pred = esvm2.predict(testFeatureVectors[fv]);
                ASSERT_LOG(doubleAlmostEquals(pred, expectedPredictions[fv]),
                           "Second model modified parameters from reset operation should return expected result (" +
                           std::to_string(pred) + " != " + std::to_string(expectedPredictions[fv]) + ")");
            }
        }
        catch (std::exception& ex)
        {
            bfs::remove_all(modelFileName);
            logger << "Final model reset parameter evaluation should not have raised an exception." << std::endl
                   << "Exception: [" << ex.what() << "]" << std::endl;
            return passThroughDisplayTestStatus(__func__, -4);
        }

        ///logger << "CLEANUP MODEL1" << std::endl;   ///TODO REMOVE
        ESVM::destroyModel(&model1);
        ///destroyDummyExemplarSvmModelContent(&model1);
        ///logger << "CLEANUP MODEL2" << std::endl;   ///TODO REMOVE
        ESVM::destroyModel(&model2);
        ///destroyDummyExemplarSvmModelContent(&model2);
        ///logger << "CLEANUP FILE" << std::endl;   ///TODO REMOVE
        bfs::remove_all(modelFileName);
        ///logger << "CLEANUP DONE" << std::endl;   ///TODO REMOVE

    } // end scope for ESVM destructor calls
    catch (...) {}

    ///////////////////////////////////////////////////////////////////////// TODO SAME TEST, BUT FOR 'BINARY' FILE RESET
    bfs::remove_all(modelFileName);

    #else/*TEST_ESVM_MODEL_MEMORY_PARAM_CHECK*/
    return passThroughDisplayTestStatus(__func__, SKIPPED);
    #endif/*TEST_ESVM_MODEL_MEMORY_PARAM_CHECK*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

/* ===============
    PROCEDURES
=============== */

int proc_readDataFiles()
{
    #if PROC_READ_DATA_FILES
    #ifdef ESVM_HAS_CHOKEPOINT
    logstream logger(LOGGER_FILE);
    logger << "Running '" << __func__ << "' test ['PROC_READ_DATA_FILES'=" << displayAsBinary<8>(PROC_READ_DATA_FILES, true) << "]..." << std::endl;

    #if PROC_READ_DATA_FILES & 0b00000001   // (1) Run ESVM training/testing using images and feature extraction on whole image
    // Specifying Size(0,0) or Size(1,1) will result in not applying patches (use whole ROI)
    cv::Size patchCounts = cv::Size(1, 1);
    cv::Size imageSize = cv::Size(64, 64);

    #elif PROC_READ_DATA_FILES & 0b00000010 // (2) Run ESVM training/testing using images and patch-based feature extraction
    // Number of patches to use in each direction, must fit within the ROIs (ex: 4x4 patches & ROI 128x128 -> 16 patches of 32x32)
    cv::Size patchCounts = cv::Size(3, 3);
    cv::Size imageSize = cv::Size(48, 48);
    #endif/* (1|2) params */

    #if PROC_READ_DATA_FILES & 0b00000011
    RETURN_ERROR(proc_runSingleSamplePerPersonStillToVideo_FullChokePoint(imageSize, patchCounts));
    #endif/* (1|2) test */

    #if PROC_READ_DATA_FILES & 0b00000100   // (4) Run ESVM training/testing using pre-generated whole image samples files
    RETURN_ERROR(proc_runSingleSamplePerPersonStillToVideo_DataFiles_WholeImage());
    #endif/* (4) */

    #if PROC_READ_DATA_FILES & 0b00001000   // (8) Run ESVM training/testing using pre-generated (feature+patch)-based samples files
    int nPatches = patchCounts.width * patchCounts.height;
    RETURN_ERROR(proc_runSingleSamplePerPersonStillToVideo_DataFiles_DescriptorAndPatchBased(nPatches));
    #endif/* (8) */

    #if PROC_READ_DATA_FILES & 0b11110000   // (16|32|64|128) Run ESVM training/testing using pre-generated patch-based negatives samples files
    RETURN_ERROR(proc_runSingleSamplePerPersonStillToVideo_NegativesDataFiles_PositivesExtraction_PatchBased());
    #endif/* (16|32|64|128) */

    #else/*ESVM_HAS_CHOKEPOINT*/
    return passThroughDisplayTestStatus(__func__, SKIPPED, "Missing required 'ESVM_HAS_CHOKEPOINT' definition.");
    #endif/*ESVM_HAS_CHOKEPOINT*/
    #else/*PROC_READ_DATA_FILES*/
    return passThroughDisplayTestStatus(__func__, SKIPPED);
    #endif/*PROC_READ_DATA_FILES*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

/******************************************************************************************************************************
FUNCTION DETAILS

    Number of patches to use in each direction, must fit within the ROIs (ex: 4x4 patches & ROI 128x128 -> 16 patches of 32x32)
    Specifying Size(0,0) or Size(1,1) will result in not applying patches (use whole ROI)

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

******************************************************************************************************************************/
int proc_runSingleSamplePerPersonStillToVideo(cv::Size patchCounts)
{
    #if PROC_ESVM_BASIC_STILL2VIDEO
    logstream logger(LOGGER_FILE);
    logger << "Running '" << __func__ << "' test..." << std::endl;
    using namespace esvm::path;

    // ------------------------------------------------------------------------------------------------------------------------
    // window to display loaded images and initialization
    // ------------------------------------------------------------------------------------------------------------------------
    cv::namedWindow(WINDOW_NAME);
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
    for (int i = 0; i < NB_ENROLLMENT; ++i)
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
    for (int i = 0; i <= 6; ++i) probeGroundThruth[i] = "ID0013";
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
    for (int i = 7; i <= 15; ++i) probeGroundThruth[i] = "ID0012";
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
    for (int i = 16; i <= 25; ++i) probeGroundThruth[i] = "ID0011";
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
    for (int i = 26; i <= 37; ++i) probeGroundThruth[i] = "ID0029";
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
    for (int i = 38; i <= 52; ++i) probeGroundThruth[i] = "ID0016";
    /* --- ID0009 --- */
    matProbeSamples[53] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_20/000410.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[54] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_20/000415.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[55] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_20/000420.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[56] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_20/000425.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[57] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_25/000441.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[58] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_25/000445.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[59] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_25/000450.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    for (int i = 53; i <= 59; ++i) probeGroundThruth[i] = "ID0009";
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
    for (int i = 60; i <= 68; ++i) probeGroundThruth[i] = "ID0004";
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
    for (int i = 69; i <= 79; ++i) probeGroundThruth[i] = "ID0020";
    /* --- ID0023 --- */
    matProbeSamples[80] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_37/000541.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[81] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_37/000545.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[82] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_37/000550.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[83] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_37/000555.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[84] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_37/000561.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[85] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_37/000565.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    matProbeSamples[86] = imSplitPatches(imReadAndDisplay(roiVideoImagesPath + "person_37/000570.png", WINDOW_NAME, cv::IMREAD_GRAYSCALE), patchCounts);
    for (int i = 80; i <= 86; ++i) probeGroundThruth[i] = "ID0023";
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
    for (int i = 87; i <= 95; ++i) probeGroundThruth[i] = "ID0026";

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
    for (int p = 0; p < nPatches; ++p)
    {
        logger << "Converting patches at index " << p << "..." << std::endl;
        mwNegativeSamples[p] = mwArray(NB_NEGATIVE_IMAGES, 1, mxCELL_CLASS);
        mwProbeSamples[p]    = mwArray(NB_PROBE_IMAGES, 1, mxCELL_CLASS);

        logger << "Converting positive training samples..." << std::endl;
        for (int i = 0; i < NB_ENROLLMENT; ++i)
        {
            // Initialize vertor only on first patch for future calls
            if (p == 0)
                mwPositiveSamples[i] = std::vector<mwArray>(nPatches);

            // Duplicate unique positive to generate a pool samples
            mwPositiveSamples[i][p] = mwArray(NB_POSITIVE_DUPLICATION, 1, mxCELL_CLASS);
            mwArray dupPositive = convertCvToMatlabMat(matPositiveSamples[i][p]);
            for (int j = 0; j < NB_POSITIVE_DUPLICATION; ++j)
                mwPositiveSamples[i][p].Get(1, j + 1).Set(dupPositive);
        }

        logger << "Converting negative training samples..." << std::endl;
        for (int i = 0; i < NB_NEGATIVE_IMAGES; ++i)
            mwNegativeSamples[p].Get(1, i + 1).Set(convertCvToMatlabMat(matNegativeSamples[i][p]));

        logger << "Converting probe testing samples..." << std::endl;
        for (int i = 0; i < NB_PROBE_IMAGES; ++i)
            mwProbeSamples[p].Get(1, i + 1).Set(convertCvToMatlabMat(matProbeSamples[i][p]));
    }

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
        for (int i = 0; i < 5; ++i)
        {
            logger << "Data detail " << i << ":" << std::endl;
            logger << mwPositiveSamples[0].Get(1,i+1).ToString() << std::endl;
        }

        cv::Size is = matNegativeSamples[0][0].size();
        cv::Mat im = imReadAndDisplay(esvm::path::refStillImagesPath + "roi" + targetName[0] + ".JPG", WINDOW_NAME, cv::IMREAD_COLOR);
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
        for (int i = 0; i < size; ++i)
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

        for (int i = 0; i < NB_ENROLLMENT; ++i)
        {
            logger << "Starting for individual " << i << ": " + targetName[i] << std::endl;
            double scoreFusion[NB_PROBE_IMAGES] = { 0 };
            for (int p = 0; p < nPatches; ++p)
            {
                logger << "Running Exemplar-SVM training..." << std::endl;
                esvm_train_individual(1, models[p], mwPositiveSamples[i][p], mwNegativeSamples[p], mwArray(targetName[i].c_str()));
                logger << "Running Exemplar-SVM testing..." << std::endl;
                esvm_test_individual(1, mwScores[p], models[p], mwProbeSamples[p]);
                double scores[NB_PROBE_IMAGES];
                mwScores[p].GetData(scores, NB_PROBE_IMAGES);
                for (int j = 0; j < NB_PROBE_IMAGES; ++j)
                {
                    // score accumulation from patches with normalization
                    double normPatchScore = normalizeClassScoreToSimilarity(scores[j]);
                    scoreFusion[j] += normPatchScore;
                    std::string probeGT = (probeGroundThruth[j] == targetName[i] ? "positive" : "negative");
                    logger << "Score for patch " << p << " of probe " << j << " (" << probeGT << "): " << normPatchScore << std::endl;
                }
            }
            for (int j = 0; j < NB_PROBE_IMAGES; ++j)
            {
                // average of score accumulation for fusion
                std::string probeGT = (probeGroundThruth[j] == targetName[i] ? "positive" : "negative");
                logger << "Score fusion of probe " << j << " (" << probeGT << "): " << scoreFusion[j] / nPatches << std::endl;
            }
            logger << "Completed for individual " << i << ": " + targetName[i] << std::endl;
        }
        logger << "Success" << std::endl;
    }
    catch (const mwException& e)
    {
        logger << e.what() << std::endl;
        return passThroughDisplayTestStatus(__func__, -1);
    }
    catch (...)
    {
        logger << "Unexpected error thrown" << std::endl;
        return passThroughDisplayTestStatus(__func__, -2);
    }

    #else/*!PROC_ESVM_BASIC_STILL2VIDEO*/
    return passThroughDisplayTestStatus(__func__, SKIPPED);
    #endif/*PROC_ESVM_BASIC_STILL2VIDEO*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

/**************************************************************************************************************************
TEST DEFINITION

    Enrolls the training target using their still image vs. non-targets of multiple video sequence from ChokePoint dataset.
    The enrolled individuals are represented by ensembles of Exemplar-SVM and are afterward tested using the probe videos.
    Classification performances are then evaluated each positive target vs. probe samples in term of FPR/TPR for ROC curbe.
**************************************************************************************************************************/
int proc_runSingleSamplePerPersonStillToVideo_FullChokePoint(cv::Size imageSize, cv::Size patchCounts)
{
    #ifdef ESVM_HAS_CHOKEPOINT
    logstream logger(LOGGER_FILE);
    logger << "Running '" << __func__ << "' test..." << std::endl;

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
    size_t nPatches = patchCounts.width * patchCounts.height;
    if (nPatches == 0) nPatches = 1;
    logger << "Starting single sample per person still-to-video full ChokePoint test..." << std::endl
           << "   useSyntheticPositives:    " << TEST_USE_SYNTHETIC_GENERATION << std::endl
           << "   imageSize:                " << imageSize << std::endl
           << "   patchCounts:              " << patchCounts << std::endl
           << "   useHistEqual:             " << ESVM_USE_HIST_EQUAL << std::endl;

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
    for (size_t pos = 0; pos < nPositives; ++pos)
    {
        // Add additional positive representations as requested
        #if TEST_USE_SYNTHETIC_GENERATION

            // Get original positive image with preprocessing but without patches splitting
            cv::Mat img = imPreprocess(esvm::path::refStillImagesPath + "roiID" + positivesID[pos] + ".tif", imageSize, cv::Size(1, 1),
                                       ESVM_USE_HIST_EQUAL, WINDOW_NAME, cv::IMREAD_GRAYSCALE)[0];
            // Get synthetic representations from original and apply patches splitting each one
            std::vector<cv::Mat> representations = imSyntheticGeneration(img);
            // Reinitialize sub-container for augmented representations using synthetic images
            nRepresentations = representations.size();
            size_t dimsRepresentation[2] { nRepresentations, nPatches };
            matPositiveSamples[pos] = xstd::mvector<2, cv::Mat>(dimsRepresentation);

            /// ############################################# #pragma omp parallel for
            for (size_t r = 0; r < nRepresentations; ++r)
            {
                std::vector<cv::Mat> patches = imSplitPatches(representations[r], patchCounts);
                for (size_t p = 0; p < nPatches; ++p)
                    matPositiveSamples[pos][r][p] = patches[p];
            }

        // Only original representation otherwise (no synthetic images)
        #else/*!TEST_USE_SYNTHETIC_GENERATION*/

            matPositiveSamples[pos][0] = imPreprocess<cv::Mat>(esvm::path::refStillImagesPath + "roiID" + positivesID[pos] + ".tif",
                                                               imageSize, patchCounts, ESVM_USE_HIST_EQUAL, WINDOW_NAME, cv::IMREAD_GRAYSCALE);

        #endif/*TEST_USE_SYNTHETIC_GENERATION*/
    }

    /// ################################################################################ DEBUG DISPLAY POSITIVES (+SYNTH)
    /*
    logger << "SHOWING DEBUG POSITIVE SAMPLES" << std::endl;
    for (int i = 0; i < nPositives; ++i)
    {
        for (int j = 0; j < matPositiveSamples[i].size(); ++j)
        {
            for (int k = 0; k < matPositiveSamples[i][j].size(); ++k)
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
    for (size_t pos = 0; pos < nPositives; ++pos)
    {
        /// ################################################## #pragma omp parallel for
        for (size_t p = 0; p < nPatches; ++p)
        {
            for (size_t d = 0; d < nDescriptors; ++d)
            {
                /// ################################################## #pragma omp parallel for
                for (int r = 0; r < nRepresentations; ++r)
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
                    for (size_t r = 0; r < nRepresentations; ++r)
                        for (size_t dup = 1; dup < nDuplications; ++dup)
                            fvPositiveSamples[pos][p][d].push_back(fvPositiveSamples[pos][p][d][r]);
            }
        }
    }
    if (nDuplications > 1)
        nRepresentations *= nDuplications;
    for (size_t d = 0; d < nDescriptors; ++d)
        logger << "Features dimension (" + descriptorNames[d] + "): " << fvPositiveSamples[0][0][d][0].size() << std::endl;

    // Tests divided per sequence information according to selected mode
    std::vector<ChokePoint::PortalType> types = ChokePoint::PORTAL_TYPES;
    bfs::directory_iterator endDir;
    std::string seq;
    for (int sn = 1; sn <= ChokePoint::SESSION_QUANTITY; ++sn)      // session number
    {
        #if TEST_CHOKEPOINT_SEQUENCES_MODE == 0                     // regroup all by sequence
        cv::namedWindow(WINDOW_NAME);
        #endif/*TEST_CHOKEPOINT_SEQUENCES_MODE*/

        for (int pn = 1; pn <= ChokePoint::PORTAL_QUANTITY; ++pn) { // portal number
        for (auto pt = types.begin(); pt != types.end(); ++pt) {    // portal type
        for (int cn = 1; cn <= ChokePoint::CAMERA_QUANTITY; ++cn)   // camera number
        {
            #if TEST_CHOKEPOINT_SEQUENCES_MODE == 1                 // slipt all per sequence/portal/camera
            cv::namedWindow(WINDOW_NAME);
            // Reset vectors for next test sequences
            matNegativeSamples.clear();
            matProbeSamples.clear();
            negativeSamplesID.clear();
            probeSamplesID.clear();
            for (size_t pos = 0; pos < nPositives; ++pos)
                probeGroundTruth[pos].clear();
            #endif/*TEST_CHOKEPOINT_SEQUENCES_MODE*/

            seq = ChokePoint::getSequenceString(pn, *pt, sn, cn);
            logger << "Loading negative and probe images for sequence " << seq << "..." << std::endl;
            #if TEST_CHOKEPOINT_SEQUENCES_MODE == 0
            seq = "S" + std::to_string(sn);
            #endif/*TEST_CHOKEPOINT_SEQUENCES_MODE*/

            // Add ROI to corresponding sample vectors according to individual IDs
            for (int id = 1; id <= ChokePoint::INDIVIDUAL_QUANTITY; ++id)
            {
                std::string dirPath = esvm::path::roiChokePointCroppedFacePath + ChokePoint::getSequenceString(pn, *pt, sn, cn, id) + "/";
                if (bfs::is_directory(dirPath))
                {
                    for (bfs::directory_iterator itDir(dirPath); itDir != endDir; ++itDir)
                    {
                        if (bfs::is_regular_file(*itDir) && itDir->path().extension() == ".pgm")
                        {
                            std::string strID = ChokePoint::getIndividualID(id);
                            if (contains(negativesID, strID))
                            {
                                size_t neg = matNegativeSamples.size();
                                matNegativeSamples.push_back(xstd::mvector<1, cv::Mat>(nPatches));
                                std::vector<cv::Mat> patches = imPreprocess<cv::Mat>(itDir->path().string(), imageSize, patchCounts,
                                                                                     ESVM_USE_HIST_EQUAL, WINDOW_NAME,
                                                                                     cv::IMREAD_GRAYSCALE);
                                for (size_t p = 0; p < nPatches; ++p)
                                    matNegativeSamples[neg][p] = patches[p];

                                negativeSamplesID.push_back(strID);
                            }
                            else if (contains(probesID, strID))
                            {
                                size_t prb = matProbeSamples.size();
                                matProbeSamples.push_back(xstd::mvector<1, cv::Mat>(nPatches));
                                std::vector<cv::Mat> patches = imPreprocess<cv::Mat>(itDir->path().string(), imageSize, patchCounts,
                                                                                     ESVM_USE_HIST_EQUAL, WINDOW_NAME,
                                                                                     cv::IMREAD_GRAYSCALE);
                                for (size_t p = 0; p < nPatches; ++p)
                                    matProbeSamples[prb][p] = patches[p];

                                probeSamplesID.push_back(strID);
                                for (size_t pos = 0; pos < nPositives; ++pos)
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
            for (int pos = 0; pos < nPositives; ++pos)
                ASSERT_LOG(!contains(negativeSamplesID, positivesID[pos]), "Positive ID found within negative samples ID");
            for (int neg = 0; neg < negativesID.size(); ++neg)
                ASSERT_LOG(!contains(probeSamplesID, negativesID[neg]), "Negative ID found within probe samples ID");
            for (int prb = 0; prb < probesID.size(); ++prb)
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
            for (size_t p = 0; p < nPatches; ++p)
            {
                for (size_t d = 0; d < nDescriptors; ++d)
                {
                    // switch to (p,d,i) order for patch-based training
                    /// ############################################# #pragma omp parallel for
                    for (size_t neg = 0; neg < nNegatives; ++neg)
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
                    for (size_t prb = 0; prb < nProbes; ++prb)
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
            #if !TEST_FEATURES_NORM_MODE
            logger << "Skipping features normalization" << std::endl;
            #else // Prepare some containers employed by each normalization method
            size_t dimsAllVectors[3]{ nPatches, nDescriptors, nPositives * nRepresentations + nNegatives + nProbes };
            size_t dimsMinMax[2]{ nDescriptors, nPatches };
            xstd::mvector<3, FeatureVector> allFeatureVectors(dimsAllVectors);      // [patch][descriptor][sample](FeatureVector)
            xstd::mvector<2, FeatureVector> minFeaturesCumul(dimsMinMax);           // [descriptor][patch](FeatureVector)
            xstd::mvector<2, FeatureVector> maxFeaturesCumul(dimsMinMax);           // [descriptor][patch](FeatureVector)
            #endif/*TEST_FEATURES_NORM_MODE*/

            // Specific min/max containers according to methods
            #if TEST_FEATURES_NORM_MODE == 1       // Per feature and per patch normalization
            logger << "Searching feature normalization values (per feature, per patch)..." << std::endl;
            #elif TEST_FEATURES_NORM_MODE == 2     // Per feature and across patches normalization
            logger << "Searching feature normalization values (per feature, across patches)..." << std::endl;
            std::vector<FeatureVector> minFeatures(nDescriptors);                   // [descriptor](FeatureVector)
            std::vector<FeatureVector> maxFeatures(nDescriptors);                   // [descriptor](FeatureVector)
            #elif TEST_FEATURES_NORM_MODE == 3     // Across features and across patches normalization
            logger << "Searching feature normalization values (across features, across patches)..." << std::endl;
            FeatureVector minFeatures(nDescriptors, DBL_MAX);                       // [descriptor](double)
            FeatureVector maxFeatures(nDescriptors, -DBL_MAX);                      // [descriptor](double)
            #endif/*ESVM_USE_FEATURES_NORMALIZATION == (1|2|3)*/

            // Accumulate all positive/negative/probes samples to find min/max features according to normalization mode
            for (size_t d = 0; d < nDescriptors; ++d)
            {
                for (size_t p = 0; p < nPatches; ++p)
                {
                    size_t s = 0;  // Sample index
                    for (size_t pos = 0; pos < nPositives; ++pos)
                        for (size_t r = 0; r < nRepresentations; ++r)
                            allFeatureVectors[p][d][s++] = fvPositiveSamples[pos][p][d][r];
                    for (size_t neg = 0; neg < nNegatives; ++neg)
                        allFeatureVectors[p][d][s++] = fvNegativeSamples[p][d][neg];
                    for (size_t prb = 0; prb < nProbes; ++prb)
                        allFeatureVectors[p][d][s++] = fvProbeSamples[p][d][prb];

                    // Find min/max features according to normalization mode
                    findNormParamsPerFeature(MIN_MAX, allFeatureVectors[p][d], minFeaturesCumul[d][p], maxFeaturesCumul[d][p]);
                    #if TEST_FEATURES_NORM_MODE == 1   // Per feature and per patch normalization
                    logger << "Found min/max features for (descriptor,patch) (" << descriptorNames[d] << "," << p << "):" << std::endl
                           << "   MIN: " << featuresToVectorString(minFeaturesCumul[d][p]) << std::endl
                           << "   MAX: " << featuresToVectorString(minFeaturesCumul[d][p]) << std::endl;
                    #endif/*ESVM_USE_FEATURES_NORMALIZATION == 1*/
                }
                #if TEST_FEATURES_NORM_MODE == 2       // Per feature and across patches normalization
                FeatureVector dummyFeatures(minFeaturesCumul[d][0].size());
                findNormParamsFeatures(MIN_MAX, minFeaturesCumul[d], &(minFeatures[d]), &dummyFeatures);
                findNormParamsFeatures(MIN_MAX, maxFeaturesCumul[d], &dummyFeatures, &(maxFeatures[d]));
                logger << "Found min/max features for descriptor '" << descriptorNames[d] << "':" << std::endl
                       << "   MIN: " << featuresToVectorString(minFeatures[d]) << std::endl
                       << "   MAX: " << featuresToVectorString(maxFeatures[d]) << std::endl;
                #elif TEST_FEATURES_NORM_MODE == 3     // Across features and across patches normalization
                double dummyMinMax;
                findNormParamsOverAll(MIN_MAX, minFeaturesCumul[d], minFeatures[d], dummyMinMax);
                findNormParamsOverAll(MIN_MAX, maxFeaturesCumul[d], dummyMinMax, maxFeatures[d]);
                logger << "Found min/max features for descriptor '" << descriptorNames[d] << "':" << std::endl
                       << "   MIN: " << minFeatures[d] << std::endl
                       << "   MAX: " << maxFeatures[d] << std::endl;
                #endif/*ESVM_USE_FEATURES_NORMALIZATION == (2|3)*/
            }

            #if   TEST_FEATURES_NORM_MODE == 1 // Per feature and per patch normalization
            logger << "Applying features normalization (per feature, per patch)..." << std::endl;
            #elif TEST_FEATURES_NORM_MODE == 2 // Per feature and across patches normalization
            logger << "Applying features normalization (per feature, across patches)..." << std::endl;
            #elif TEST_FEATURES_NORM_MODE == 3 // Across features and across patches normalization
            logger << "Applying features normalization (across feature, across patches)..." << std::endl;
            #endif/*ESVM_USE_FEATURES_NORMALIZATION == (1|2|3)*/
            FeatureVector minNorm, maxNorm;
            for (size_t p = 0; p < nPatches; ++p)
            {
                for (size_t d = 0; d < nDescriptors; ++d)
                {
                    #if   TEST_FEATURES_NORM_MODE == 1 // Per feature and per patch normalization
                    minNorm = minFeaturesCumul[d][p];
                    maxNorm = minFeaturesCumul[d][p];
                    #elif TEST_FEATURES_NORM_MODE == 2 // Per feature and across patches normalization
                    minNorm = minFeatures[d];
                    maxNorm = minFeatures[d];
                    #elif TEST_FEATURES_NORM_MODE == 3 // Across features and across patches normalization
                    size_t nFeatures = fvPositiveSamples[0][0][0][0].size();
                    minNorm = FeatureVector(nFeatures, minFeatures[d]);
                    maxNorm = FeatureVector(nFeatures, maxFeatures[d]);
                    #endif/*ESVM_USE_FEATURES_NORMALIZATION == (1|2|3)*/

                    for (size_t pos = 0; pos < nPositives; ++pos)
                        for (size_t r = 0; r < nRepresentations; ++r)
                            fvPositiveSamples[pos][p][d][r] = normalizePerFeature(MIN_MAX, fvPositiveSamples[pos][p][d][r],
                                                                                  minNorm, maxNorm, ESVM_FEATURE_NORM_CLIP);
                    for (size_t neg = 0; neg < nNegatives; ++neg)
                        fvNegativeSamples[p][d][neg] = normalizePerFeature(MIN_MAX, fvNegativeSamples[p][d][neg],
                                                                           minNorm, maxNorm, ESVM_FEATURE_NORM_CLIP);
                    for (size_t prb = 0; prb < nProbes; ++prb)
                        fvProbeSamples[p][d][prb] = normalizePerFeature(MIN_MAX, fvProbeSamples[p][d][prb],
                                                                        minNorm, maxNorm, ESVM_FEATURE_NORM_CLIP);
                }
            }

            // ESVM samples files for each (sequence,positive,feature-extraction,train/test,patch) combination
            #if PROC_WRITE_DATA_FILES
            logger << "Writing ESVM train/test samples files..." << std::endl;
            for (size_t p = 0; p < nPatches; ++p)
            {
                std::string strPatch = std::to_string(p);
                for (size_t d = 0; d < nDescriptors; ++d)
                {
                    for (size_t pos = 0; pos < nPositives; ++pos)
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
                        for (size_t galleryPos = 0; galleryPos < nPositives; ++galleryPos)
                            for (size_t r = 0; r < nRepresentations; ++r)
                        {
                            int gt = (pos == galleryPos ? ESVM_POSITIVE_CLASS : ESVM_NEGATIVE_CLASS);
                            trainFile << featuresToSvmString(fvPositiveSamples[galleryPos][p][d][r], gt) << std::endl;
                        }
                        #else/*TEST_USE_OTHER_POSITIVES_AS_NEGATIVES*/
                        // Add only corresponding positive representations
                        for (size_t r = 0; r < nRepresentations; ++r)
                            trainFile << featuresToSvmString(fvPositiveSamples[pos][p][d][r], ESVM_POSITIVE_CLASS) << std::endl;
                        #endif/*TEST_USE_OTHER_POSITIVES_AS_NEGATIVES*/
                        for (size_t neg = 0; neg < nNegatives; ++neg)
                            trainFile << featuresToSvmString(fvNegativeSamples[p][d][neg], ESVM_NEGATIVE_CLASS) << std::endl;
                        for (size_t prb = 0; prb < nProbes; ++prb)
                            testFile << featuresToSvmString(fvProbeSamples[p][d][prb], probeGroundTruth[pos][prb]) << std::endl;
                    }
                }
            }
            #endif/*PROC_WRITE_DATA_FILES*/

            // Classifiers training and testing
            logger << "Starting classification training/testing..." << std::endl;
            for (size_t pos = 0; pos < nPositives; ++pos)
            {
                logger << "Starting for individual " << pos << ": " + positivesID[pos] << std::endl;
                std::vector<double> fusionScores(nProbes, 0.0);
                std::vector<double> combinedScores(nProbes, 0.0);
                std::vector<double> combinedScoresRaw(nProbes, 0.0);
                for (size_t d = 0; d < nDescriptors; ++d)
                {
                    std::vector<double> descriptorScores(nProbes, 0.0);
                    for (size_t p = 0; p < nPatches; ++p)
                    {
                        try
                        {
                            logger << "Running Exemplar-SVM training..." << std::endl;
                            esvmModels[pos][p][d] = ESVM(fvPositiveSamples[pos][p][d], fvNegativeSamples[p][d], positivesID[pos]);

                            #if PROC_WRITE_DATA_FILES
                            std::string esvmModelFile = "chokepoint-" + seq + "-id" + positivesID[pos] + "-" +
                                                        descriptorNames[d] + "-patch" + std::to_string(p) + ".model";
                            logger << "Saving Exemplar-SVM model to file..." << std::endl;
                            bool isSaved = esvmModels[pos][p][d].saveModelFile(esvmModelFile);
                            logger << std::string(isSaved ? "Saved" : "Failed to save") +
                                      " Exemplar-SVM model to file: '" + esvmModelFile + "'" << std::endl;
                            #endif/*PROC_WRITE_DATA_FILES*/
                        }
                        catch (const std::exception& e)
                        {
                            logger << e.what() << std::endl;
                            return passThroughDisplayTestStatus(__func__, -1);
                        }
                        catch (...)
                        {
                            logger << "Unexpected error thrown" << std::endl;
                            return passThroughDisplayTestStatus(__func__, -2);
                        }

                        logger << "Running Exemplar-SVM testing..." << std::endl;
                        std::vector<double> patchScores(nProbes, 0.0);
                        // test probes per patch and normalize scores
                        for (size_t prb = 0; prb < nProbes; ++prb)
                            patchScores[prb] = esvmModels[pos][p][d].predict(fvProbeSamples[p][d][prb]);
                        std::vector<double> patchScoresNorm = normalizeClassScores(MIN_MAX, patchScores, ESVM_SCORE_NORM_CLIP);

                        /*########################################### DEBUG */
                        std::string strPatch = std::to_string(p);
                        logger << "PATCH " + strPatch + " SCORES:      " << featuresToVectorString(patchScores) << std::endl;
                        logger << "PATCH " + strPatch + " SCORES NORM: " << featuresToVectorString(patchScoresNorm) << std::endl;
                        /*########################################### DEBUG */

                        for (size_t prb = 0; prb < nProbes; ++prb)
                        {
                            descriptorScores[prb] += patchScoresNorm[prb];  // accumulation with normalized scores for patch-based score fusion
                            combinedScores[prb] += patchScoresNorm[prb];    // accumulation of scores for (patch,descriptor)-based score fusion
                            combinedScoresRaw[prb] += patchScores[prb];     // accumulation of all probe scores without any pre-fusion normalization

                            std::string probeGT = (probeSamplesID[prb] == positivesID[pos] ? "positive" : "negative");
                            logger << "Score for patch " << p << " of probe " << prb << " (ID" << probeSamplesID[prb] << ", "
                                   << probeGT << "): " << patchScoresNorm[prb] << std::endl;
                        }
                    }
                    for (size_t prb = 0; prb < nProbes; ++prb)
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

                size_t nCombined = nDescriptors * nPatches;
                for (size_t prb = 0; prb < nProbes; ++prb)
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
                std::vector<double> combinedScoresNorm = normalizeClassScores(MIN_MAX, combinedScoresRaw, ESVM_SCORE_NORM_CLIP);
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

    #else/*ESVM_HAS_CHOKEPOINT*/
    return passThroughDisplayTestStatus(__func__, SKIPPED, "Missing required 'ESVM_HAS_CHOKEPOINT' definition.");
    #endif/*ESVM_HAS_CHOKEPOINT*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

/**************************************************************************************************************************
TEST DEFINITION

    Similar procedure as in 'proc_runSingleSamplePerPersonStillToVideo_FullChokePoint' but using pre-computed feature
    vectors stored in the data files.

    NB:
        Vectors depend on the configuration of images, patches, data duplication, feature extraction method, etc.
        Changing any configuration will require new data file to be generated by running "FullChokePoint" at least once.
**************************************************************************************************************************/
int proc_runSingleSamplePerPersonStillToVideo_DataFiles_WholeImage()
{
    #ifdef ESVM_HAS_CHOKEPOINT
    logstream logger(LOGGER_FILE);
    logger << "Running '" << __func__ << "' test..." << std::endl;

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
        std::vector<double> normScores = normalizeClassScores(MIN_MAX, scores);
        for (int prb = 0; prb < scores.size(); ++prb)
        {
            std::string probeGT = (probeGroundTruths[prb] > 0 ? "positive" : "negative");
            logger << "Score for probe " << prb << " (" << probeGT << "): " << scores[prb] << " | normalized: " << normScores[prb] << std::endl;
        }

        // Evaluate results
        eval_PerformanceClassificationScores(normScores, probeGroundTruths);
    }
    logger << "Test complete" << std::endl;

    #else/*ESVM_HAS_CHOKEPOINT*/
    return passThroughDisplayTestStatus(__func__, SKIPPED, "Missing required 'ESVM_HAS_CHOKEPOINT' definition.");
    #endif/*ESVM_HAS_CHOKEPOINT*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

/**************************************************************************************************************************
TEST DEFINITION

    Similar procedure as in 'proc_runSingleSamplePerPersonStillToVideo_FullChokePoint' but using pre-computed feature
    vectors stored in the data files.

    This test allows score fusion first for patch-based files, and then for descriptor-based files.

        S_pos* = ∑_d [ ∑_p [ s_(p,d) ] / N_p ] / N_d     ∀pos positives (targets), ∀d descriptor, ∀p patches

    NB:
        Vectors depend on the configuration of images, patches, data duplication, feature extraction method, etc.
        Changing any configuration will require new data files to be generated by running "FullChokePoint" at least once.
**************************************************************************************************************************/
int proc_runSingleSamplePerPersonStillToVideo_DataFiles_DescriptorAndPatchBased(size_t nPatches)
{
    #ifdef ESVM_HAS_CHOKEPOINT

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
            for (size_t p = 0; p < nPatches; ++p)
            {
                std::string strPatch = std::to_string(p);
                std::string trainFileName = dataFileDir + "chokepoint-S1-id0011-" + *d + "-patch" + strPatch + "-train.data";
                std::string testFileName = dataFileDir + "chokepoint-S1-id0011-" + *d + "-patch" + strPatch + "-test.data";

                // Train/test ESVM from files
                logger << "Training ESVM with data file: '" << trainFileName << "'..." << std::endl;
                ESVM esvm = ESVM(trainFileName, *posID);
                logger << "Testing ESVM with data file: '" << testFileName << "'..." << std::endl;
                std::vector<double> scores = esvm.predict(testFileName, &probeGroundTruths);
                std::vector<double> normScores = normalizeClassScores(MIN_MAX, scores);

                nProbes = scores.size();
                if (p == 0)
                {
                    // Initialize fusion scores accumulators on first patch / feature extraction method as required
                    patchFusionScores = std::vector<double>(nProbes, 0.0);
                    if (d == descriptorNames.begin())
                        descriptorFusionScores = std::vector<double>(nProbes, 0.0);
                }
                for (size_t prb = 0; prb < nProbes; ++prb)
                {
                    patchFusionScores[prb] += normScores[prb];          // Accumulation of patch-based scores
                    combinedFusionScores[prb] += normScores[prb];       // Accumulation of (patch,descriptor)-based scores

                    std::string probeGT = (probeGroundTruths[prb] > 0 ? "positive" : "negative");
                    logger << "Score for probe " << prb << " (" << probeGT << "): " << scores[prb]
                           << " | normalized: " << normScores[prb] << std::endl;
                }
            }

            for (size_t prb = 0; prb < nProbes; ++prb)
            {
                patchFusionScores[prb] /= (double)nPatches;             // Average of accumulated patch-based scores
                descriptorFusionScores[prb] += patchFusionScores[prb];  // Accumulation of feature-based scores
            }

            // Evaluate results per feature extraction method
            logger << "Performance evaluation for patch-based score fusion of '" + *d + "' descriptor:" << std::endl;
            eval_PerformanceClassificationScores(patchFusionScores, probeGroundTruths);
        }

        size_t nCombined = nDescriptors * nPatches;
        for (size_t prb = 0; prb < nProbes; ++prb)
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

    #else/*ESVM_HAS_CHOKEPOINT*/
    return passThroughDisplayTestStatus(__func__, SKIPPED, "Missing required 'ESVM_HAS_CHOKEPOINT' definition.");
    #endif/*ESVM_HAS_CHOKEPOINT*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

/**************************************************************************************************************************
TEST DEFINITION

    Use pre-generated negative samples data files (HOG-588 9-patches) and extract features from enroll still to train an
    Ensemble of Exemplar-SVM. Test against various probes from corresponding process pre-generated data files, and also
    against probes using feature extraction process with first ChokePoint sequence (mode 0).
**************************************************************************************************************************/
int proc_runSingleSamplePerPersonStillToVideo_NegativesDataFiles_PositivesExtraction_PatchBased()
{
    #if PROC_READ_DATA_FILES & 0b11110000
    #ifdef ESVM_HAS_CHOKEPOINT
    logstream logger(LOGGER_FILE);
    logger << "Running '" << __func__ << "' test..." << std::endl;

    // Set paths
    ASSERT_LOG(!(PROC_READ_DATA_FILES & 0b10000000) != !(PROC_READ_DATA_FILES & 0b01110000),
               "Invalid 'PROC_READ_DATA_FILES' options flag (128) cannot be used simultaneously with [(16),(32),(64)]");
    const std::string hogTypeFilesPreGen = (PROC_READ_DATA_FILES & 0b10000000) ? "-C++" : "-MATLAB";
    const std::string imageTypeFilesPreGen = (PROC_READ_DATA_FILES & 0b10110000) ? "" : "-transposed";
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
    for (size_t pos = 0; pos < nPositives; ++pos)
    {
        std::string stillPath = roiChokePointEnrollStillPath + "roi" + ChokePoint::getIndividualID(positivesID[pos], true) + ".tif";
        std::vector<cv::Mat> patches = imPreprocess(stillPath, imageSize, patchCounts, false, WINDOW_NAME, cv::IMREAD_GRAYSCALE);
        for (size_t p = 0; p < nPatches; ++p)
            fvPositiveSamples[pos][p] = hog.compute(patches[p]);
    }

    // Load probe images and extract features if required (PROC_READ_DATA_FILES & 16|128)
    #if PROC_READ_DATA_FILES & 0b10010000
    std::vector<ChokePoint::PortalTypes> types = ChokePoint::PORTAL_TYPES;
    bfs::directory_iterator endDir;
    std::string seq;
    int sn = 1;                                                 // session number
    for (int pn = 1; pn <= ChokePoint::PORTAL_QUANTITY; ++pn)   // portal number
    for (auto pt = types.begin(); pt != types.end(); ++pt)      // portal type
    for (int cn = 1; cn <= ChokePoint::CAMERA_QUANTITY; ++cn)   // camera number
    {
        seq = ChokePoint::getSequenceString(pn, *pt, sn, cn);
        logger << "Loading probe images and extracting features from sequence " << seq << "..." << std::endl;

        // Add ROI to corresponding sample vectors according to individual IDs
        for (int id = 1; id <= ChokePoint::INDIVIDUAL_QUANTITY; ++id)
        {
            std::string dirPath = roiChokePointCroppedFacePath + ChokePoint::getSequenceString(pn, *pt, sn, cn, id) + "/";
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
                            for (size_t p = 0; p < nPatches; ++p)
                                fvProbeLoadedSamples[p].push_back(hog.compute(patches[p]));

                            probesLoadedID.push_back(ChokePoint::getIndividualID(id, true));
                        }
                    }
                }
            }
        }
    }
    #endif/*PROC_READ_DATA_FILES & (16|128)*/
    cv::destroyWindow(WINDOW_NAME);

    // load negatives from pre-generated files
    size_t dimsNegatives[2]{ nPatches, 0 };                             // dynamically fill patches from file loading
    xstd::mvector<2, FeatureVector> fvNegativeSamples(dimsNegatives);   // [patch][negative](FeatureVector)
    for (int p = 0; p < nPatches; ++p)
    {
        std::string strPatch = std::to_string(p);
        std::string negativeTrainFile = negativesDir + "negatives-patch" + strPatch + ".data";
        logger << "Loading pre-generated negative samples file for patch " << strPatch << "..." << std::endl
               << "   Using file: '" << negativeTrainFile << "'" << std::endl;
        std::vector<FeatureVector> fvNegativeSamplesPatch;
        std::vector<int> negativeGroundTruths;
        ESVM::readSampleDataFile(negativeTrainFile, fvNegativeSamplesPatch, negativeGroundTruths);
        fvNegativeSamples[p] = xstd::mvector<1, FeatureVector>(fvNegativeSamplesPatch);
    }

    // execute feature normalization as required
    //    N.B. Pre-generated negatives and probes samples from files are already normalized
    #if TEST_FEATURES_NORM_MODE == 3
    logger << "Applying feature normalization (across features, across patches) for loaded positives and probes..." << std::endl;
    double hardcodedFoundMin = 0;               // Min found using 'FullChokePoint' test
    double hardcodedFoundMax = 0.675058;        // Max found using 'FullChokePoint' test
    size_t nProbesLoaded = fvProbeLoadedSamples[0].size();
    for (int p = 0; p < nPatches; ++p)
    {
        for (size_t pos = 0; pos < nPositives; ++pos)
            fvPositiveSamples[pos][p] = normalizeOverAll(MIN_MAX, fvPositiveSamples[pos][p],
                                                         hardcodedFoundMin, hardcodedFoundMax, ESVM_FEATURE_NORM_CLIP);
        for (size_t prb = 0; prb < nProbesLoaded; ++prb)
            fvProbeLoadedSamples[p][prb] = normalizeOverAll(MIN_MAX, fvProbeLoadedSamples[p][prb],
                                                            hardcodedFoundMin, hardcodedFoundMax, ESVM_FEATURE_NORM_CLIP);
    }
    #endif/*TEST_FEATURES_NORM_MODE == 3*/

    // train and test ESVM
    for (size_t pos = 0; pos < nPositives; ++pos)
    {
        std::vector<int> probeGroundTruthsPreGen, probeGroundTruthsLoaded;                                  // [probe](int)
        std::vector<double> probeFusionScoresPreGen, probeFusionScoresLoaded;                               // [probe](double)
        xstd::mvector<2, double> probePatchScoresPreGen(dimsProbes), probePatchScoresLoaded(dimsProbes);    // [patch][probe](double)
        std::string posID = ChokePoint::getIndividualID(positivesID[pos], true);
        logger << "Starting ESVM training/testing for '" << posID << "'..." << std::endl;
        for (size_t p = 0; p < nPatches; ++p)
        {
            // train with positive extracted features and negative loaded features
            std::string strPatch = std::to_string(p);
            logger << "Starting ESVM training for '" << posID << "', patch " << strPatch << "..." << std::endl;
            std::vector<FeatureVector> fvPositiveSingleSamplePatch = { fvPositiveSamples[pos][p] };
            std::vector<FeatureVector> fvNegativeSamplesPatch = std::vector<FeatureVector>(fvNegativeSamples[p]);
            esvm[pos][p] = ESVM(fvPositiveSingleSamplePatch, fvNegativeSamplesPatch, posID + "-" + strPatch);

            // test against pre-generated probes and loaded probes
            #if PROC_READ_DATA_FILES & 0b10010000   // (16|128) use feature extraction on probe images
            logger << "Starting ESVM testing for '" << posID << "', patch " << strPatch << " (probe images and extract feature)..." << std::endl;
            std::vector<double> scoresLoaded = esvm[pos][p].predict(fvProbeLoadedSamples[p]);
            probePatchScoresLoaded[p] = xstd::mvector<1, double>(scoresLoaded);
            #endif/*PROC_READ_DATA_FILES & (16|128)*/
            #if PROC_READ_DATA_FILES & 0b01100000   // (32|64) use pre-generated probe sample file
            std::string probePreGenTestFile = probesFileDir + "test-target" + posID + "-patch" + strPatch + ".data";
            logger << "Starting ESVM testing for '" << posID << "', patch " << strPatch << " (probe pre-generated samples files)..." << std::endl
                   << "   Using file: '" << probePreGenTestFile << "'" << std::endl;
            std::vector<double> scoresPreGen = esvm[pos][p].predict(probePreGenTestFile, &probeGroundTruthsPreGen);
            probePatchScoresPreGen[p] = xstd::mvector<1, double>(scoresPreGen);
            #endif/*PROC_READ_DATA_FILES & (32|64)*/
        }

        logger << "Starting score fusion and normalization for '" << posID << "'..." << std::endl;

        /* ---------------------------------------------------
           (16|128) use feature extraction on probe images
        --------------------------------------------------- */
        #if PROC_READ_DATA_FILES & 0b10010000

        // accumulated sum of scores for score fusion
        size_t nProbesLoaded = probePatchScoresLoaded[0].size();
        probeFusionScoresLoaded = std::vector<double>(nProbesLoaded, 0.0);
        for (size_t p = 0; p < nPatches; ++p)
            for (size_t prb = 0; prb < nProbesLoaded; ++prb)
                probeFusionScoresLoaded[prb] += probePatchScoresLoaded[p][prb];

        // average accumulated scores and execute post-fusion normalization
        // also find ground truths of feature vectors
        for (size_t prb = 0; prb < nProbesLoaded; ++prb)
        {
            probeGroundTruthsLoaded.push_back(probesLoadedID[prb] == posID ? ESVM_POSITIVE_CLASS : ESVM_NEGATIVE_CLASS);
            probeFusionScoresLoaded[prb] /= (double)nPatches;
        }
        probeFusionScoresLoaded = normalizeClassScores(MIN_MAX, probeFusionScoresLoaded, ESVM_SCORE_NORM_CLIP);

        // evaluate results with fusioned patch scores
        logger << "Performance evaluation for loaded/extracted probes (no pre-norm, post-fusion norm) of '" << posID << "':" << std::endl;
        eval_PerformanceClassificationScores(probeFusionScoresLoaded, probeGroundTruthsLoaded);

        #endif/*PROC_READ_DATA_FILES & (16|128)*/

        /* -------------------------------------------------------------------------------------------------------
           (32|64) use pre-generated probe sample file ([normal|transposed] images employed to generate files)
        ------------------------------------------------------------------------------------------------------- */
        #if PROC_READ_DATA_FILES & 0b01100000

        // accumulated sum of scores for score fusion
        size_t nProbesPreGen = probePatchScoresPreGen[0].size();
        probeFusionScoresPreGen = std::vector<double>(nProbesPreGen, 0.0);
        for (size_t p = 0; p < nPatches; ++p)
            for (size_t prb = 0; prb < nProbesPreGen; ++prb)
                probeFusionScoresPreGen[prb] += probePatchScoresPreGen[p][prb];

        // average accumulated scores and execute post-fusion normalization
        for (size_t prb = 0; prb < nProbesPreGen; ++prb)
            probeFusionScoresPreGen[prb] /= (double)nPatches;
        probeFusionScoresPreGen = normalizeClassScores(MIN_MAX, probeFusionScoresPreGen, ESVM_SCORE_NORM_CLIP);

        // evaluate results with fusioned patch scores
        logger << "Performance evaluation for pre-generated probes (no pre-norm, post-fusion norm) of '" << posID << "':" << std::endl;
        eval_PerformanceClassificationScores(probeFusionScoresPreGen, probeGroundTruthsPreGen);

        #endif/*PROC_READ_DATA_FILES & (32|64)*/
    }

    #else/*ESVM_HAS_CHOKEPOINT*/
    return passThroughDisplayTestStatus(__func__, SKIPPED, "Missing required 'ESVM_HAS_CHOKEPOINT' definition.");
    #endif/*ESVM_HAS_CHOKEPOINT*/
    #else/*PROC_READ_DATA_FILES & 0b11110000*/
    return passThroughDisplayTestStatus(__func__, SKIPPED);
    #endif/*PROC_READ_DATA_FILES & 0b11110000*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

/**************************************************************************************************************************
TEST DEFINITION

    Use person-based track ROIs obtained from (FD+FT) of 'Fast-DT + CompressiveTracking + 3 Haar Cascades' extracted from
    TITAN Unit's videos dataset to enroll stills with ESVM and test against probes under the same environment.
    Use negative samples from ChokePoint dataset across multiple camera angles.
**************************************************************************************************************************/
int proc_runSingleSamplePerPersonStillToVideo_TITAN(cv::Size imageSize, cv::Size patchCounts, bool useSyntheticPositives)
{
    #if PROC_ESVM_TITAN
    #ifdef ESVM_HAS_TITAN_UNIT
    logstream logger(LOGGER_FILE);
    logger << "Running '" << __func__ << "' test..." << std::endl;

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
    size_t nPatches = patchCounts.width * patchCounts.height;
    if (nPatches == 0) nPatches = 1;
    logger << "Starting single sample per person still-to-video full ChokePoint test..." << std::endl
           << "   useSyntheticPositives: " << useSyntheticPositives << std::endl
           << "   imageSize:             " << imageSize << std::endl
           << "   patchCounts:           " << patchCounts << std::endl
           << "   useHistEqual:          " << ESVM_USE_HIST_EQUAL << std::endl;

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
                Dimensions should therefore be mentioned explicitly using an array of size for each 'vector' level, be initialized later
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
    for (size_t pos = 0; pos < nPositives; ++pos)
    {
        // Add additional positive representations as requested
        if (useSyntheticPositives)
        {
            // Get original positive image with preprocessing but without patches splitting
            cv::Mat img = imPreprocess(positiveImageStills[pos].Path, imageSize, cv::Size(1,1),
                                       ESVM_USE_HIST_EQUAL, WINDOW_NAME, cv::IMREAD_COLOR)[0];
            // Get synthetic representations from original and apply patches splitting each one
            std::vector<cv::Mat> representations = imSyntheticGeneration(img);
            // Reinitialize sub-container for augmented representations using synthetic images
            nRepresentations = representations.size();
            size_t dimsRepresentation[2] { nRepresentations, nPatches };
            matPositiveSamples[pos] = xstd::mvector<2, cv::Mat>(dimsRepresentation);

            /// ############################################# #pragma omp parallel for
            for (size_t r = 0; r < nRepresentations; ++r)
            {
                std::vector<cv::Mat> patches = imSplitPatches(representations[r], patchCounts);
                for (size_t p = 0; p < nPatches; ++p)
                    matPositiveSamples[pos][r][p] = patches[p];
            }
        }
        // Only original representation otherwise (no synthetic images)
        else
        {
            //// matPositiveSamples[pos] = std::vector< std::vector< cv::Mat> >(1);
            std::vector<cv::Mat> patches = imPreprocess(positiveImageStills[pos].Path, imageSize, patchCounts,
                                                        ESVM_USE_HIST_EQUAL, WINDOW_NAME, cv::IMREAD_COLOR);
            for (size_t p = 0; p < nPatches; ++p)
                matPositiveSamples[pos][0][p] = patches[p];
        }
    }

    /// ################################################################################ DEBUG DISPLAY POSITIVES (+SYNTH)
    /*
    logger << "SHOWING DEBUG POSITIVE SAMPLES" << std::endl;
    for (int i = 0; i < nPositives; ++i)
    {
        for (int j = 0; j < matPositiveSamples[i].size(); ++j)
        {
            for (int k = 0; k < matPositiveSamples[i][j].size(); ++k)
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
    for (size_t pos = 0; pos < nPositives; ++pos)
    {
        /// ################################################## #pragma omp parallel for
        for (size_t p = 0; p < nPatches; ++p)
        {
            for (size_t d = 0; d < nDescriptors; ++d)
            {
                /// ################################################## #pragma omp parallel for
                for (int r = 0; r < nRepresentations; ++r)
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
                    for (size_t r = 0; r < nRepresentations; ++r)
                        for (size_t dup = 1; dup < nDuplications; ++dup)
                            fvPositiveSamples[pos][p][d].push_back(fvPositiveSamples[pos][p][d][r]);
            }
        }
    }
    nRepresentations *= nDuplications;
    for (size_t d = 0; d < nDescriptors; ++d)
        logger << "Features dimension (" + descriptorNames[d] + "): " << fvPositiveSamples[0][0][d][0].size() << std::endl;

    // Load negative samples from ChokePoint dataset
    std::vector<ChokePoint::PortalTypes> types = ChokePoint::PORTAL_TYPES;
    bfs::directory_iterator endDir;
    int sn = 1;                                                 // session number
    for (int pn = 1; pn <= ChokePoint::PORTAL_QUANTITY; ++pn) { // portal number
    for (auto pt = types.begin(); pt != types.end(); ++pt) {    // portal type
    for (int cn = 1; cn <= ChokePoint::CAMERA_QUANTITY; ++cn)   // camera number
    {
        std::string seq = ChokePoint::getSequenceString(pn, *pt, sn, cn);
        logger << "Loading negative and probe images for sequence " << seq << "..." << std::endl;

        // Add ROI to corresponding sample vectors
        for (int id = 1; id <= ChokePoint::INDIVIDUAL_QUANTITY; ++id)
        {
            std::string dirPath = roiChokePointCroppedFacePath + ChokePoint::getSequenceString(pn, *pt, sn, cn, id) + "/";
            if (bfs::is_directory(dirPath))
            {
                for (bfs::directory_iterator itDir(dirPath); itDir != endDir; ++itDir)
                {
                    if (bfs::is_regular_file(*itDir) && itDir->path().extension() == ".pgm")
                    {
                        size_t neg = matNegativeSamples.size();
                        matNegativeSamples.push_back(xstd::mvector<1, cv::Mat>(nPatches));
                        std::vector<cv::Mat> patches = imPreprocess(itDir->path().string(), imageSize, patchCounts,
                                                                    ESVM_USE_HIST_EQUAL, WINDOW_NAME, cv::IMREAD_GRAYSCALE);
                        for (size_t p = 0; p < nPatches; ++p)
                            matNegativeSamples[neg][p] = patches[p];
    } } } } } } }   // End of negatives loading

    // Load probe samples
    /*
    else if (contains(probesID, strID))
    {
        size_t prb = matProbeSamples.size();
        matProbeSamples.push_back(xstd::mvector<1, cv::Mat>(nPatches));
        std::vector<cv::Mat> patches = imPreprocess(itDir->path().string(), imageSize, patchCounts,
                                                    ESVM_USE_HIST_EQUAL, WINDOW_NAME, cv::IMREAD_GRAYSCALE);
        for (size_t p = 0; p < nPatches; ++p)
            matProbeSamples[prb][p] = patches[p];

        probeID.push_back(strID);
        for (size_t pos = 0; pos < nPositives; ++pos)
            probeGroundTruth[pos].push_back(strID == positivesID[pos] ? ESVM_POSITIVE_CLASS : ESVM_NEGATIVE_CLASS);
    }
    */

    #else/*ESVM_HAS_TITAN_UNIT*/
    return passThroughDisplayTestStatus(__func__, SKIPPED, "Missing required 'ESVM_HAS_TITAN_UNIT' definition.");
    #endif/*ESVM_HAS_TITAN_UNIT*/
    #else/*PROC_ESVM_TITAN*/
    return passThroughDisplayTestStatus(__func__, SKIPPED);
    #endif/*PROC_ESVM_TITAN*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

/**************************************************************************************************************************
TEST DEFINITION

    Uses pre-generated sample feature train/test files from SAMAN MATLAB code to enroll targets and test against probes.
    Parameters are pre-defined according to MATLAB code.
    Sample features are pre-extracted with HOG+PCA and normalized.
**************************************************************************************************************************/
int proc_runSingleSamplePerPersonStillToVideo_DataFiles_SAMAN()
{
    #if PROC_ESVM_SAMAN
    logstream logger(LOGGER_FILE);
    logger << "Running '" << __func__ << "' test..." << std::endl;

    std::vector<std::string> positivesID = { "ID0003", "ID0005", "ID0006", "ID0010", "ID0024" };
    size_t nPositives = positivesID.size();
    size_t nPatches = 9;
    size_t nProbes = 0;     // set when read from testing file
    #if PROC_ESVM_SAMAN == 1
    size_t nFeatures = 128;
    std::string dataFileDir = "data_SAMAN_48x48-MATLAB_HOG-PCA-descriptor+9-patches/";
    #elif PROC_ESVM_SAMAN == 2
    size_t nFeatures = 588;
    std::string dataFileDir = "data_SAMAN_48x48-MATLAB_HOG-descriptor+9-patches/";
    #elif PROC_ESVM_SAMAN == 3
    size_t nFeatures = 588;
    std::string dataFileDir = "data_SAMAN_48x48-MATLAB-transposed_HOG-descriptor+9-patches/";
    #elif PROC_ESVM_SAMAN == 4
    size_t nFeatures = 588;
    std::string dataFileDir = "data_ChokePoint_48x48_HOG-impl-588_9-patches_PreNormOverall-Mode3/";
    #endif/*PROC_ESVM_SAMAN*/

    size_t dimsESVM[2] = { nPositives, nPatches };
    xstd::mvector<2, ESVM> esvm(dimsESVM);          // [positive][patch](ESVM)

    for (size_t pos = 0; pos < nPositives; ++pos)
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
        for (size_t p = 0; p < nPatches; ++p)
        {
            // run training / testing from files
            std::vector<double> probePatchScores;
            std::string strPatch = std::to_string(p);
            #if PROC_ESVM_SAMAN == 4        // Using files generated by 'FullChokePoint' test
            std::string trainFile = dataFileDir + "chokepoint-S1-" + posID + "-hog-patch" + strPatch + "-train.data";
            std::string testFile = dataFileDir + "chokepoint-S1-" + posID + "-hog-patch" + strPatch + "-test.data";
            #else/*PROC_ESVM_SAMAN != 4*/   // Using files generated by the SAMAN MATLAB code
            std::string trainFile = dataFileDir + "train-target" + posID + "-patch" + strPatch + ".data";
            std::string testFile = dataFileDir + "test-target" + posID + "-patch" + strPatch + ".data";
            #endif/*PROC_ESVM_SAMAN*/
            logger << "Starting ESVM training from pre-generated file for '" << posID << "', patch " << strPatch << "..." << std::endl
                   << "   Using file: '" << trainFile << "'" << std::endl;
            esvm[pos][p] = ESVM(trainFile, posID);
            logger << "Starting ESVM testing from pre-generated file for '" << posID << "', patch " << strPatch << "..." << std::endl
                   << "   Using file: '" << testFile << "'" << std::endl;
            probePatchScores = esvm[pos][p].predict(testFile, &probeGroundTruths);

            // score normalization for patch
            std::vector<double> normPatchScores = normalizeClassScores(MIN_MAX, probePatchScores);
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
                for (size_t prb = 0; prb < nProbes; ++prb)
                    probeFusionScoresNormGradual[prb] += normPatchScores[prb];  // gradually accumulate normalized scores
        }

        ASSERT_LOG(nProbes > 0, "Number of probes should have been updated from loaded samples and be greater than zero");

        // score fusion of patches
        probeFusionScoresNormFinal = std::vector<double>(nProbes);
        probeFusionScoresNormSkipped = std::vector<double>(nProbes);
        for (size_t prb = 0; prb < nProbes; ++prb)
        {
            probeFusionScoresNormGradual[prb] /= (double)nProbes;               // average of gradually accumulated and normalized patch scores
            double probeScoresSum = 0, probeNormScoresSum = 0;
            std::vector<double> tmpProbeScores;
            for (size_t p = 0; p < nPatches; ++p)
            {
                tmpProbeScores.push_back(probeFusionScoresCumul[p][prb]);
                probeScoresSum += probeFusionScoresCumul[p][prb];               // accumulate across patch scores of corresponding probe
            }
            std::vector<double> normProbeScores = normalizeClassScores(MIN_MAX, tmpProbeScores);
            for (size_t p = 0; p < nPatches; ++p)
                probeNormScoresSum += normProbeScores[p];                       // accumulate across normalized patch scores of corresponding probe
            probeFusionScoresNormFinal[prb] = probeNormScoresSum / (double)nPatches;
            probeFusionScoresNormSkipped[prb] = probeScoresSum / (double)nPatches;
        }
        probeFusionScoresNormGradualPostNorm = normalizeClassScores(MIN_MAX, probeFusionScoresNormGradual);
        probeFusionScoresNormFinalPostNorm = normalizeClassScores(MIN_MAX, probeFusionScoresNormFinal);
        probeFusionScoresNormSkippedPostNorm = normalizeClassScores(MIN_MAX, probeFusionScoresNormSkipped);

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

    #else/*PROC_ESVM_SAMAN*/
    return passThroughDisplayTestStatus(__func__, SKIPPED);
    #endif/*PROC_ESVM_SAMAN*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

/**************************************************************************************************************************
TEST DEFINITION

    This test corresponds to the complete and working procedure to enroll and test image stills against pre-generated
    negative samples files from the ChokePoint dataset (session 1).
**************************************************************************************************************************/
int proc_runSingleSamplePerPersonStillToVideo_DataFiles_SimplifiedWorking()
{
    #if PROC_ESVM_SIMPLIFIED_WORKING
    logstream logger(LOGGER_FILE);
    logger << "Running '" << __func__ << "' test..." << std::endl;

    #if PROC_ESVM_SIMPLIFIED_WORKING == 1
    std::string sampleFileExt = ".data";
    FileFormat sampleFileFormat = LIBSVM;
    #elif PROC_ESVM_SIMPLIFIED_WORKING == 2
    std::string sampleFileExt = ".bin";
    FileFormat sampleFileFormat = BINARY;
    #else
    THROW("Unknown 'PROC_ESVM_SIMPLIFIED_WORKING' mode");
    #endif/*PROC_ESVM_SIMPLIFIED_WORKING*/

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
    xstd::mvector<2, double> minmaxClassificationScores(dimsResults);   // [positive][probe](double)
    xstd::mvector<2, double> zscoreClassificationScores(dimsResults);   // [positive][probe](double)
    xstd::mvector<2, int> probeGroundTruths(dimsResults);               // [positive][probe](int)

    // Exemplar-SVM
    xstd::mvector<2, ESVM> esvm(dimsPositives);                         // [patch][positive](ESVM)

    // prepare hog feature extractor
    cv::Size blockSize(2, 2);
    cv::Size blockStride(2, 2);
    cv::Size cellSize(2, 2);
    int nBins = 3;
    cv::Size windowSize = cv::Size(imageSize.width / patchCounts.width, imageSize.height / patchCounts.height);
    FeatureExtractorHOG hog(windowSize, blockSize, blockStride, cellSize, nBins);

    double hogRefMin = 0;            // Min found using 'FullChokePoint' test with SAMAN pre-generated files
    double hogRefMax = 0.675058;     // Max found using 'FullChokePoint' test with SAMAN pre-generated files
    /////////////////////////////////////////////////////// TESTING ///////////////////////////////////////////////////
    //double hogRefMax = 0.978284;
    /////////////////////////////////////////////////////// TESTING ///////////////////////////////////////////////////

    // load positive target still images, extract features and normalize
    logger << "Loading positive image stills, extracting feature vectors and normalizing..." << std::endl;
    for (size_t pos = 0; pos < nPositives; ++pos)
    {
        auto path = esvm::path::refStillImagesPath + "roi" + positivesID[pos] + ".tif";
        auto test = cv::imread(path);
        std::vector<cv::Mat> patches = imPreprocess(esvm::path::refStillImagesPath + "roi" + positivesID[pos] + ".tif",
                                                    imageSize, patchCounts, false, "", cv::IMREAD_GRAYSCALE, cv::INTER_LINEAR);
        for (size_t p = 0; p < nPatches; ++p)
            positiveSamples[p][pos] = normalizeOverAll(MIN_MAX, hog.compute(patches[p]), hogRefMin, hogRefMax, false);
    }

    // load negative samples from pre-generated files for training (samples in files are pre-normalized)
    logger << "Loading negative samples from files..." << std::endl;
    for (size_t p = 0; p < nPatches; ++p)
        ESVM::readSampleDataFile(negativeSamplesDir + "negatives-hog-patch" + std::to_string(p) + sampleFileExt,
                                 negativeSamples[p], sampleFileFormat);

    // load probe samples from pre-generated files for testing (samples in files are pre-normalized)
    logger << "Loading probe samples from files..." << std::endl;
    /////////////////////////////////////////////////////// ORIGINAL ==> TO REAPPLY AFTER TESTING ////////////////////
    for (size_t p = 0; p < nPatches; ++p)
        for (size_t pos = 0; pos < nPositives; ++pos)
            ESVM::readSampleDataFile(testingSamplesDir + positivesID[pos] + "-probes-hog-patch" + std::to_string(p) + sampleFileExt,
                                     probeSamples[p][pos], probeGroundTruths[pos], sampleFileFormat);

    /////////////////////////////////////////////////////// TESTING ///////////////////////////////////////////////////
    //#define INCLUDE_HARD_CASES 0;
    //#define INCLUDE_MEDIUM_CASES 0;
    //bfs::directory_iterator endDir;
    //-------------------------- FIRST TEST (Flash Disk) --------------------------
    //std::string fastDTProbePath = "F:/images/";
    //std::map<std::string, std::string> posGT{ { "person_0", "ID0003"}, {"person_2", ""}, {"person_3", "" }, {"person_4", "ID0005"},
    //                                          { "person_6", "ID0013"},{ "person_10", "ID0015" }
    //                                            #if INCLUDE_HARD_CASES
    //                                            , {"person_0_hard", "ID0003"},{ "person_4_hard", "ID0005" }
    //                                            #endif
    //                                        };
    //
    //-------------------------- SECOND TEST (FaceRecog Dir) --------------------------
    //std::string fastDTProbePath = "C:/Users/Francis/Programs/DEVELOPMENT/Face Recognition/ExemplarSVM-LIBSVM/bld/FAST_DT_IMAGES_RESIZE_NN/";
    //std::map<std::string, std::string> posGT{ { "person_1", "ID0005"},{ "person_3", "ID0003" }, {"person_4", "ID0019"},
    //                                          { "person_8", "ID0013"},{ "person_12", "ID0010" },{ "person_13", "ID0011" },{ "person_14", "ID0011" },
    //                                          { "person_19", "ID0010" },{ "person_20", "ID0012" },{ "person_21", "ID0020" },{ "person_23", "ID0024"},
    //                                          { "person_24", "ID0024" }
    //                                        #if INCLUDE_MEDIUM_CASES
    //                                        , {"person_1_medium", "ID0005"},{ "person_3_medium", "ID0003" },{ "person_8_medium", "ID0013" },
    //                                        { "person_12_medium", "ID0010" },{ "person_19_medium", "ID0010"},{ "person_20_medium", "ID0012" },
    //                                        { "person_21_medium", "ID0020" }
    //                                        #endif
    //                                        #if INCLUDE_HARD_CASES
    //                                        , {"person_1", "ID0005"}, {"person_2","ID0005"},{"person_3_hard","ID0003" },{"person_8_hard","ID0013"}
    //                                        ,{ "person_12_hard", "ID0010" },{ "person_16", "ID0011" },{ "person_19_hard", "ID0010" },
    //                                        { "person_21_hard", "ID0020" },{ "person_24_hard", "ID0024" }
    //                                        #endif
    //                                        };*/
    //-------------------------- THIRD TEST (FaceRecog+LocalSearchROI) --------------------------
    //std::string fastDTProbePath = "C:/Users/Francis/Programs/DEVELOPMENT/Face Recognition/ExemplarSVM-LIBSVM/bld/FAST_DT_IMAGES_LOCAL_SEARCH_P1E_S1_C1/";
    //std::vector<std::string> positivesDirs = { "0003", "0005", "0006", "0010", "0024" };
    //bfs::directory_iterator endDir;
    //#define SKIP_EXTRA_PERSON_DIR 1
    //#define SKIP_TRACK_PERSON_DIR 1
    //for (bfs::directory_iterator itPersDir(fastDTProbePath); itPersDir != endDir; ++itPersDir)  // 'persons' dirs
    //{
    //    if (bfs::is_directory(*itPersDir))
    //    {
    //        logger << "[DEBUG] dir: " << *itPersDir << std::endl;
    //        #if SKIP_EXTRA_PERSON_DIR
    //        if (itPersDir->path().filename() == "extra") {
    //            logger << "[DEBUG] 'extra' person dir skipped..." << std::endl;
    //            continue;
    //        }
    //        #endif/*SKIP_EXTRA_PERSON_DIR*/
    //        #if SKIP_TRACK_PERSON_DIR
    //        if (itPersDir->path().filename() == "persons") {
    //            logger << "[DEBUG] 'person_#' track dirs skipped..." << std::endl;
    //            continue;
    //        }
    //        #endif/*SKIP_TRACK_PERSON_DIR*/

    //        for (bfs::directory_iterator itPrb(*itPersDir); itPrb != endDir; ++itPrb)  // probes in 'person' dir
    //        {
    //            logger << "[DEBUG] img: " << *itPrb << std::endl;
    //            if (bfs::is_regular_file(*itPrb) && itPrb->path().extension() == ".png")
    //            {
    //                std::vector<cv::Mat> patches = imPreprocess(itPrb->path().string(), imageSize, patchCounts,
    //                                                            false, "", IMREAD_GRAYSCALE, INTER_LINEAR);
    //                for (size_t pos = 0; pos < nPositives; ++pos) {  // duplicate only to avoid full refactoring
    //                    for (size_t p = 0; p < nPatches; ++p)
    //                        probeSamples[p][pos].push_back(normalizeOverAll(MIN_MAX, hog.compute(patches[p]), hogRefMin, hogRefMax, false));
    //                    std::string prbId = itPersDir->path().stem().string();
    //                    bool isPos = positivesDirs[pos] == prbId;
    //                    logger << "[DEBUG] GT: " << prbId << " =?= " << positivesDirs[pos] << " | " << isPos << std::endl;
    //                    probeGroundTruths[pos].push_back(isPos ? 1 : -1);
    //                }
    //            }
    //        }
    //    }
    //}
    //double actualMin, actualMax, realMin = DBL_MAX, realMax = -DBL_MAX;
    //for (size_t pos = 0; pos < nPositives; ++pos)
    //    for (size_t p = 0; p < nPatches; ++p) {
    //        findNormParamsOverAll(MIN_MAX, probeSamples[p][pos], actualMin, actualMax);
    //        logger << "[DEBUG] (pos,p): " << pos << "," << p << " MIN: " << actualMin << " <? REFMIN " << hogRefMin << " " << (actualMin < hogRefMin)
    //               << " | MAX: " << actualMax << " >? REFMAX " << hogRefMax << " " << (actualMax > hogRefMax) << std::endl;
    //        if (realMin > actualMin)
    //            realMin = actualMin;
    //        if (realMax < actualMax)
    //            realMax = actualMax;
    //    }
    //logger << "REAL MIN: " << realMin << " REAL MAX: " << realMax << std::endl;
    /////////////////////////////////////////////////////// TESTING ///////////////////////////////////////////////////

    // training
    try {
    logger << "Training ESVM with positives and negatives..." << std::endl;
    for (size_t p = 0; p < nPatches; ++p)
        logger << "NEG p=" << p << ": " << negativeSamples[p].size() << std::endl;

    for (size_t p = 0; p < nPatches; ++p)
        for (size_t pos = 0; pos < nPositives; ++pos)
            esvm[p][pos] = ESVM({ positiveSamples[p][pos] }, negativeSamples[p], positivesID[pos] + "-patch" + std::to_string(p));
    }
    catch(std::exception&ex)
    { logger << "EXCPTION: " << ex.what() << std::endl; }

    // testing, score fusion, normalizatio    logger << "Testing probe samples against enrolled targets..." << std::endl;
    double minScore = DBL_MAX, maxScore = -DBL_MAX, meanScore = 0, stddevScore = 0, varScore = 0;
    std::vector<double> meanScorePerPatch(nPatches, 0.0), stddevScorePerPatch(nPatches, 0.0), varScorePerPatch(nPatches, 0.0);
    std::vector<size_t> nProbesPerPositive(nPositives, 0);
    for (size_t pos = 0; pos < nPositives; ++pos)
    {
        nProbesPerPositive[pos] = probeSamples[0][pos].size();   // variable number of probes according to tested positive
        classificationScores[pos] = xstd::mvector<1, double>(nProbesPerPositive[pos], 0.0);
        minmaxClassificationScores[pos] = xstd::mvector<1, double>(nProbesPerPositive[pos], 0.0);
        zscoreClassificationScores[pos] = xstd::mvector<1, double>(nProbesPerPositive[pos], 0.0);
        for (size_t prb = 0; prb < nProbesPerPositive[pos]; ++prb)
        {
            for (size_t p = 0; p < nPatches; ++p)
            {
                scores[p][pos].push_back( esvm[p][pos].predict(probeSamples[p][pos][prb]) );
                classificationScores[pos][prb] += scores[p][pos][prb];                          // score accumulation
            }
            classificationScores[pos][prb] /= (double)nPatches;                                 // average score fusion
            if (minScore > classificationScores[pos][prb])
                minScore = classificationScores[pos][prb];
            if (maxScore < classificationScores[pos][prb])
                maxScore = classificationScores[pos][prb];
        }
        // score normalization post-fusion
        minmaxClassificationScores[pos] = normalizeClassScores(MIN_MAX, classificationScores[pos], false);
        zscoreClassificationScores[pos] = normalizeClassScores(Z_SCORE, classificationScores[pos], false);
    }
    // mean / stddev evaluation
    std::vector<size_t> nTotalPatch(nPatches, 0);
    size_t nTotal = 0;
    for (size_t p = 0; p < nPatches; ++p)
    {
        for (size_t pos = 0; pos < nPositives; ++pos)
        {
            for (size_t prb = 0; prb < nProbesPerPositive[pos]; ++prb)
            {
                meanScore += scores[p][pos][prb];
                meanScorePerPatch[p] += scores[p][pos][prb];
                nTotal++;
                nTotalPatch[p]++;
            }
        }
        meanScorePerPatch[p] /= (double)nTotalPatch[p];
    }
    meanScore /= (double)nTotal;
    for (size_t p = 0; p < nPatches; ++p)
    {
        for (size_t pos = 0; pos < nPositives; ++pos)
        {
            for (size_t prb = 0; prb < nProbesPerPositive[pos]; ++prb)
            {
                varScore += (scores[p][pos][prb] - meanScore) * (scores[p][pos][prb] - meanScore);
                varScorePerPatch[p] += (scores[p][pos][prb] - meanScorePerPatch[p]) * (scores[p][pos][prb] - meanScorePerPatch[p]);
            }
        }
        varScorePerPatch[p] /= (double)nTotalPatch[p];
        stddevScorePerPatch[p] = std::sqrt(varScorePerPatch[p]);
    }
    varScore /= (double)nTotal;
    stddevScore = std::sqrt(varScore);

    // classification scores per target
    for (size_t pos = 0; pos < nPositives; ++pos)
    {
        logger << "Performance evaluation results (min-max normalization) for target " << positivesID[pos] << ":" << std::endl;
        eval_PerformanceClassificationScores(minmaxClassificationScores[pos], probeGroundTruths[pos]);
        logger << "Performance evaluation results (z-score normalization) for target " << positivesID[pos] << ":" << std::endl;
        eval_PerformanceClassificationScores(zscoreClassificationScores[pos], probeGroundTruths[pos]);
    }
    // normalization values
    logger << "Found min/max/mean/stddev/var classification fusion scores across all probes: "
           << minScore << ", " << maxScore << ", " << meanScore << ", " << stddevScore << ", " << varScore << std::endl;
    logger << "Found mean classification scores per patch across all probes:   " << featuresToVectorString(meanScorePerPatch) << std::endl;
    logger << "Found stddev classification scores per patch across all probes: " << featuresToVectorString(stddevScorePerPatch) << std::endl;
    logger << "Found var classification scores per patch across all probes:    " << featuresToVectorString(varScorePerPatch) << std::endl;
    // performance evaluation
    logger << "Summary of performance evaluation results (min-max normalization):" << std::endl;
    eval_PerformanceClassificationSummary(positivesID, minmaxClassificationScores, probeGroundTruths);
    logger << "Summary of performance evaluation results (z-score normalization):" << std::endl;
    eval_PerformanceClassificationSummary(positivesID, zscoreClassificationScores, probeGroundTruths);

    #else/*PROC_ESVM_SIMPLIFIED_WORKING*/
    return passThroughDisplayTestStatus(__func__, SKIPPED);
    #endif/*PROC_ESVM_SIMPLIFIED_WORKING*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

/**************************************************************************************************************************
TEST DEFINITION

/// TO VALIDATE
/// This test corresponds to the complete and working procedure to enroll and test image stills against pre-generated
/// negative samples files from the ChokePoint dataset(session 1).
///
**************************************************************************************************************************/
int proc_runSingleSamplePerPersonStillToVideo_FullGenerationAndTestProcess()
{
    #if PROC_ESVM_FULL_GENERATION_TESTING
    logstream logger(LOGGER_FILE);
    logger << "Running '" << __func__ << "' test..." << std::endl;

    // Extra/External Samples
    std::string samplePath_EX = "./FAST_DT_IMAGES_LOCAL_SEARCH_P1E_S1_C1/";
    ASSERT_LOG(bfs::is_directory(samplePath_EX), "Samples required for test");
    ASSERT_LOG(checkPathEndSlash(samplePath_EX), "Missing ending directory separator");

    /* -----------------------
        SAMPLES CATEGORIES
    ----------------------- */
    /* Training Targets:        single high quality still image for enrollment */
    std::vector<std::string> positivesID = { "0003", "0005", "0006", "0010", "0024" };
    /* Training Non-Targets:    as many video negatives as possible */
    std::vector<std::string> negativesID = { "0001", "0002", "0007", "0009", "0011",
                                             "0013", "0014", "0016", "0017", "0018",
                                             "0019", "0020", "0021", "0022", "0025" };
    /* Testing Probes:          some video positives and negatives */
    std::vector<std::string> probesID = { "0003", "0004", "0005", "0006", "0010",
                                          "0012", "0015", "0023", "0024", "0028" };

    /* -------------------------
        NORMALIZATION VALUES
    ------------------------- */
    double hogRefMin_S1 = 0;
    double hogRefMax_S1 = 1;
    double scoreRefMin_S1 = -4.91721;
    double scoreRefMax_S1 = 0.156316;
    double hogRefMin_S3 = 0;
    double hogRefMax_S3 = 1;
    double scoreRefMin_S3 = -5.15837;
    double scoreRefMax_S3 = 0.0505579;
    double hogRefMin_EX = 0;
    double hogRefMax_EX = 0.704862;
    double scoreRefMin_EX = -4.63064;
    double scoreRefMax_EX = 0.791653;

    try {

    /* --------------------------
        FEATURE EXTRACTOR HOG
    -------------------------- */
    logger << "Feature extractor HOG..." << std::endl;
    size_t nPatches = 9;
    cv::Size imageSize(48, 48);
    cv::Size patchSize(16, 16);
    cv::Size patchCounts(3, 3);
    cv::Size blockSize(2, 2);
    cv::Size blockStride(2, 2);
    cv::Size cellSize(2, 2);
    int nBins = 3;
    FeatureExtractorHOG hog(patchSize, blockSize, blockStride, cellSize, nBins);

    /* ------------------------------
        FEATURE VECTOR CONTAINERS
    ------------------------------ */
    logger << "Feature vector containers..." << std::endl;
    size_t nPositives = positivesID.size();
    size_t dimsGroundTruth[2]{ nPositives, 0 };
    size_t dimsPositives[2]{ nPositives, nPatches };
    xstd::mvector<2, FeatureVector> fvPositiveSamples(dimsPositives);       // [positive][patch](FeatureVector)
    xstd::mvector<2, FeatureVector> fvNegativeSamples;                      // [negative][patch](FeatureVector)
    xstd::mvector<2, FeatureVector> fvProbeSamples_S1;                      // [probe][patch](FeatureVector)
    xstd::mvector<2, FeatureVector> fvProbeSamples_S3;                      // [probe][patch](FeatureVector)
    xstd::mvector<2, FeatureVector> fvProbeSamples_EX;                      // [probe][patch](FeatureVector)
    xstd::mvector<2, int> probeGroundTruth_S1(dimsGroundTruth);             // [positive][probe](int)
    xstd::mvector<2, int> probeGroundTruth_S3(dimsGroundTruth);             // [positive][probe](int)
    xstd::mvector<2, int> probeGroundTruth_EX(dimsGroundTruth);             // [positive][probe](int)
    std::vector<std::string> negativeSamplesID;                             // [negative](string)
    std::vector<std::string> probeSamplesID_S1;                             // [probe](string)
    std::vector<std::string> probeSamplesID_S3;                             // [probe](string)
    std::vector<std::string> probeSamplesID_EX;                             // [probe](string)

    /* --------------------------
        LOAD SAMPLES FROM ROI
    -------------------------- */
    logger << "Load samples from ROI..." << std::endl;
    std::vector<ChokePoint::PortalTypes> types = ChokePoint::PORTAL_TYPES;
    bfs::directory_iterator endDir;
    cv::namedWindow(WINDOW_NAME);

    // optional CascadeClassifier for localized search, LBP Cascade focuses on more localized image representation (less background)
    cv::CascadeClassifier ccFaceLocalSearch;
    #if ESVM_ROI_PREPROCESS_MODE == 1
        std::string faceCascadeFilePath = sourcesOpenCV + "data/lbpcascades/lbpcascade_frontalface_improved.xml";
        assert(bfs::is_regular_file(faceCascadeFilePath));
        assert(ccFaceLocalSearch.load(faceCascadeFilePath));
    #endif/*ESVM_ROI_PREPROCESS_MODE == 1*/

    // load positives
    for (size_t pos = 0; pos < nPositives; ++pos) {
        cv::Mat roi = imReadAndDisplay(esvm::path::refStillImagesPath + "roiID" + positivesID[pos] + ".tif", "", cv::IMREAD_GRAYSCALE);
        roi = preprocessFromMode(roi, ccFaceLocalSearch);
        std::vector<cv::Mat> patches = imPreprocess(roi, imageSize, patchCounts, ESVM_USE_HIST_EQUAL, WINDOW_NAME);
        for (size_t p = 0; p < nPatches; ++p)
            fvPositiveSamples[pos][p] = hog.compute(patches[p]);
    }

    // load negatives / probes from ChokePoint
    int s1 = 1, s3 = 3;
    for (int pn = 1; pn <= ChokePoint::PORTAL_QUANTITY; ++pn) { // portal number
    for (auto pt = types.begin(); pt != types.end(); ++pt) {    // portal type
    for (int cn = 1; cn <= ChokePoint::CAMERA_QUANTITY; ++cn)   // camera number
    {
        // for each individual in S1 (negatives + probes)
        for (int id = 1; id <= ChokePoint::INDIVIDUAL_QUANTITY; ++id) {
            std::string dirPath = roiChokePointCroppedFacePath + ChokePoint::getSequenceString(pn, *pt, s1, cn, id) + "/";
            if (bfs::is_directory(dirPath)) {
                for (bfs::directory_iterator itDir(dirPath); itDir != endDir; ++itDir) {
                    if (bfs::is_regular_file(*itDir) && itDir->path().extension() == ".pgm") {
                        std::string strID = ChokePoint::getIndividualID(id);
                        // negatives
                        if (contains(negativesID, strID)) {
                            size_t neg = fvNegativeSamples.size();
                            fvNegativeSamples.push_back(xstd::mvector<1, FeatureVector>(nPatches));
                            cv::Mat roi = imReadAndDisplay(itDir->path().string(), "", cv::IMREAD_GRAYSCALE);
                            roi = preprocessFromMode(roi, ccFaceLocalSearch);
                            std::vector<cv::Mat> patches = imPreprocess(roi, imageSize, patchCounts, ESVM_USE_HIST_EQUAL, WINDOW_NAME);
                            for (size_t p = 0; p < nPatches; ++p)
                                fvNegativeSamples[neg][p] = hog.compute(patches[p]);
                            negativeSamplesID.push_back(strID);
                        }
                        // probes
                        else if (contains(probesID, strID)) {
                            size_t prb = fvProbeSamples_S1.size();
                            fvProbeSamples_S1.push_back(xstd::mvector<1, FeatureVector>(nPatches));
                            cv::Mat roi = imReadAndDisplay(itDir->path().string(), "", cv::IMREAD_GRAYSCALE);
                            roi = preprocessFromMode(roi, ccFaceLocalSearch);
                            std::vector<cv::Mat> patches = imPreprocess(roi, imageSize, patchCounts, ESVM_USE_HIST_EQUAL, WINDOW_NAME);
                            for (size_t p = 0; p < nPatches; ++p)
                                fvProbeSamples_S1[prb][p] = hog.compute(patches[p]);
                            probeSamplesID_S1.push_back(strID);
                            for (size_t pos = 0; pos < nPositives; ++pos)
                                probeGroundTruth_S1[pos].push_back(strID == positivesID[pos] ? ESVM_POSITIVE_CLASS : ESVM_NEGATIVE_CLASS);
        } } } } }   // end for each individual in S1

        // for each individual in S3 (only other probes)
        for (int id = 1; id <= INDIVIDUAL_QUANTITY; ++id) {
            std::string dirPath = roiChokePointCroppedFacePath + ChokePoint::getSequenceString(pn, *pt, s3, cn, id) + "/";
            if (bfs::is_directory(dirPath)) {
                for (bfs::directory_iterator itDir(dirPath); itDir != endDir; ++itDir) {
                    if (bfs::is_regular_file(*itDir) && itDir->path().extension() == ".pgm") {
                        std::string strID = ChokePoint::getIndividualID(id);
                        // probes
                        if (contains(probesID, strID)) {
                            size_t prb = fvProbeSamples_S3.size();
                            fvProbeSamples_S3.push_back(xstd::mvector<1, FeatureVector>(nPatches));
                            cv::Mat roi = imReadAndDisplay(itDir->path().string(), "", cv::IMREAD_GRAYSCALE);
                            roi = preprocessFromMode(roi, ccFaceLocalSearch);
                            std::vector<cv::Mat> patches = imPreprocess(roi, imageSize, patchCounts, ESVM_USE_HIST_EQUAL, WINDOW_NAME);
                            for (size_t p = 0; p < nPatches; ++p)
                                fvProbeSamples_S3[prb][p] = hog.compute(patches[p]);
                            probeSamplesID_S3.push_back(strID);
                            for (size_t pos = 0; pos < nPositives; ++pos)
                                probeGroundTruth_S3[pos].push_back(strID == positivesID[pos] ? ESVM_POSITIVE_CLASS : ESVM_NEGATIVE_CLASS);
        } } } } }   // end for each individual in S3

    } } }   // end chokepoint loops

    // for each individual in extra/external samples (extra probes)
    for (int id = 1; id <= INDIVIDUAL_QUANTITY; ++id) {
        std::string strID = ChokePoint::getIndividualID(id, false);
        bfs::path idDir(samplePath_EX + strID);
        // probes
        if (bfs::is_directory(idDir) && contains(probesID, strID)) {
            for (bfs::directory_iterator itDir(idDir); itDir != endDir; ++itDir) {
                if (bfs::is_regular_file(*itDir) && itDir->path().extension() == ".png") {
                    size_t prb = fvProbeSamples_EX.size();
                    fvProbeSamples_EX.push_back(xstd::mvector<1, FeatureVector>(nPatches));
                    cv::Mat roi = imReadAndDisplay(itDir->path().string(), "", cv::IMREAD_GRAYSCALE);
                    roi = preprocessFromMode(roi, ccFaceLocalSearch);
                    std::vector<cv::Mat> patches = imPreprocess(roi, imageSize, patchCounts, ESVM_USE_HIST_EQUAL, WINDOW_NAME);
                    for (size_t p = 0; p < nPatches; ++p)
                        fvProbeSamples_EX[prb][p] = hog.compute(patches[p]);
                    probeSamplesID_EX.push_back(strID);
                    for (size_t pos = 0; pos < nPositives; ++pos)
                        probeGroundTruth_EX[pos].push_back(strID == positivesID[pos] ? ESVM_POSITIVE_CLASS : ESVM_NEGATIVE_CLASS);
    } } } }   // end for each individual in extra/external samples
    cv::destroyWindow(WINDOW_NAME);

    size_t nNegatives = fvNegativeSamples.size();
    size_t nProbes_S1 = fvProbeSamples_S1.size();
    size_t nProbes_S3 = fvProbeSamples_S3.size();
    size_t nProbes_EX = fvProbeSamples_EX.size();

    // obtain min/max features (not normalized)
    double minFeatS1 = DBL_MAX, maxFeatS1 = -DBL_MAX, minFeatS3 = DBL_MAX, maxFeatS3 = -DBL_MAX, minFeatEX = DBL_MAX, maxFeatEX = -DBL_MAX;
    for (size_t f = 0; f < hog.getFeatureCount(); ++f) {
        for (size_t p = 0; p < nPatches; ++p) {
            for (size_t prb = 0; prb < nProbes_S1; ++prb) {
                if (minFeatS1 > fvProbeSamples_S1[prb][p][f])
                    minFeatS1 = fvProbeSamples_S1[prb][p][f];
                if (maxFeatS1 < fvProbeSamples_S1[prb][p][f])
                    maxFeatS1 = fvProbeSamples_S1[prb][p][f];
            }
            for (size_t prb = 0; prb < nProbes_S3; ++prb) {
                if (minFeatS3 > fvProbeSamples_S3[prb][p][f])
                    minFeatS3 = fvProbeSamples_S3[prb][p][f];
                if (maxFeatS3 < fvProbeSamples_S3[prb][p][f])
                    maxFeatS3 = fvProbeSamples_S3[prb][p][f];
            }
            for (size_t prb = 0; prb < nProbes_EX; ++prb) {
                if (minFeatEX > fvProbeSamples_EX[prb][p][f])
                    minFeatEX = fvProbeSamples_EX[prb][p][f];
                if (maxFeatEX < fvProbeSamples_EX[prb][p][f])
                    maxFeatEX = fvProbeSamples_EX[prb][p][f];
            }
        }
    }

    /* ------------------------------
        NORMALIZE FEATURE VECTORS
    ------------------------------ */
    logger << "Normalize feature vectors..." << std::endl;
    size_t dimsPosNorm[2]{ nPatches, nPositives };
    size_t dimsNegNorm[2]{ nPatches, nNegatives };
    size_t dimsPrbNorm_S1[2]{ nPatches, nProbes_S1 };
    size_t dimsPrbNorm_S3[2]{ nPatches, nProbes_S3 };
    size_t dimsPrbNorm_EX[2]{ nPatches, nProbes_EX };
    xstd::mvector<2, FeatureVector> fvPosNorm_S1(dimsPosNorm);                  // [positive][patch](FeatureVector)
    xstd::mvector<2, FeatureVector> fvPosNorm_S3(dimsPosNorm);                  // [positive][patch](FeatureVector)
    xstd::mvector<2, FeatureVector> fvPosNorm_EX(dimsPosNorm);                  // [positive][patch](FeatureVector)
    xstd::mvector<2, FeatureVector> fvNegNorm_S1(dimsNegNorm);                  // [negative][patch](FeatureVector)
    xstd::mvector<2, FeatureVector> fvNegNorm_S3(dimsNegNorm);                  // [negative][patch](FeatureVector)
    xstd::mvector<2, FeatureVector> fvNegNorm_EX(dimsNegNorm);                  // [negative][patch](FeatureVector)
    xstd::mvector<2, FeatureVector> fvPrbNorm_S1(dimsPrbNorm_S1);               // [probe][patch](FeatureVector)
    xstd::mvector<2, FeatureVector> fvPrbNorm_S3(dimsPrbNorm_S3);               // [probe][patch](FeatureVector)
    xstd::mvector<2, FeatureVector> fvPrbNorm_EX(dimsPrbNorm_EX);               // [probe][patch](FeatureVector)

    for (size_t p = 0; p < nPatches; ++p) {             // flip [patch] <==> [pos|neg|prb] order
        for (size_t pos = 0; pos < nPositives; ++pos) {
            fvPosNorm_S1[p][pos] = normalizeOverAll(MIN_MAX, fvPositiveSamples[pos][p], hogRefMin_S1, hogRefMax_S1, ESVM_FEATURE_NORM_CLIP);
            fvPosNorm_S3[p][pos] = normalizeOverAll(MIN_MAX, fvPositiveSamples[pos][p], hogRefMin_S3, hogRefMax_S3, ESVM_FEATURE_NORM_CLIP);
            fvPosNorm_EX[p][pos] = normalizeOverAll(MIN_MAX, fvPositiveSamples[pos][p], hogRefMin_EX, hogRefMax_EX, ESVM_FEATURE_NORM_CLIP);
        }
        for (size_t neg = 0; neg < nNegatives; ++neg) {
            fvNegNorm_S1[p][neg] = normalizeOverAll(MIN_MAX, fvNegativeSamples[neg][p], hogRefMin_S1, hogRefMax_S1, ESVM_FEATURE_NORM_CLIP);
            fvNegNorm_S3[p][neg] = normalizeOverAll(MIN_MAX, fvNegativeSamples[neg][p], hogRefMin_S3, hogRefMax_S3, ESVM_FEATURE_NORM_CLIP);
            fvNegNorm_EX[p][neg] = normalizeOverAll(MIN_MAX, fvNegativeSamples[neg][p], hogRefMin_EX, hogRefMax_EX, ESVM_FEATURE_NORM_CLIP);
        }
        for (size_t prb = 0; prb < nProbes_S1; ++prb)
            fvPrbNorm_S1[p][prb] = normalizeOverAll(MIN_MAX, fvProbeSamples_S1[prb][p], hogRefMin_S1, hogRefMax_S1, ESVM_FEATURE_NORM_CLIP);
        for (size_t prb = 0; prb < nProbes_S3; ++prb)
            fvPrbNorm_S3[p][prb] = normalizeOverAll(MIN_MAX, fvProbeSamples_S3[prb][p], hogRefMin_S3, hogRefMax_S3, ESVM_FEATURE_NORM_CLIP);
        for (size_t prb = 0; prb < nProbes_EX; ++prb)
            fvPrbNorm_EX[p][prb] = normalizeOverAll(MIN_MAX, fvProbeSamples_EX[prb][p], hogRefMin_EX, hogRefMax_EX, ESVM_FEATURE_NORM_CLIP);
    }

    /* ------------------
        ESVM TRAINING
    ------------------ */
    logger << "ESVM training..." << std::endl;
    size_t dimsESVM[2]{ nPositives, nPatches };
    xstd::mvector<2, ESVM> esvmModels(dimsESVM);                                // [target][patch](ESVM)
    for (size_t pos = 0; pos < nPositives; ++pos)
        for (size_t p = 0; p < nPatches; ++p)
            esvmModels[pos][p] = ESVM({ fvPosNorm_S1[p][pos] }, fvNegNorm_S1[p], positivesID[pos] + "-patch" + std::to_string(p));

    //////////////////////////////////////////////////////
    //// sample files produce different min/max score values
    //scoreRefMin = -0.802784;
    //scoreRefMax = 1.03869;
    //// load files and train ESVM with them
    //size_t dimsNegSampleFromFile[2]{ nPatches, 0 };
    //xstd::mvector<2, FeatureVector> fvNegFromFile(dimsNegSampleFromFile);       // [patch][negative](FeatureVector)
    //for (size_t p = 0; p < nPatches; ++p)                                       // "pre-genrated & normalized negative samples
    //    ESVM::readSampleDataFile(negativeSamplesDir + "negatives-hog-patch" + std::to_string(p) + ".bin", fvNegFromFile[p], BINARY);
    //for (size_t pos = 0; pos < nPositives; ++pos)
    //    for (size_t p = 0; p < nPatches; ++p)
    //        esvmModels[pos][p] = ESVM({ fvPosNorm[p][pos] }, fvNegFromFile[p], positivesID[pos] + "-patch" + std::to_string(p));
    ///////////////////////////////////////////////////////

    /* -----------------
        ESVM TESTING
    ----------------- */
    logger << "ESVM testing..." << std::endl;
    size_t dimsPrbPatchScores_S1[3]{ nPositives, nPatches, nProbes_S1 };
    size_t dimsPrbPatchScores_S3[3]{ nPositives, nPatches, nProbes_S3 };
    size_t dimsPrbPatchScores_EX[3]{ nPositives, nPatches, nProbes_EX };
    size_t dimsPrbScores_S1[2]{ nPositives, nProbes_S1 };
    size_t dimsPrbScores_S3[2]{ nPositives, nProbes_S3 };
    size_t dimsPrbScores_EX[2]{ nPositives, nProbes_EX };
    xstd::mvector<3, double> prbPatchScores_S1(dimsPrbPatchScores_S1);          // [positive][patch][probe]
    xstd::mvector<3, double> prbPatchScores_S3(dimsPrbPatchScores_S3);          // [positive][patch][probe]
    xstd::mvector<3, double> prbPatchScores_EX(dimsPrbPatchScores_EX);          // [positive][patch][probe]
    xstd::mvector<2, double> prbScores_S1(dimsPrbScores_S1, 0);                 // [positive][patch][probe]
    xstd::mvector<2, double> prbScores_S3(dimsPrbScores_S3, 0);                 // [positive][patch][probe]
    xstd::mvector<2, double> prbScores_EX(dimsPrbScores_EX, 0);                 // [positive][patch][probe]

    // predict scores
    for (size_t pos = 0; pos < nPositives; ++pos) {
        for (size_t p = 0; p < nPatches; ++p) {
            prbPatchScores_S1[pos][p] = esvmModels[pos][p].predict(fvPrbNorm_S1[p]);
            prbPatchScores_S3[pos][p] = esvmModels[pos][p].predict(fvPrbNorm_S3[p]);
            prbPatchScores_EX[pos][p] = esvmModels[pos][p].predict(fvPrbNorm_EX[p]);
        }
    }

    // obtain min/max scores (not normalized)
    double minS1 = DBL_MAX, maxS1 = -DBL_MAX, minS3 = DBL_MAX, maxS3 = -DBL_MAX, minEX = DBL_MAX, maxEX = -DBL_MAX;
    for (size_t pos = 0; pos < nPositives; ++pos) {
        for (size_t p = 0; p < nPatches; ++p) {
            for (size_t prb = 0; prb < nProbes_S1; ++prb) {
                if (minS1 > prbPatchScores_S1[pos][p][prb])
                    minS1 = prbPatchScores_S1[pos][p][prb];
                if (maxS1 < prbPatchScores_S1[pos][p][prb])
                    maxS1 = prbPatchScores_S1[pos][p][prb];
            }
            for (size_t prb = 0; prb < nProbes_S3; ++prb) {
                if (minS3 > prbPatchScores_S3[pos][p][prb])
                    minS3 = prbPatchScores_S3[pos][p][prb];
                if (maxS3 < prbPatchScores_S3[pos][p][prb])
                    maxS3 = prbPatchScores_S3[pos][p][prb];
            }
            for (size_t prb = 0; prb < nProbes_EX; ++prb) {
                if (minEX > prbPatchScores_EX[pos][p][prb])
                    minEX = prbPatchScores_EX[pos][p][prb];
                if (maxEX < prbPatchScores_EX[pos][p][prb])
                    maxEX = prbPatchScores_EX[pos][p][prb];
            }
        }
    }

    // score normalization and patches fusion
    for (size_t pos = 0; pos < nPositives; ++pos) {
        for (size_t p = 0; p < nPatches; ++p) {
            for (size_t prb = 0; prb < nProbes_S1; ++prb)
                prbScores_S1[pos][prb] += normalize(MIN_MAX, prbPatchScores_S1[pos][p][prb],
                                                    scoreRefMin_S1, scoreRefMax_S1, ESVM_SCORE_NORM_CLIP);
            for (size_t prb = 0; prb < nProbes_S3; ++prb)
                prbScores_S3[pos][prb] += normalize(MIN_MAX, prbPatchScores_S3[pos][p][prb],
                                                    scoreRefMin_S3, scoreRefMax_S3, ESVM_SCORE_NORM_CLIP);
            for (size_t prb = 0; prb < nProbes_EX; ++prb)
                prbScores_EX[pos][prb] += normalize(MIN_MAX, prbPatchScores_EX[pos][p][prb],
                                                    scoreRefMin_EX, scoreRefMax_EX, ESVM_SCORE_NORM_CLIP);
        }
        for (size_t prb = 0; prb < nProbes_S1; ++prb)
            prbScores_S1[pos][prb] /= (double)nPatches;
        for (size_t prb = 0; prb < nProbes_S3; ++prb)
            prbScores_S3[pos][prb] /= (double)nPatches;
        for (size_t prb = 0; prb < nProbes_EX; ++prb)
            prbScores_EX[pos][prb] /= (double)nPatches;
    }

    // calculate performance results
    size_t nResultType = 3;    // 3x for FPR,TPR,PPV results
    size_t dimsResults[3]{ nPositives, nResultType, 0 };
    xstd::mvector<3, double> results_S1(dimsResults), results_S3(dimsResults), results_EX(dimsResults);
    for (size_t pos = 0; pos < nPositives; ++pos) {
        eval_PerformanceClassificationScores(prbScores_S1[pos], probeGroundTruth_S1[pos],
                                             results_S1[pos][0], results_S1[pos][1], results_S1[pos][2]);
        eval_PerformanceClassificationScores(prbScores_S3[pos], probeGroundTruth_S3[pos],
                                             results_S3[pos][0], results_S3[pos][1], results_S3[pos][2]);
        eval_PerformanceClassificationScores(prbScores_EX[pos], probeGroundTruth_EX[pos],
                                             results_EX[pos][0], results_EX[pos][1], results_EX[pos][2]);
    }
    size_t nThresholds = results_S1[0][0].size();

    // display results (combined)
    logger << "Results combined details for all individuals (ChokePoint S1):" << std::endl;
    for (size_t t = 0; t < nThresholds; ++t) {
        logger << "(FPR,TPR,PPV)[" << t << "] = ";
        for (size_t pos = 0; pos < nPositives; ++pos)
            logger << results_S1[pos][0][t] << ", " << results_S1[pos][1][t] << ", " << results_S1[pos][2][t] << ", ";
        logger << std::endl;
    }
    logger << "Results combined details for all individuals (ChokePoint S3):" << std::endl;
    for (size_t t = 0; t < nThresholds; ++t) {
        logger << "(FPR,TPR,PPV)[" << t << "] = ";
        for (size_t pos = 0; pos < nPositives; ++pos)
            logger << results_S3[pos][0][t] << ", " << results_S3[pos][1][t] << ", " << results_S3[pos][2][t] << ", ";
        logger << std::endl;
    }
    logger << "Results combined details for all individuals (Fast-DT):" << std::endl;
    for (size_t t = 0; t < nThresholds; ++t) {
        logger << "(FPR,TPR,PPV)[" << t << "] = ";
        for (size_t pos = 0; pos < nPositives; ++pos)
            logger << results_EX[pos][0][t] << ", " << results_EX[pos][1][t] << ", " << results_EX[pos][2][t] << ", ";
        logger << std::endl;
    }

    // display results (summary)
    logger << "NORMALIZATION VALUES SPECIFIED:" << std::endl
        << "  S1 - hogRef - Min/Max:   " << hogRefMin_S1 << "/" << hogRefMax_S1 << std::endl
        << "  S3 - hogRef - Min/Max:   " << hogRefMin_S3 << "/" << hogRefMax_S3 << std::endl
        << "  EX - scoreRef - Min/Max: " << hogRefMin_EX << "/" << hogRefMax_EX << std::endl
        << "  S1 - scoreRef - Min/Max: " << scoreRefMin_S1 << "/" << scoreRefMax_S1 << std::endl
        << "  S3 - scoreRef - Min/Max: " << scoreRefMin_S3 << "/" << scoreRefMax_S3 << std::endl
        << "  EX - scoreRef - Min/Max: " << scoreRefMin_EX << "/" << scoreRefMax_EX << std::endl;
    logger << "FEATURES NORMALIZATION VALUES FOUND" << std::endl
        << "  S1 - Min/Max: " << minFeatS1 << "/" << maxFeatS1 << std::endl
        << "  S3 - Min/Max: " << minFeatS3 << "/" << maxFeatS3 << std::endl
        << "  EX - Min/Max: " << minFeatEX << "/" << maxFeatEX << std::endl;
    logger << "SCORES NORMALIZATION VALUES FOUND" << std::endl
        << "  S1 - Min/Max: " << minS1 << "/" << maxS1 << std::endl
        << "  S3 - Min/Max: " << minS3 << "/" << maxS3 << std::endl
        << "  EX - Min/Max: " << minEX << "/" << maxEX << std::endl;
    logger << "Results summary for samples from ChokePoint S1:" << std::endl;
    eval_PerformanceClassificationSummary(positivesID, prbScores_S1, probeGroundTruth_S1);
    logger << "Results summary for samples from ChokePoint S3:" << std::endl;
    eval_PerformanceClassificationSummary(positivesID, prbScores_S3, probeGroundTruth_S3);
    logger << "Results summary for samples from Fast-DT (extra/external):" << std::endl;
    eval_PerformanceClassificationSummary(positivesID, prbScores_EX, probeGroundTruth_EX);

    } // end try
    catch (std::exception& ex) {
        logger << ex.what() << std::endl;
        return passThroughDisplayTestStatus(__func__, -1);
    }

    #else/*PROC_ESVM_FULL_GENERATION_TESTING*/
    return passThroughDisplayTestStatus(__func__, SKIPPED);
    #endif/*PROC_ESVM_FULL_GENERATION_TESTING*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

/* ===========================
    PERFORMANCE EVALUATION
=========================== */

/*
    Evaluates various performance mesures of classification scores according to ground truths
*/
void eval_PerformanceClassificationScores(std::vector<double> normScores, std::vector<int> probeGroundTruths)
{
    std::vector<double> FPR, TPR, PPV;
    eval_PerformanceClassificationScores(normScores, probeGroundTruths, FPR, TPR, PPV);
}

/*
    Evaluates various performance mesures of classification scores according to ground truths and return (FPR,TPR) results
*/
void eval_PerformanceClassificationScores(std::vector<double> normScores, std::vector<int> probeGroundTruths,
                                          std::vector<double>& FPR, std::vector<double>& TPR, std::vector<double>& PPV)
{
    ASSERT_LOG(normScores.size() == probeGroundTruths.size(), "Number of classification scores and ground truth must match");

    logstream logger(LOGGER_FILE);

    // Evaluate results
    int steps = 100;
    std::vector<double> thresholds(steps + 1);
    FPR = std::vector<double>(steps + 1, 0);
    TPR = std::vector<double>(steps + 1, 0);
    PPV = std::vector<double>(steps + 1, 0);
    for (int i = 0; i <= steps; ++i)
    {
        int FP, FN, TP, TN;
        thresholds[i] = (double)(steps - i) / (double)steps; // Go in reverse threshold order to respect 'calcAUC' requirement
        countConfusionMatrix(normScores, probeGroundTruths, thresholds[i], &TP, &TN, &FP, &FN);
        TPR[i] = calcTPR(TP, FN);
        FPR[i] = calcFPR(FP, TN);
        PPV[i] = calcPPV(TP, FP);
    }
    double AUC = calcAUC(FPR, TPR);
    double pAUC10 = calcAUC(FPR, TPR, 0.10);
    double pAUC20 = calcAUC(FPR, TPR, 0.20);
    for (size_t i = 0; i < FPR.size(); ++i)
        logger << "(FPR,TPR,PPV)[" << i << "] = " << FPR[i] << "," << TPR[i] << "," << PPV[i] << " | T = " << thresholds[i] << std::endl;
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
    size_t steps = 100;
    size_t dimsSummary[2]{ nTargets, 4 };
    size_t dimsThresholds[2]{ nTargets, steps + 1 };
    xstd::mvector<2, double> summaryResults(dimsSummary);   // [target][0: AUC | 1: pAUC(10%) | 2: pAUC(20%) | 3: AUPR](double)
    xstd::mvector<2, ConfusionMatrix> CM(dimsThresholds);   // [target][threshold](ConfusionMatrix)
    for (size_t pos = 0; pos < nTargets; ++pos)
    {
        for (size_t i = 0; i <= steps; ++i)
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
    for (size_t pos = 0; pos < nTargets; ++pos)
        targetLen = std::max(positivesID[pos].size(), targetLen);
    targetLen++;
    header += std::string(targetLen - header.size(), ' ');
    for (size_t c = 0; c < cols.size(); ++c)
        header += "|" + cols[c];
    logger << header << std::endl << std::string(header.size(), '-') << std::endl;
    for (size_t pos = 0; pos < nTargets; ++pos)
    {
        logger << positivesID[pos] << std::string(targetLen - positivesID[pos].size() - 1, ' ');
        for (size_t c = 0; c < cols.size(); ++c)
            logger << " |" << std::setw(cols[c].size() - 1) << summaryResults[pos][c];
        logger << std::endl;
    }
}

} // namespace test
} // namespace esvm

#endif/*ESVM_HAS_TESTS*/
