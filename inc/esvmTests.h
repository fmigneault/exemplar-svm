#ifndef ESVM_TESTS_H
#define ESVM_TESTS_H

#ifdef ESVM_HAS_TESTS

#include "esvm.h"
#include "esvmTypes.h"

#include "opencv2/opencv.hpp"
#include "types.h"
#include "mvector.hpp"

namespace esvm {
namespace test {

/* Test utilities */
svm_model* buildDummyExemplarSvmModel(FreeModelState free_sv = MODEL);
void destroyDummyExemplarSvmModelContent(svm_model *model, FreeModelState free_sv);
void generateDummySamples(std::vector<FeatureVector>& samples, std::vector<int>& targetOutputs, size_t nSamples, size_t nFeatures);
bool generateDummySampleFile_libsvm(std::string filePath, size_t nSamples, size_t nFeatures);
bool generateDummySampleFile_binary(std::string filePath, size_t nSamples, size_t nFeatures);
void displayHeader();
void displayOptions();

/* Tests */
int test_paths();
int test_imagePatchExtraction();
int test_imagePreprocessing();
int test_multiLevelVectors();
int test_normalizationFunctions();
int test_performanceEvaluationFunctions();
int test_ESVM_BasicFunctionalities();
int test_ESVM_BasicClassification();
int test_ESVM_ReadSampleFile_libsvm();
int test_ESVM_ReadSampleFile_binary();
int test_ESVM_ReadSampleFile_compare();
int test_ESVM_ReadSampleFile_timing(size_t nSamples, size_t nFeatures);
int test_ESVM_WriteSampleFile_timing(size_t nSamples, size_t nFeatures);
int test_ESVM_SaveLoadModelFile_libsvm();
int test_ESVM_SaveLoadModelFile_binary();
int test_ESVM_SaveLoadModelFile_compare();
int test_ESVM_ModelFromStructSVM();
int test_ESVM_ModelMemoryOperations();
int test_ESVM_ModelMemoryParamCheck();

/* Procedures */
int proc_readDataFiles();
int proc_runSingleSamplePerPersonStillToVideo(cv::Size patchCounts);
int proc_runSingleSamplePerPersonStillToVideo_FullChokePoint(cv::Size imageSize, cv::Size patchCounts);
int proc_runSingleSamplePerPersonStillToVideo_DataFiles_WholeImage();
int proc_runSingleSamplePerPersonStillToVideo_DataFiles_DescriptorAndPatchBased(size_t nPatches);
int proc_runSingleSamplePerPersonStillToVideo_NegativesDataFiles_PositivesExtraction_PatchBased();
int proc_runSingleSamplePerPersonStillToVideo_TITAN(cv::Size imageSize, cv::Size patchCounts, bool useSyntheticPositives);
int proc_runSingleSamplePerPersonStillToVideo_DataFiles_SAMAN();
int proc_runSingleSamplePerPersonStillToVideo_DataFiles_SimplifiedWorking();
int proc_runSingleSamplePerPersonStillToVideo_FullGenerationAndTestProcess();

/* Performance Evaluation */
void eval_PerformanceClassificationScores(std::vector<double> normScores, std::vector<int> probeGroundTruths);
void eval_PerformanceClassificationScores(std::vector<double> normScores, std::vector<int> probeGroundTruths,
                                          std::vector<double>& FPR, std::vector<double>& TPR, std::vector<double>& PPV);
void eval_PerformanceClassificationSummary(std::vector<std::string> positivesID,
                                           xstd::mvector<2, double> normScores, xstd::mvector<2, int> probeGroundTruths);

} // namespace test
} // namespace esvm

#endif/*ESVM_HAS_TESTS*/
#endif/*ESVM_TESTS_H*/
