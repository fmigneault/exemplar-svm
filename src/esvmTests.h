#ifndef ESVM_TESTS_H
#define ESVM_TESTS_H

#include "opencv2/opencv.hpp"

/* ChokePoint Dataset:
      P#T_S#_C#:      2 portals, 2 types (E:enter/L:leave), 4 sessions, , 3 cameras = 48 video dirs
      Crop Face dir:  up to 30 individuals (if present) with N face ROIs
*/
// Possible sequence information
enum PORTAL_TYPE { ENTER, LEAVE };
const int PORTAL_NUMBER = 2;
const int SESSION_NUMBER = 4;
const int CAMERA_NUMBER = 3;
const int INDIVIDUAL_NUMBER = 30;
// Combine sequence information
std::string buildChokePointSequenceString(int portal, PORTAL_TYPE type, int session, int camera, int id = 0);
std::string buildChokePointIndividualID(int id);
bool checkPathEndSlash(std::string path);

/* Tests */
int test_outputOptions();
int test_imagePaths();
int test_imagePatchExtraction();
int test_multiLevelVectors();
int test_normalizationFunctions();
int test_runBasicExemplarSvmFunctionalities();
int test_runBasicExemplarSvmClassification();
int test_runSingleSamplePerPersonStillToVideo(cv::Size patchCounts);
int test_runSingleSamplePerPersonStillToVideo_FullChokePoint(cv::Size imageSize, cv::Size patchCounts, bool useSyntheticPositives);
int test_runSingleSamplePerPersonStillToVideo_DataFiles_WholeImage();
int test_runSingleSamplePerPersonStillToVideo_DataFiles_DescriptorAndPatchBased(int nPatches);
int test_runSingleSamplePerPersonStillToVideo_TITAN(cv::Size imageSize, cv::Size patchCounts, bool useSyntheticPositives);
int test_runSingleSamplePerPersonStillToVideo_DataFiles_SAMAN();

/* Performance Evaluation */
void eval_PerformanceClassificationScores(std::vector<double> normScores, std::vector<int> probeGroundTruths);

#endif/*ESVM_TESTS_H*/