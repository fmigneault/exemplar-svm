#ifndef EXAMPLAR_SVM_TESTS_H
#define EXAMPLAR_SVM_TESTS_H

#include "feHOG.h"
#include "feLBP.h"
#include "esvm.h"
#include "esvmTypesDef.h"
#include "helperFunctions.h"

/* ChokePoint Dataset:
      P#T_S#_C#:      2 portals, 2 types (E:enter/L:leave), 4 sessions, , 3 cameras = 48 video dirs
      Crop Face dir:  up to 30 individuals (if present) with N face ROIs
*/
// Specify how the training samples are regrouped into training sequences
//    0: use all cameras in a corresponding session as a common list of training samples (ie: 4 session = 4 sequences)
//    1: use each scene as an independant list of training samples (ie: 2 portals x 2 types x 4 sessions x 3 cameras = 48 sequences) 
#define CHOKEPOINT_FULL_TEST_SEQUENCES_MODE 0
// Possible sequence information
enum PORTAL_TYPE { ENTER, LEAVE };
const int PORTAL_NUMBER = 2;
const int SESSION_NUMBER = 4;
const int CAMERA_NUMBER = 3;
const int INDIVIDUAL_NUMBER = 30;
// Combine sequence information
std::string buildChokePointSequenceString(int portal, PORTAL_TYPE type, int session, int camera, int id = 0);
std::string buildChokePointIndividualID(int id);

/* Tests */
int test_imagePatchExtraction(void);
int test_runBasicExemplarSvmFunctionalities(void);
int test_runBasicExemplarSvmClassification(void);
int test_runSingleSamplePerPersonStillToVideo(cv::Size patchCounts);
int test_runSingleSamplePerPersonStillToVideo_FullChokePoint(cv::Size imageSize, cv::Size patchCounts, bool useSyntheticPositives);

#endif/*EXAMPLAR_SVM_TESTS_H*/