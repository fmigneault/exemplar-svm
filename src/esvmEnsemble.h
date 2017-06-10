#ifndef ESVM_ENSEMBLE_H
#define ESVM_ENSEMBLE_H

#include "esvm.h"
#include "esvmTypes.h"
#include "svm.h"
#include "mvector.hpp"
#include "feHOG.h"
#include "esvmOptions.h"

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

class esvmEnsemble
{
public:
    esvmEnsemble() {};
    esvmEnsemble(const std::vector<std::vector<cv::Mat> >& positiveROIs, const std::string negativesDir,
                 const std::vector<std::string>& positiveIDs = {}, const std::vector<std::vector<cv::Mat> >& additionalNegativeROIs = {});
    std::vector<double> predict(const cv::Mat& roi);
    inline size_t getPositiveCount() { return enrolledPositiveIDs.size(); }
    inline size_t getPatchCount() { return patchCounts.area(); }
    inline std::string getPositiveID(int positiveIndex);

private:
    void setConstants(std::string negativesDir);
    std::vector<std::string> enrolledPositiveIDs;

    // Constants
    cv::Size imageSize;
    cv::Size patchCounts;
    cv::Size windowSize;
    cv::Size blockSize;
    cv::Size blockStride;
    cv::Size cellSize;
    int nBins;
    FeatureExtractorHOG hog;

    xstd::mvector<2, ESVM> EoESVM;

    std::string sampleFileExt;
    FileFormat sampleFileFormat;

    /* --- Reference 'hardcoded' normalization values --- */

    #if ESVM_FEATURE_NORMALIZATION_MODE == 1    // Min-Max features - overall normalization - across patches
    double hogRefMin;
    double hogRefMax;
    #elif ESVM_FEATURE_NORMALIZATION_MODE == 2  // Z-Score features - overall normalization - across patches
    double hogRefMean;
    double hogRefStdDev;
    #elif ESVM_FEATURE_NORMALIZATION_MODE == 3  // Min-Max features - per feature normalization - across patches
    FeatureVector hogRefMin;
    FeatureVector hogRefMax;
    #elif ESVM_FEATURE_NORMALIZATION_MODE == 4  // Z-Score features - per feature normalization - across patches
    FeatureVector hogRefMean;
    FeatureVector hogRefStdDev;
    #elif ESVM_FEATURE_NORMALIZATION_MODE == 5  // Min-Max features - overall normalization - for each patch
    std::vector<double> hogRefMin;
    std::vector<double> hogRefMax;
    #elif ESVM_FEATURE_NORMALIZATION_MODE == 6  // Z-Score features - overall normalization - for each patch
    std::vector<double> hogRefMean;
    std::vector<double> hogRefStdDev;
    #elif ESVM_FEATURE_NORMALIZATION_MODE == 7  // Min-Max features - per feature normalization - for each patch
    std::vector<FeatureVector> hogRefMin;
    std::vector<FeatureVector> hogRefMax;
    #elif ESVM_FEATURE_NORMALIZATION_MODE == 8  // Z-Score features - per feature normalization - for each patch
    std::vector<FeatureVector> hogRefMean;
    std::vector<FeatureVector> hogRefStdDev;
    #endif/*ESVM_FEATURE_NORMALIZATION_MODE*/

    #if ESVM_SCORE_NORMALIZATION_MODE == 1      // Min-Max scores normalization
    double scoreRefMin;
    double scoreRefMax;
    #elif ESVM_SCORE_NORMALIZATION_MODE == 2    // Z-Score scores normalization
    double scoreRefMean;
    double scoreRefStdDev;
    #endif/*ESVM_SCORE_NORMALIZATION_MODE*/
};

#endif/*ESVM_ENSEMBLE_H*/
