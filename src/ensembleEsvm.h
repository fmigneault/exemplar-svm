#ifndef ENSEMBLE_ESVM_LIBSVM_H
#define ENSEMBLE_ESVM_LIBSVM_H

#include "esvm.h"
#include "esvmTypes.h"
#include "svm.h"
#include "mvector.hpp"      // Multi-Dimension vectors
#include "feHOG.h"
#include "esvmOptions.h"

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

class EnsembleESVM
{
public:
    EnsembleESVM() {};
    EnsembleESVM::EnsembleESVM(std::vector<cv::Mat> positiveRois, std::string negativesDir);
    std::vector<double> predict(const cv::Mat roi);
    inline std::string getTargetID() { return targetID; }

private:
    void setContants();
    // Constants
    std::string targetID;
    xstd::mvector<1, FeatureVector> probeSampleFeats;
    size_t nPatches;
    size_t nPositives;
    cv::Size imageSize;
    cv::Size patchCounts;
    cv::Size blockSize;
    cv::Size blockStride;
    cv::Size cellSize;
    int nBins;
    std::string sampleFileExt;
    FileFormat sampleFileFormat;

    xstd::mvector<2, double> scores;
    xstd::mvector<1, double> classificationScores;

    FeatureExtractorHOG hog;
    xstd::mvector<2, ESVM> ensembleEsvm; 

    double hogHardcodedFoundMin;            // Min found using 'FullChokePoint' test with SAMAN pre-generated files
    double hogHardcodedFoundMax;     // Max found using 'FullChokePoint' test with SAMAN pre-generated files

};

#endif/*ENSEMBLE_ESVM_LIBSVM_H*/
