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
    esvmEnsemble(std::vector<cv::Mat> positiveROIs, std::string negativesDir, std::vector<std::string> positiveIDs = {});
    std::vector<double> predict(const cv::Mat roi);
    inline size_t getPositiveCount() { return enrolledPositiveIDs.size(); }
    inline size_t getPatchCount() { return patchCounts.area(); }
    inline std::string getPositiveID(int positiveIndex);    

private:
    void setConstants();    
    std::vector<std::string> enrolledPositiveIDs;

    // Constants
    cv::Size imageSize;
    cv::Size patchCounts;
    cv::Size blockSize;
    cv::Size blockStride;
    cv::Size cellSize;
    int nBins;
    FeatureExtractorHOG hog;

    bool useHistEqual;
    
    xstd::mvector<2, ESVM> EoESVM; 

    std::string sampleFileExt;
    FileFormat sampleFileFormat;

    double hogHardcodedFoundMin;
    double hogHardcodedFoundMax;

    double scoresHardcodedFoundMin;
    double scoresHardcodedFoundMax;

    double scoresHardcodedFoundMean;
    double scoresHardcodedFoundStdDev;
};

#endif/*ESVM_ENSEMBLE_H*/
