#include "esvmUtils.h"
#include "imgUtils.h"
#include "generic.h"

#include <assert.h>


cv::Mat esvmPreprocessFromMode(cv::Mat roi, cv::CascadeClassifier ccLocalSearch)
{
    #if ESVM_ROI_PREPROCESS_MODE == 0
    return roi;
    
    #elif ESVM_ROI_PREPROCESS_MODE == 1
    ASSERT_LOG(ccLocalSearch.empty(), "CascadeClassifier must be loaded for preprocessing with 'ESVM_ROI_PREPROCESS_MODE == 1'");
    double scaleFactor = 1.1;
    int nmsThreshold = 1;                           // 0 generates multiple detections, >0 usually returns only 1 detection
    cv::Size minSize(20, 20), maxSize = roi.size();
    std::vector<cv::Rect> detections;
    ccLocalSearch.detectMultiScale(roi, detections, scaleFactor, nmsThreshold, cv::CASCADE_SCALE_IMAGE, minSize, maxSize);
    return detections.size() > 0 ? roi(detections[0]) : roi;

    #elif ESVM_ROI_PREPROCESS_MODE == 2
    return imCropByRatio(roi, ESVM_ROI_CROP_RATIO, CENTER_MIDDLE);

    #else
    return cv::Mat();

    #endif/*ESVM_ROI_PREPROCESS_MODE*/
}

std::string svm_type_name(svm_model *model)
{
    if (model == nullptr) return "'null'";
    return svm_type_name(model->param.svm_type);
}

std::string svm_type_name(int type)
{
    switch (type)
    {
        case C_SVC:         return "C_SVC";
        case NU_SVC:        return "NU_SVC";
        case ONE_CLASS:     return "ONE_CLASS";
        case EPSILON_SVR:   return "EPSILON_SVR";
        case NU_SVR:        return "NU_SVR";
        default:            return "UNDEFINED (" + std::to_string(type) + ")";
    }
}

std::string svm_kernel_name(svm_model *model)
{
    if (model == nullptr) return "'null'";
    return svm_kernel_name(model->param.kernel_type);
}

std::string svm_kernel_name(int type)
{
    switch (type)
    {
        case LINEAR:        return "LINEAR";
        case POLY:          return "POLY";
        case RBF:           return "RBF";
        case SIGMOID:       return "SIGMOID";
        case PRECOMPUTED:   return "PRECOMPUTED";
        default:            return "UNDEFINED (" + std::to_string(type) + ")";
    }
}
