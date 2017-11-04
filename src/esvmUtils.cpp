#include "esvmUtils.h"
#include "imgUtils.h"
#include "generic.h"

#include <assert.h>

namespace esvm {

cv::Mat preprocessFromMode(cv::Mat roi, cv::CascadeClassifier ccLocalSearch)
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

    #else // unknown ROI preprocessing mode
    return cv::Mat();

    #endif/*ESVM_ROI_PREPROCESS_MODE*/
}

std::string svm_type_name(svmModel *model)
{
    if (model == nullptr) return "'null'";
    #if ESVM_USE_LIBSVM
    return svm_type_name(model->param.svm_type);
    #elif ESVM_USE_LIBLINEAR
    return svm_type_name(model->param.solver_type);
    #endif/*ESVM_USE_LIBSVM | ESVM_USE_LIBLINEAR*/
}

std::string svm_type_name(int type)
{
    switch (type)
    {
        #if ESVM_USE_LIBSVM
        case C_SVC:         return "C_SVC";
        case NU_SVC:        return "NU_SVC";
        case ONE_CLASS:     return "ONE_CLASS";
        case EPSILON_SVR:   return "EPSILON_SVR";
        case NU_SVR:        return "NU_SVR";
        #elif ESVM_USE_LIBLINEAR
        case L2R_LR:
        case L1R_LR:
        case L2R_LR_DUAL:
            return "L_LR";
        case L2R_L2LOSS_SVC_DUAL:
        case L2R_L2LOSS_SVC:
        case L2R_L1LOSS_SVC_DUAL:
        case L1R_L2LOSS_SVC:
        case MCSVM_CS:
            return "C_SVC";
        case L2R_L2LOSS_SVR:
        case L2R_L2LOSS_SVR_DUAL:
        case L2R_L1LOSS_SVR_DUAL:
            return "L_SVR";
        #endif/*ESVM_USE_LIBSVM | ESVM_USE_LIBLINEAR*/
        default:            return "UNDEFINED (" + std::to_string(type) + ")";
    }
}

std::string svm_kernel_name(svmModel *model)
{
    #if ESVM_USE_LIBSVM
    if (model == nullptr) return "'null'";
    return svm_kernel_name(model->param.kernel_type);
    #elif ESVM_USE_LIBLINEAR
    return "LINEAR";
    #endif/*ESVM_USE_LIBSVM | ESVM_USE_LIBLINEAR*/
}

std::string svm_kernel_name(int type)
{
    switch (type)
    {
        #if ESVM_USE_LIBSVM
        case LINEAR:        return "LINEAR";
        case POLY:          return "POLY";
        case RBF:           return "RBF";
        case SIGMOID:       return "SIGMOID";
        case PRECOMPUTED:   return "PRECOMPUTED";
        #elif ESVM_USE_LIBLINEAR
        return "LINEAR";
        #endif/*ESVM_USE_LIBSVM | ESVM_USE_LIBLINEAR*/
        default:            return "UNDEFINED (" + std::to_string(type) + ")";
    }
}

} // namespace esvm
