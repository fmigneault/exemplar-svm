#ifndef ESVM_TYPES_H
#define ESVM_TYPES_H

//namespace esvm {

// SVM implementation library include, types and functions
#if defined(ESVM_USE_LIBSVM) && !defined(ESVM_USE_LIBLINEAR)
    #include "svm.h"
    #define ESVM_BASE "LIBSVM"
    typedef struct svm_model        svmModel;
    typedef struct svm_parameter    svmParam;
    typedef struct svm_problem      svmProblem;
    typedef struct svm_node         svmFeature;
    #define svmTrain                svm_train
    #define svmPredict              svm_predict
    #define svmPredictProbability   svm_predict_probability
    #define svmPredictValues        svm_predict_values
    #define svmCheckParam           svm_check_parameter
    #define svmLoadModel            svm_load_model
    #define svmSaveModel            svm_save_model
    #define svmFreeModel            svm_free_model_content
    #define svmDestroyModel         svm_free_and_destroy_model
#elif defined(ESVM_USE_LIBLINEAR) && !defined(ESVM_USE_LIBSVM)
    #include "linear.h"
    #define ESVM_BASE "LIBLINEAR"
    typedef struct model            svmModel;
    typedef struct parameter        svmParam;
    typedef struct problem          svmProblem;
    typedef struct feature_node     svmFeature;
    #define svmTrain                train
    #define svmPredict              predict
    #define svmPredictProbability   predict_probability
    #define svmPredictValues        predict_values
    #define svmCheckParam           check_parameter
    #define svmLoadModel            load_model
    #define svmSaveModel            save_model
    #define svmFreeModel            free_model_content
    #define svmDestroyModel         free_and_destroy_model
#else
    #error "Invalid SVM implementation library"
#endif/*Base SVM Library*/

// Status to free model memory, matches libsvm for '0'/'1'
enum FreeModelState {
    PARAM = 0,
    MODEL = 1,
    MULTI = 2   // only for testing purposes, model shouldn't have both in 'live' operation
};

//} // namespace esvm

#endif/*ESVM_TYPES_H*/
