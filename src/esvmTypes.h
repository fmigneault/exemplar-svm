#ifndef ESVM_TYPES_H
#define ESVM_TYPES_H

#include <vector>

typedef std::vector<double> FeatureVector;

/*
    indicates how the model memory has to be released according to the way it was allocated
    to ensure properly handling cases according to 'malloc'/'free' or 'new[]'/'delete[]'
*/
enum ModelFreeType
{
    TRAIN_MODEL_LIBSVM = 0,     // model obtained from 'trainModel'/'readSampleDataFile_libsvm' -> 'svm_train'  (matches libsvm's value)
    LOAD_MODEL_LIBSVM = 1,      // model obtained from 'loadModelFile_libsvm' -> 'svm_load_model'               (matches libsvm's value)
    TRAIN_MODEL_BINARY = 2,     // model obtained from 'trainModel'/'readSampleDataFile_binary' -> 'svm_train'
    LOAD_MODEL_BINARY = 3       // model obtained from 'loadModelFile_binary' -> directly set parameters
};

enum FileFormat { BINARY, LIBSVM };

#endif/*ESVM_TYPES_H*/
