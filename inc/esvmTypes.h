#ifndef ESVM_TYPES_H
#define ESVM_TYPES_H

enum FileFormat { BINARY, LIBSVM };

// Status to free model memory, matches libsvm for '0'/'1'
enum FreeModelState { 
    PARAM = 0,
    MODEL = 1,
    MULTI = 2   // only for testing purposes, model shouldn't have both in 'live' operation
};

#endif/*ESVM_TYPES_H*/
