/* ====================
Normalization oprations
==================== */
#ifndef NORM_H
#define NORM_H

#include "esvmTypes.h"
#include <vector>

#if 0

// Min-Max normalization formula, clip value to [0,1] if specified
double normalizeMinMax(double value, double min, double max, bool clipValue = false);
// Z-Score normalization formula centered around 0.5 with ±3σ, clip value to [0,1] if specified
double normalizeZScore(double value, double mean, double stddev, bool clipValue = false);
// Find the Min-Max values along a vector (not per feature)
void findMinMax(FeatureVector vector, double* min, double* max, int* posMin = nullptr, int* posMax = nullptr);
// Find the Min-Max values acros features and across a whole list of feature vectors
void findMinMaxOverall(std::vector<FeatureVector> featureVectors, double* min, double* max);
// Find the min/max per feature across a whole list of feature vectors
void findMinMaxFeatures(std::vector<FeatureVector> featureVectors, FeatureVector* minFeatures, FeatureVector* maxFeatures);
// Normalization along a feature vector using the specified min/max features, min/max of vector are used if not specified
FeatureVector normalizeMinMaxAllFeatures(FeatureVector featureVector, double min, double max);
FeatureVector normalizeMinMaxAllFeatures(FeatureVector featureVector);
// Normalization [0, 1] across a feature vector using the corresponding min/max features
FeatureVector normalizeMinMaxPerFeatures(FeatureVector featureVector, FeatureVector minFeatures, FeatureVector maxFeatures);

#endif

/* --------------------
Calculation functions
-------------------- */

// Template for function calculation 
/*
template <typename Ret, typename... Args>
using NormFunction = Ret(*)(Args...);
*/
template<NormSignature>
using NormFunction = double(*NormSignature)(double, double, double);

// Min-Max normalization formula
double normMinMax(double value, double min, double max);
// Standard score (z-score) normalization formula
double normZScore(double value, double min, double max);

template<NormFunction>
using MinMax = normMinMax;

template<NormFunction>
using ZScore = normZScore;

/* ---------------------------------------------
Operation function using Calculation functions
--------------------------------------------- */

// Find the norm parameters along a vector (not per feature)
template<NormFunction>
void findNormParams<NormFunction>(FeatureVector featureVector, double* param1, double* param2, int* posParam1 = nullptr, int* posParam2 = nullptr);
// Find the norm parameters acros features and across a whole list of feature vectors
template<typename NormFunction>
void findNormParamsOverall<NormFunction>(std::vector<FeatureVector> featureVectors, double* param1, double* param2);
// Find the norm parameters per feature across a whole list of feature vectors
template<typename NormFunction>
void findNormParamsFeatures<NormFunction>(std::vector<FeatureVector> featureVectors, FeatureVector* featuresParam1, FeatureVector* featuresParam2);
// Normalization along a feature vector using the specified norm features, norm values of vector are used if not specified
template<typename NormFunction>
FeatureVector normalizeAllFeatures<NormFunction>(FeatureVector featureVector, double param1, double param2);
template<typename NormFunction>
FeatureVector normalizeAllFeatures<NormFunction>(FeatureVector featureVector);
// Normalization [0, 1] across a feature vector using the corresponding norm features
template<typename NormFunction>
FeatureVector normalizePerFeatures<NormFunction>(FeatureVector featureVector, FeatureVector featuresParam1, FeatureVector featuresParam2);
// Normalization [0, 1] over all the scores specified in the vector using the found norm values
template<typename NormFunction>
std::vector<double> normalizeClassScores<NormFunction>(std::vector<double> scores);

/* --------------
Extra functions
-------------- */

// Similarity [0, 1] equivalent of class prediction score [-1, 1] by Min-Max rule
inline double normalizeClassScoreToSimilarity(double score) { return (score + 1) / 2; }

#endif/*NORM_H*/
