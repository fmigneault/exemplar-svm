/* ====================
Normalization oprations
==================== */
#ifndef NORM_H
#define NORM_H

#include "esvmTypes.h"
#include <vector>

/* --------------------
Calculation functions
-------------------- */

// Structure for templated function calculation using Min-Max
// Min-Max normalization formula, clip value to [0,1] if specified
double MinMax(double value, double min, double max, bool clipValue);
/*
class MinMax {
    double operator() (double value, double min, double max, bool clipValue = false);
};
*/
// Structure for templated function calculation using Standard Score (z-score)
// Z-Score normalization formula centered around 0.5 with ±3σ, clip value to [0,1] if specified
double ZScore(double value, double mean, double stddev, bool clipValue);
/*
class ZScore {
    double operator() (double value, double min, double max, bool clipValue = false);
};
*/
/*
template<double NormFunction(double, double, double, bool)>
inline double normalize(double value, double min, double max, bool clipValue = false)
{
    NormFunction::operator();
}
*/

/* ---------------------------------------------
Operation function using Calculation functions
--------------------------------------------- */

// Normalization formula, clip value to [0,1] if specified
template<double NormFunction(double, double, double, bool)>
double normalize(double value, double min, double max, bool clipValue = false);
// Find the norm parameters along a vector (not per feature)
template<double NormFunction(double, double, double, bool)>
void findNormParams(FeatureVector featureVector, double* param1, double* param2, int* posParam1 = nullptr, int* posParam2 = nullptr);
// Find the norm parameters acros features and across a whole list of feature vectors
template<double NormFunction(double, double, double, bool)>
void findNormParamsOverall(std::vector<FeatureVector> featureVectors, double* param1, double* param2);
// Find the norm parameters per feature across a whole list of feature vectors
template<double NormFunction(double, double, double, bool)>
void findNormParamsFeatures(std::vector<FeatureVector> featureVectors, FeatureVector* param1Features, FeatureVector* param2Features);
template<double NormFunction(double, double, double, bool)>
void findNormParamsFeatures<MinMax>(std::vector<FeatureVector> featureVectors, FeatureVector* minFeatures, FeatureVector* maxFeatures);
// Normalization along a feature vector using the specified norm features, norm values of vector are used if not specified
template<double NormFunction(double, double, double, bool)>
FeatureVector normalizeAllFeatures(FeatureVector featureVector, double param1, double param2);
template<double NormFunction(double, double, double, bool)>
FeatureVector normalizeAllFeatures(FeatureVector featureVector);
// Normalization [0, 1] across a feature vector using the corresponding norm features
template<double NormFunction(double, double, double, bool)>
FeatureVector normalizePerFeatures(FeatureVector featureVector, FeatureVector featuresParam1, FeatureVector featuresParam2);
// Normalization [0, 1] over all the scores specified in the vector using the found norm values
template<double NormFunction(double, double, double, bool)>
std::vector<double> normalizeClassScores(std::vector<double> scores);

/* --------------
Extra functions
-------------- */

// Similarity [0, 1] equivalent of class prediction score [-1, 1] by Min-Max rule
inline double normalizeClassScoreToSimilarity(double score) { return (score + 1) / 2; }

#endif/*NORM_H*/
