/* ====================
Normalization oprations
==================== */
#ifndef NORM_H
#define NORM_H

#include "esvmTypes.h"
#include <vector>

#ifndef OUT_PARAM
#define OUT_PARAM
#endif

/* Available normalization methods

       min-max:  x = (value - min) / (max - min)                                            param1 : min,   param2 : max
       z-score:  z = (value - mean) / (6 * stddev), then centered around 0.5 for ±3σ        param1 : mean,  param2 : stddev
*/
enum NormType { MIN_MAX, Z_SCORE };

// Normalization formula, clip value to [0,1] if specified
double normalize(NormType norm, double value, double param1, double param2, bool clipValue = false);
double normalizeMinMax(double value, double min, double max, bool clipValue = false);
double normalizeZScore(double value, double mean, double stddev, bool clipValue = false);

// Normalization along a feature vector using the specified norm features, or using found norm values of vector if not specified
FeatureVector normalizeOverAll(NormType norm, const FeatureVector& featureVector, double param1, double param2, bool clipFeatures = false);
FeatureVector normalizeOverAll(NormType norm, const FeatureVector& featureVector, bool clipFeatures = false);

// Normalization [0, 1] for each feature in the vector using the corresponding norm features
FeatureVector normalizePerFeature(NormType norm, const FeatureVector& featureVector,
                                  const FeatureVector& featuresParam1, const FeatureVector& featuresParam2, bool clipFeatures = false);

// Normalization [0, 1] over all the scores with specified norm values, or using the found norm values from the scores otherwise
std::vector<double> normalizeClassScores(NormType norm, const std::vector<double>& scores, double param1, double param2, bool clipScores = false);
std::vector<double> normalizeClassScores(NormType norm, const std::vector<double>& scores, bool clipScores = false);

// Similarity [0, 1] equivalent of class prediction score [-1, 1] by Min-Max rule
inline double normalizeClassScoreToSimilarity(double score) { return (score + 1) / 2; }

// Find the norm parameters across a feature vector (not per feature)
void findNormParamsAcrossFeatures(NormType norm, const FeatureVector& featureVector, OUT_PARAM double& param1, OUT_PARAM double& param2,
                                  int *posParam1 = nullptr, int *posParam2 = nullptr);

// Find the norm parameters across features and across a whole list of feature vectors
void findNormParamsOverAll(NormType norm, const std::vector<FeatureVector>& featureVectors, OUT_PARAM double& param1, OUT_PARAM double& param2);

// Find the norm parameters per feature across a whole list of feature vectors
void findNormParamsPerFeature(NormType norm, const std::vector<FeatureVector>& featureVectors, 
                              OUT_PARAM FeatureVector& featuresParam1, OUT_PARAM FeatureVector& featuresParam2);

// Find the norm parameters from specified scores
void findNormParamsClassScores(NormType norm, const std::vector<double>& scores, OUT_PARAM double& param1, OUT_PARAM double& param2);

#endif/*NORM_H*/
