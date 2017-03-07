/* --------------------
Normalization oprations
-------------------- */
#ifndef NORM_H
#define NORM_H

#include "esvmTypes.h"
#include <vector>

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
// Normalization [0, 1] over all the scores specified in the vector using the found min/max values
std::vector<double> normalizeMinMaxClassScores(std::vector<double> scores);
// Similarity [0, 1] equivalent of class prediction score [-1, 1] by Min-Max rule
inline double normalizeClassScoreToSimilarity(double score) { return (score + 1) / 2; }

#endif/*NORM_H*/
