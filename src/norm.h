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

static class NormFunction
{
public:
    virtual double operator() (double value, double param1, double param2, bool clipValue = false) { return normalize(value, param1, param2, clipValue); }
    virtual double normalize(double value, double param1, double param2, bool clipValue = false) { return -DBL_MAX; }
    virtual void findNormParams(FeatureVector featureVector, double *param1, double *param2, int *posParam1 = nullptr, int *posParam2 = nullptr) {}
    virtual void findNormParamsOverall(std::vector<FeatureVector> featureVectors, double* param1, double* param2) {}
    virtual void findNormParamsFeatures(std::vector<FeatureVector> featureVectors, FeatureVector *featuresParam1 = nullptr, FeatureVector *featuresParam2 = nullptr) {}
    virtual FeatureVector normalizeAllFeatures(FeatureVector featureVector);
    virtual FeatureVector normalizeAllFeatures(FeatureVector featureVector, double param1, double param2);
    virtual FeatureVector normalizePerFeatures(FeatureVector featureVector, FeatureVector featuresParam1, FeatureVector featuresParam2);
    std::vector<double> normalizeClassScores(std::vector<double> scores);
};


static class MinMax : public NormFunction
{
public:    
    static double normalize(double value, double min, double max, bool clipValue = false);
    static void findNormParams(FeatureVector featureVector, double *min, double *max, int *posMin = nullptr, int *posMax = nullptr);
    static void findNormParamsOverall(std::vector<FeatureVector> featureVectors, double *min, double *max);
    static void findNormParamsFeatures(std::vector<FeatureVector> featureVectors, FeatureVector *featuresMin, FeatureVector *featuresMax);    
};

static class ZScore : public NormFunction
{
public:    
    static double normalize(double value, double mean, double stddev, bool clipValue = false);
    static void findNormParams(FeatureVector featureVector, double *mean, double *stddev, int *posMean = nullptr, int *posStdDev = nullptr);
    static void findNormParamsOverall(std::vector<FeatureVector> featureVectors, double *mean, double *stddev);
    static void findNormParamsFeatures(std::vector<FeatureVector> featureVectors, FeatureVector *featuresMean = nullptr, FeatureVector *featuresStdDev = nullptr) {}
};

/*
class NormFunction
{
public:
    double operator() (double value, double min, double max, bool clipValue);
};
*/
// Structure for templated function calculation using Min-Max
// Min-Max normalization formula, clip value to [0,1] if specified
///NormFunction MinMax;
/*
template<class NormFunction>
struct MinMax {
    double operator() (double value, double min, double max, bool clipValue);
};
*/
/*
class MinMax {
    double operator() (double value, double min, double max, bool clipValue = false);
};
*/
// Structure for templated function calculation using Standard Score (z-score)
// Z-Score normalization formula centered around 0.5 with ±3σ, clip value to [0,1] if specified
///NormFunction ZScore;
/*
template<class NormFunction>
struct ZScore {
    double operator() (double value, double mean, double stddev, bool clipValue);
};
*/
/*
class ZScore {
    double operator() (double value, double min, double max, bool clipValue = false);
};
*/
/*
template<class NormFunction>
inline double normalize(double value, double min, double max, bool clipValue = false)
{
    NormFunction::operator();
}
*/

/* ---------------------------------------------
Operation function using Calculation functions
--------------------------------------------- */

#define NORM_TEMPLATES 0
#if NORM_TEMPLATES == 1

// Normalization formula, clip value to [0,1] if specified
template<class NormFunction>
double normalize(double value, double param1, double param2, bool clipValue = false);

// Find the norm parameters along a vector (not per feature)
template<class NormFunction>
void findNormParams(FeatureVector featureVector, double *param1, double *param2, int *posParam1 = nullptr, int *posParam2 = nullptr);
template<>
void findNormParams<MinMax>(FeatureVector featureVector, double *min, double *max, int *posMin, int *posMax);
template<>
void findNormParams<ZScore>(FeatureVector featureVector, double *mean, double *stddev, int *posMean, int *posStdDev);

// Find the norm parameters acros features and across a whole list of feature vectors
template<class NormFunction>
void findNormParamsOverall(std::vector<FeatureVector> featureVectors, double *param1, double *param2);

// Find the norm parameters per feature across a whole list of feature vectors
template<class NormFunction>
void findNormParamsFeatures(std::vector<FeatureVector> featureVectors, FeatureVector *param1Features, FeatureVector *param2Features);

// Normalization along a feature vector using the specified norm features, norm values of vector are used if not specified
template<class NormFunction>
FeatureVector normalizeAllFeatures(FeatureVector featureVector, double param1, double param2);
template<class NormFunction>
FeatureVector normalizeAllFeatures(FeatureVector featureVector);

// Normalization [0, 1] across a feature vector using the corresponding norm features
template<class NormFunction>
FeatureVector normalizePerFeatures(FeatureVector featureVector, FeatureVector featuresParam1, FeatureVector featuresParam2);

// Normalization [0, 1] over all the scores specified in the vector using the found norm values
template<class NormFunction>
std::vector<double> normalizeClassScores(std::vector<double> scores);

#endif/*NORM_TEMPLATES*/

#define NORM_TEMPLATES_V2 1
#if NORM_TEMPLATES_V2

// Normalization formula, clip value to [0,1] if specified
template<class NormFunction>
double normalize(double value, double param1, double param2, bool clipValue = false)
{
    NormFunction nf;
    return nf.normalize(value, param1, param2, clipValue);
}

// Find the norm parameters along a vector (not per feature)
template<class NormFunction>
void findNormParams(FeatureVector featureVector, double *param1, double *param2, int *posParam1 = nullptr, int *posParam2 = nullptr)
{
    NormFunction nf;
    nf.findNormParams(featureVector, param1, param2, posParam1, posParam2);
}
/// template<>
/// void findNormParams<MinMax>(FeatureVector featureVector, double *min, double *max, int *posMin, int *posMax);
/// template<>
/// void findNormParams<ZScore>(FeatureVector featureVector, double *mean, double *stddev, int *posMean, int *posStdDev);

// Find the norm parameters acros features and across a whole list of feature vectors
template<class NormFunction>
void findNormParamsOverall(std::vector<FeatureVector> featureVectors, double *param1, double *param2)
{
    NormFunction nf;
    nf.findNormParamOverall(featureVectors, param1, param2);
}

// Find the norm parameters per feature across a whole list of feature vectors
template<class NormFunction>
void findNormParamsFeatures(std::vector<FeatureVector> featureVectors, FeatureVector *param1Features, FeatureVector *param2Features)
{
    NormFunction nf;
    nf.findNormParamsFeatures(featureVectors, param1Features, param2Features);
}

// Normalization along a feature vector using the specified norm features, norm values of vector are used if not specified
template<class NormFunction>
FeatureVector normalizeAllFeatures(FeatureVector featureVector, double param1, double param2)
{
    NormFunction nf; 
    return nf.normalizeAllFeatures(featureVector, param1, param2);
}
template<class NormFunction>
FeatureVector normalizeAllFeatures(FeatureVector featureVector)
{
    NormFunction nf;
    return nf.normalizeAllFeatures(featureVector);
}

// Normalization [0, 1] across a feature vector using the corresponding norm features
template<class NormFunction>
FeatureVector normalizePerFeatures(FeatureVector featureVector, FeatureVector featuresParam1, FeatureVector featuresParam2)
{
    NormFunction nf;
    return nf.normalizePerFeatures(featureVector, featuresParam1, featuresParam2);
}

// Normalization [0, 1] over all the scores specified in the vector using the found norm values
template<class NormFunction>
std::vector<double> normalizeClassScores(std::vector<double> scores)
{
    NormFunction nf;
    return nf.normalizeClassScores(scores);
}

#endif/*NORM_TEMPLATES_V2*/

/* --------------
Extra functions
-------------- */

// Similarity [0, 1] equivalent of class prediction score [-1, 1] by Min-Max rule
inline double normalizeClassScoreToSimilarity(double score) { return (score + 1) / 2; }

#endif/*NORM_H*/
