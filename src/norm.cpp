#include "norm.h"
#include "generic.h"
#include <algorithm>
#include <float.h>

double MinMax::normalize(double value, double min, double max, bool clipValue)
{
    ASSERT_THROW(max > min, "max must be greater than min (max: " + std::to_string(max) + ", min: " + std::to_string(min) + ")");
    double normValue = (value - min) / (max - min);
    return clipValue ? std::min(std::max(normValue, 0.0), 1.0) : normValue;
}

double ZScore::normalize(double value, double mean, double stddev, bool clipValue)
{
    ASSERT_THROW(stddev != 0, "stddev must be different than zero");
    double sigmaFactor = 3.0;
    double normValue = ((value - mean) / stddev) / (2.0 * sigmaFactor * stddev) + 0.5;
    return clipValue ? std::min(std::max(normValue, 0.0), 1.0) : normValue;
}

void MinMax::findNormParams(FeatureVector featureVector, double* min, double* max, int* posMin, int* posMax)
{
    // check values/references
    ASSERT_THROW(min != nullptr, "min reference not specified");
    ASSERT_THROW(max != nullptr, "max reference not specified");

    int nFeatures = featureVector.size();
    ASSERT_THROW(nFeatures > 0, "vector cannot be empty");

    // initialization
    *min = featureVector[0], *max = featureVector[0];
    if (posMin != nullptr)
        *posMin = 0;
    if (posMax != nullptr)
        *posMax = 0;

    // update min/max
    for (int f = 1; f < featureVector.size(); f++)
    {
        if (featureVector[f] < *min)
        {
            *min = featureVector[f];
            if (posMax != nullptr)
                *posMin = f;
        }
        else if (featureVector[f] > *max)
        {
            *max = featureVector[f];
            if (posMax != nullptr)
                *posMax = f;
        }
    }
}

void ZScore::findNormParams(FeatureVector featureVector, double *mean, double *stddev, int *posMean, int *posStdDev)
{
    throw std::runtime_error("NOT IMPLEMENTED");
}

void MinMax::findNormParamsOverall(std::vector<FeatureVector> featureVectors, double *min, double *max)
{
    double minFound, maxFound;
    size_t nSamples = featureVectors.size();
    for (size_t s = 0; s < nSamples; s++)
    {
        findNormParams(featureVectors[s], &minFound, &maxFound);
        if (s == 0 || minFound < *min) *min = minFound;
        if (s == 0 || maxFound > *max) *max = maxFound;
    }
}

void ZScore::findNormParamsOverall(std::vector<FeatureVector> featureVectors, double *mean, double *stddev)
{    
    size_t nSamples = featureVectors.size();
    ASSERT_LOG(nSamples > 0, "vector must contain at least one feature vector");
    size_t nFeatures = featureVectors[0].size();
    size_t total = nFeatures * nSamples;

    double meanFound = 0, stdDevFound = 0;
    for (size_t s = 0; s < nSamples; s++)
        for (size_t f = 0; f < nFeatures; f++)
            meanFound += featureVectors[s][f];
    meanFound /= (double)total;

    for (size_t s = 0; s < nSamples; s++)
        for (size_t f = 0; f < nFeatures; f++)
            stdDevFound += (featureVectors[s][f] - meanFound) * (featureVectors[s][f] - meanFound);
    stdDevFound /= (double)total;

    *mean = meanFound;
    *stddev = stdDevFound;
}

void MinMax::findNormParamsFeatures(std::vector<FeatureVector> featureVectors, FeatureVector *featuresMin, FeatureVector *featuresMax)
{
    ASSERT_THROW(featuresMin != nullptr, "feature vector for min features not specified");
    ASSERT_THROW(featuresMax != nullptr, "feature vector for max features not specified");

    // initialize with first vector
    int nFeatures = featureVectors[0].size();
    FeatureVector min = featureVectors[0];
    FeatureVector max = featureVectors[0];

    // find min/max values
    /// ############################################# #pragma omp parallel for
    for (int v = 1; v < featureVectors.size(); v++)
    {
        for (int f = 0; f < nFeatures; f++)
        {
            if (featureVectors[v][f] < min[f])
                min[f] = featureVectors[v][f];
            if (featureVectors[v][f] > max[f])
                max[f] = featureVectors[v][f];
        }
    }

    // update values
    *featuresMin = min;
    *featuresMax = max;
}

FeatureVector NormFunction::normalizeAllFeatures(FeatureVector featureVector, double param1, double param2)
{
    int nFeatures = featureVector.size();
    for (int f = 0; f < nFeatures; f++)
        featureVector[f] = normalize(featureVector[f], param1, param2);
    return featureVector;
}

FeatureVector NormFunction::normalizeAllFeatures(FeatureVector featureVector)
{
    double param1, param2;
    findNormParams(featureVector, &param1, &param2);
    return normalizeAllFeatures(featureVector, param1, param2);
}

FeatureVector NormFunction::normalizePerFeatures(FeatureVector featureVector, FeatureVector featuresParam1, FeatureVector featuresParam2)
{
    // check number of features
    size_t nFeatures = featureVector.size();
    ASSERT_LOG(featuresParam1.size() == nFeatures, "param1 features dimension doesn't match feature vector to normalize");
    ASSERT_LOG(featuresParam2.size() == nFeatures, "param2 features dimension doesn't match feature vector to normalize");

    // normalize values    
    for (size_t f = 0; f < nFeatures; f++)
        featureVector[f] = normalize(featureVector[f], featuresParam1[f], featuresParam2[f]);

    return featureVector;
}

std::vector<double> NormFunction::normalizeClassScores(std::vector<double> scores)
{
    double param1, param2;
    findNormParams(scores, &param1, &param2);
    return normalizeAllFeatures(scores, param1, param2);
}

#if NORM_TEMPLATES == 1

template<class NormFunction>
double normalize(double value, double param1, double param2, bool clipValue)
{
    return NormFunction::normalize(value, param1, param2, clipValue);
}

template<>
double normalize<MinMax>(double value, double param1, double param2, bool clipValue)
{
    return MinMax::normalize(value, param1, param2, clipValue);
}

template<>
double normalize<ZScore>(double value, double param1, double param2, bool clipValue)
{
    return ZScore::normalize(value, param1, param2, clipValue);
}

template<class NormFunction>
void findNormParams<NormFunction>(FeatureVector featureVector, double* min, double* max, int* posMin, int* posMax)
{
    findNormParams(featureVector, min, max, posMin, posMax);
}

template<class NormFunction>
void findNormParamsOverall<NormFunction>(std::vector<FeatureVector> featureVectors, double *min, double *max)
{
    findNormParamsOverall(featureVectors, min, max);
}

template<class NormFunction>
void findNormParamsFeatures<NormFunction>(std::vector<FeatureVector> featureVectors, FeatureVector *featuresMin, FeatureVector *featuresMax)
{
    findNormParamsFeatures(featureVector, featuresMin, featuresMax);
}

template<class NormFunction>
FeatureVector normalizeAllFeatures<NormFunction>(FeatureVector featureVector, double param1, double param2)
{
    normalizeAllFeatures(featureVector, param1, param2);
}

template<class NormFunction>
FeatureVector normalizeAllFeatures<NormFunction>(FeatureVector featureVector)
{
    normalizeAllFeatures(featureVector)
}

template<class NormFunction>
FeatureVector normalizePerFeatures<NormFunction>(FeatureVector featureVector, FeatureVector featuresParam1, FeatureVector featuresParam2)
{
    normalizePerFeatures(featureVector, featuresParam1, featuresParam2);
}

template<class NormFunction>
std::vector<double> normalizeClassScores<NormFunction>(std::vector<double> scores)
{
    double param1, param2;
    findNormParams<NormFunction>(scores, &param1, &param2);
    return normalizeAllFeatures<NormFunction>(scores, param1, param2);
}

#endif/*NORM_TEMPLATES*/
