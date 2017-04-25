#include "norm.h"
#include "generic.h"
#include <algorithm>
#include <float.h>
#include <map>

double normalize(NormType norm, double value, double param1, double param2, bool clipValue)
{
    if (norm == MIN_MAX)
        return normalizeMinMax(value, param1, param2, clipValue);
    else if (norm == Z_SCORE)
        return normalizeZScore(value, param1, param2, clipValue);
    throw std::runtime_error("Undefined normalization method");
}

double normalizeMinMax(double value, double min, double max, bool clipValue)
{
    ASSERT_THROW(max > min, "max must be greater than min (max: " + std::to_string(max) + ", min: " + std::to_string(min) + ")");
    double normValue = (value - min) / (max - min);
    return clipValue ? std::min(std::max(normValue, 0.0), 1.0) : normValue;
}

double normalizeZScore(double value, double mean, double stddev, bool clipValue)
{
    ASSERT_THROW(stddev != 0, "stddev must be different than zero");
    double sigmaFactor = 3.0;
    double normValue = ((value - mean) / stddev) / (2.0 * sigmaFactor * stddev) + 0.5;
    return clipValue ? std::min(std::max(normValue, 0.0), 1.0) : normValue;
}

FeatureVector normalizeOverAll(NormType norm, const FeatureVector& featureVector, double param1, double param2, bool clipFeatures)
{
    size_t nFeatures = featureVector.size();
    FeatureVector normFeatureVector(featureVector.size());
    for (size_t f = 0; f < nFeatures; f++)
        normFeatureVector[f] = normalize(norm, featureVector[f], param1, param2, clipFeatures);
    return normFeatureVector;
}

FeatureVector normalizeOverAll(NormType norm, const FeatureVector& featureVector, bool clipFeatures)
{
    double param1, param2;
    findNormParamsClassScores(norm, featureVector, param1, param2);
    return normalizeOverAll(norm, featureVector, param1, param2, clipFeatures);
}

FeatureVector normalizePerFeature(NormType norm, const FeatureVector& featureVector, 
                                  FeatureVector& featuresParam1, FeatureVector& featuresParam2, bool clipFeatures)
{
    // check number of features
    size_t nFeatures = featureVector.size();
    ASSERT_THROW(featuresParam1.size() == nFeatures, "param1 features dimension doesn't match feature vector to normalize");
    ASSERT_THROW(featuresParam2.size() == nFeatures, "param2 features dimension doesn't match feature vector to normalize");
    FeatureVector normFeatureVector(nFeatures);

    // normalize values    
    for (size_t f = 0; f < nFeatures; f++)
        normFeatureVector[f] = normalize(norm, featureVector[f], featuresParam1[f], featuresParam2[f], clipFeatures);
    return normFeatureVector;
}

std::vector<double> normalizeClassScores(NormType norm, const std::vector<double>& scores, double param1, double param2, bool clipScores)
{
    return normalizeOverAll(norm, scores, param1, param2, clipScores);
}

std::vector<double> normalizeClassScores(NormType norm, const std::vector<double>& scores, bool clipScores)
{
    return normalizeOverAll(norm, scores, clipScores);
}

void findNormParamsAcrossFeatures(NormType norm, const FeatureVector& featureVector, 
                                  OUT_PARAM double& min, OUT_PARAM double& max, int* posMin, int* posMax)
{
    size_t nFeatures = featureVector.size();
    ASSERT_THROW(nFeatures > 0, "vector cannot be empty");

    // initialization
    min = featureVector[0], max = featureVector[0];
    if (posMin != nullptr)
        *posMin = 0;
    if (posMax != nullptr)
        *posMax = 0;

    // update min/max
    for (size_t f = 1; f < nFeatures; f++)
    {
        if (featureVector[f] < min)
        {
            min = featureVector[f];
            if (posMax != nullptr)
                *posMin = (int)f;
        }
        else if (featureVector[f] > max)
        {
            max = featureVector[f];
            if (posMax != nullptr)
                *posMax = (int)f;
        }
    }
}

void findNormParamsOverAll(NormType norm, const std::vector<FeatureVector>& featureVectors, OUT_PARAM double& param1, OUT_PARAM double& param2)
{
    double foundParam1, foundParam2;
    size_t nSamples = featureVectors.size();
    ASSERT_THROW(nSamples > 0, "vector must contain at least one feature vector");

    if (norm == MIN_MAX)
    {

        for (size_t s = 0; s < nSamples; s++)
        {
            findNormParamsAcrossFeatures(norm, featureVectors[s], foundParam1, foundParam2);
            if (s == 0 || foundParam1 < param1) param1 = foundParam1; // min
            if (s == 0 || foundParam2 > param2) param2 = foundParam2; // max
        }
        return;
    }
    else if (norm == Z_SCORE)
    {        
        
        size_t nFeatures = featureVectors[0].size();
        size_t total = nFeatures * nSamples;

        for (size_t s = 0; s < nSamples; s++)
            for (size_t f = 0; f < nFeatures; f++)
                foundParam1 += featureVectors[s][f];
        foundParam1 /= (double)total;                           // mean

        for (size_t s = 0; s < nSamples; s++)
            for (size_t f = 0; f < nFeatures; f++)
                foundParam2 += (featureVectors[s][f] - foundParam1) * (featureVectors[s][f] - foundParam1);
        foundParam2 = std::sqrt(foundParam2 / (double)total);   // stddev

        param1 = foundParam1;
        param2 = foundParam2;
        return;
    }
    
    throw std::runtime_error("Undefined normalization method");
}

void findNormParamsPerFeature(NormType norm, const std::vector<FeatureVector>& featureVectors, 
                              OUT_PARAM FeatureVector& featuresParam1, OUT_PARAM FeatureVector& featuresParam2)
{
    size_t nSamples = featureVectors.size();
    ASSERT_THROW(nSamples > 0, "vector must contain at least one feature vector");
    size_t nFeatures = featureVectors[0].size();
    ASSERT_THROW(nFeatures > 0, "feature vectors must contain at least one feature");

    if (norm == MIN_MAX)
    {
        // initialize with first vector        
        FeatureVector min = featureVectors[0];
        FeatureVector max = featureVectors[0];

        // find min/max values
        for (size_t v = 1; v < nSamples; v++)
        {
            for (size_t f = 0; f < nFeatures; f++)
            {
                if (featureVectors[v][f] < min[f])
                    min[f] = featureVectors[v][f];
                if (featureVectors[v][f] > max[f])
                    max[f] = featureVectors[v][f];
            }
        }

        // update values
        featuresParam1 = min;
        featuresParam2 = max;
        return;
    }
    else if (norm == Z_SCORE)
    {
        FeatureVector mean(nFeatures), stddev(nFeatures);
        for (size_t f = 0; f < nFeatures; f++)
        {
            for (size_t v = 0; v < nSamples; v++)
                mean[f] += featureVectors[v][f];
            mean[f] /= (double)nSamples;

            for (size_t v = 0; v < nSamples; v++)
                stddev[f] += (featureVectors[v][f] - mean[f]) * (featureVectors[v][f] - mean[f]);
            stddev[f] = std::sqrt(stddev[f] / (double)nSamples);
        }

        // update values
        featuresParam1 = mean;
        featuresParam2 = stddev;
        return;
    }
    
    throw std::runtime_error("Undefined normalization method");
}

void findNormParamsClassScores(NormType norm, const std::vector<double>& scores, OUT_PARAM double& param1, OUT_PARAM double& param2)
{    
    findNormParamsAcrossFeatures(norm, scores, param1, param2);
}
