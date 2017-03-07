#include "norm.h"
#include "generic.h"
#include <algorithm>
#include <float.h>

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

void findMinMax(FeatureVector vector, double* min, double* max, int* posMin, int* posMax)
{
    // check values/references
    ASSERT_THROW(min != nullptr, "min reference not specified");
    ASSERT_THROW(max != nullptr, "max reference not specified");

    int nFeatures = vector.size();
    ASSERT_THROW(nFeatures > 0, "vector cannot be empty");

    // initialization
    *min = vector[0], *max = vector[0];
    if (posMin != nullptr)
        *posMin = 0;
    if (posMax != nullptr)
        *posMax = 0;

    // update min/max
    for (int f = 1; f < vector.size(); f++)
    {
        if (vector[f] < *min)
        {
            *min = vector[f];
            if (posMax != nullptr)
                *posMin = f;
        }
        else if (vector[f] > *max)
        {
            *max = vector[f];
            if (posMax != nullptr)
                *posMax = f;
        }
    }
}

void findMinMaxOverall(std::vector<FeatureVector> featureVectors, double* min, double* max)
{    
    double minFound, maxFound;
    int nSamples = featureVectors.size();
    for (size_t s = 0; s < nSamples; s++)
    {
        findMinMax(featureVectors[s], &minFound, &maxFound);
        if (s == 0 || minFound < *min) *min = minFound;
        if (s == 0 || maxFound > *max) *max = maxFound;
    }
}

void findMinMaxFeatures(std::vector<FeatureVector> featureVectors, FeatureVector* minFeatures, FeatureVector* maxFeatures)
{
    ASSERT_THROW(minFeatures != nullptr, "feature vector for min features not specified");
    ASSERT_THROW(maxFeatures != nullptr, "feature vector for max features not specified");

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
    *minFeatures = min;
    *maxFeatures = max;
}

FeatureVector normalizeMinMaxAllFeatures(FeatureVector featureVector, double min, double max)
{    
    int nFeatures = featureVector.size();
    for (int f = 0; f < nFeatures; f++)
        featureVector[f] = normalizeMinMax(featureVector[f], min, max);
    return featureVector;
}

FeatureVector normalizeMinMaxAllFeatures(FeatureVector featureVector)
{
    double min, max;
    findMinMax(featureVector, &min, &max);
    return normalizeMinMaxAllFeatures(featureVector, min, max);
}

FeatureVector normalizeMinMaxPerFeatures(FeatureVector featureVector, FeatureVector minFeatures, FeatureVector maxFeatures)
{
    // check number of features
    int nFeatures = featureVector.size();
    ASSERT_THROW(minFeatures.size() == nFeatures, "min features dimension doesn't match feature vector to normalize");
    ASSERT_THROW(maxFeatures.size() == nFeatures, "max features dimension doesn't match feature vector to normalize");

    // normalize values    
    for (int f = 0; f < nFeatures; f++)
        featureVector[f] = normalizeMinMax(featureVector[f], minFeatures[f], maxFeatures[f]);

    return featureVector;
}

std::vector<double> normalizeMinMaxClassScores(std::vector<double> scores)
{
    double minScore, maxScore;
    findMinMax(scores, &minScore, &maxScore);
    return normalizeMinMaxAllFeatures(scores, minScore, maxScore);
}
