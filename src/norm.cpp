#include "norm.h"
#include "generic.h"
#include <algorithm>

double normalizeMinMax(double value, double min, double max)
{
    ASSERT_LOG(max > min, "max must be greater than min (max: " + std::to_string(max) + ", min: " + std::to_string(min) + ")");
    return (value - min) / (max - min);
}

void findMinMax(FeatureVector vector, double* min, double* max, int* posMin, int* posMax)
{
    // check values/references
    ASSERT_LOG(min != nullptr, "min reference not specified");
    ASSERT_LOG(max != nullptr, "max reference not specified");

    int nFeatures = vector.size();
    ASSERT_LOG(nFeatures > 0, "vector cannot be empty");

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

void findMinMaxFeatures(std::vector<FeatureVector> featureVectors, FeatureVector* minFeatures, FeatureVector* maxFeatures)
{
    ASSERT_LOG(minFeatures != nullptr, "feature vector for min features not specified");
    ASSERT_LOG(maxFeatures != nullptr, "feature vector for max features not specified");

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
    ASSERT_LOG(minFeatures.size() == nFeatures, "min features dimension doesn't match feature vector to normalize");
    ASSERT_LOG(maxFeatures.size() == nFeatures, "max features dimension doesn't match feature vector to normalize");

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