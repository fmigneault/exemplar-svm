#include "helperFunctions.h"

#define ASSERT_MSG(cond, msg) do \
{ if (!(cond)) { std::ostringstream str; str << msg; std::cerr << str.str(); std::abort(); } \
} while(0)

cv::Mat imReadAndDisplay(std::string imagePath, std::string windowName, cv::ImreadModes readMode)
{
    std::cout << "Reading image: " << imagePath << std::endl;
    cv::Mat img = cv::imread(imagePath, readMode);
    cv::imshow(windowName, img);
    cv::waitKey(1); // allow window redraw
    return img;
}

cv::Mat imTranslate(cv::Mat image, cv::Point offset)
{
    cv::Rect source = cv::Rect(cv::max(0, -offset.x), cv::max(0, -offset.y), image.cols - abs(offset.x), image.rows - abs(offset.y));
    cv::Rect target = cv::Rect(cv::max(0, offset.x), cv::max(0, offset.y), image.cols - abs(offset.x), image.rows - abs(offset.y));
    cv::Mat trans = cv::Mat::zeros(image.size(), image.type());
    image(source).copyTo(trans(target));
    return trans;
}

cv::Mat imFlip(cv::Mat image, FlipCode flipCode)
{
    cv::Mat flip;
    cv::flip(image, flip, flipCode);
    return flip;
}

std::vector<cv::Mat> imSyntheticGeneration(cv::Mat image)
{
    std::vector<cv::Mat> synth(6);
    synth[0] = image;
    synth[1] = imTranslate(image, cv::Point(4, 0));
    synth[2] = imTranslate(image, cv::Point(0, 4));
    synth[3] = imTranslate(image, cv::Point(-4, 0));
    synth[4] = imTranslate(image, cv::Point(0, -4));
    synth[5] = imFlip(image, HORIZONTAL);
    return synth;
}

std::vector<cv::Mat> imSplitPatches(cv::Mat image, cv::Size patchCounts)
{
    if (patchCounts == cv::Size(0, 0) || patchCounts == cv::Size(1, 1))
    {
        std::vector<cv::Mat> vImg(1);
        vImg[0] = image;
        return vImg;
    }
    else if (image.size().width % patchCounts.width == 0 && image.size().height % patchCounts.height == 0)
    {
        // Define and return image patches
        cv::Size patchSize(image.size().width / patchCounts.width, image.size().height / patchCounts.height);
        std::vector<cv::Mat> patches(patchCounts.width * patchCounts.height);
        int i = 0;
        for (int r = 0; r < image.rows; r += patchSize.height)
            for (int c = 0; c < image.cols; c += patchSize.width)
                patches[i++] = image(cv::Range(r, r + patchSize.height), cv::Range(c, c + patchSize.width));
        return patches;
    }
    return std::vector<cv::Mat>();
}

std::vector<cv::Mat> imPreprocess(std::string imagePath, cv::Size imSize, cv::Size patchCounts, std::string windowName, cv::ImreadModes readMode)
{
    cv::Mat img = imReadAndDisplay(imagePath, windowName, readMode);
    if (readMode == cv::IMREAD_COLOR || img.channels() > 1)
        cv::cvtColor(img, img, CV_BGR2GRAY);
    cv::resize(img, img, imSize, 0, 0, cv::INTER_CUBIC);
    cv::equalizeHist(img, img);
    return imSplitPatches(img, patchCounts);
}

double normalizeMinMax(double value, double min, double max)
{
    ASSERT_MSG(max > min, "max must be greater than min (max: " + std::to_string(max) + ", min: " + std::to_string(min) + ")");
    return (value - min) / (max - min);
}

void findMinMax(FeatureVector vector, double* min, double* max, int* posMin, int* posMax)
{
    // check values/references
    ASSERT_MSG(min != nullptr, "min reference not specified");
    ASSERT_MSG(max != nullptr, "max reference not specified");

    int nFeatures = vector.size();
    ASSERT_MSG(nFeatures > 0, "vector cannot be empty");

    // initialization
    *min = vector[0] , *max = vector[0];
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
        else if(vector[f] > *max)
        {
            *max = vector[f];
            if (posMax != nullptr)
                *posMax = f;
        }
    }
}

void findMinMaxFeatures(std::vector<FeatureVector> featureVectors, FeatureVector* minFeatures, FeatureVector* maxFeatures)
{
    ASSERT_MSG(minFeatures != nullptr, "feature vector for min features not specified");
    ASSERT_MSG(maxFeatures != nullptr, "feature vector for max features not specified");

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

FeatureVector normalizeMinMaxPerFeatures(FeatureVector featureVector, FeatureVector minFeatures, FeatureVector maxFeatures)
{
    // check number of features
    int nFeatures = featureVector.size();
    ASSERT_MSG(minFeatures.size() == nFeatures, "min features dimension doesn't match feature vector to normalize");
    ASSERT_MSG(maxFeatures.size() == nFeatures, "max features dimension doesn't match feature vector to normalize");

    // normalize values    
    for (int f = 0; f < nFeatures; f++)
        featureVector[f] = normalizeMinMax(featureVector[f], minFeatures[f], maxFeatures[f]);

    return featureVector;
}

std::string currentTimeStamp()
{
    time_t now = time(0);
    char dt[64];
    ctime_s(dt, sizeof dt, &now);
    return std::string(dt);
}

logstream::logstream(std::string filepath)
{
    // Open in append mode to log continously from different functions open/close calls
    coss.open(filepath, std::fstream::app);
}

logstream::~logstream(void)
{
    if (coss.is_open()) coss.close();
}

logstream& logstream::operator<<(std::ostream& (*pfun)(std::ostream&))
{
    pfun(coss);
    pfun(std::cout);
    return *this;
}

std::string featuresToVectorString(FeatureVector features)
{
    std::string s = "[" + std::to_string(features.size()) + "] {";
    for (int f = 0; f < features.size(); f++)
    {
        if (f != 0) s += ", ";
        s += std::to_string(features[f]);
    }
    s += "}";
    return s;
}

std::string featuresToSvmString(FeatureVector features, int label)
{
    std::string s = std::to_string(label) + " ";
    for (int f = 0; f < features.size(); f++)
    {
        if (f != 0) s += " ";
        s += std::to_string(f + 1) + ":";
        s += std::to_string(features[f]);
    }
    return s;
}
