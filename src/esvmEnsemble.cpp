#include "esvmEnsemble.h"
#include "esvmOptions.h"
#include "generic.h"
#include "norm.h"
#include "imgUtils.h"

#include <fstream>
#include <sstream>

/*
    Initializes an ESVM Ensemble
*/
esvmEnsemble::esvmEnsemble(std::vector<cv::Mat> positiveROIs, std::string negativesDir, std::vector<std::string> positiveIDs)
{ 
    setConstants();
    size_t nPositives = positiveROIs.size();
    size_t nPatches = getPatchCount();
    if (positiveIDs.size() == nPositives)
        enrolledPositiveIDs = positiveIDs;
    else
    {
        enrolledPositiveIDs = std::vector<std::string>(nPositives);
        for (size_t pos = 0; pos < nPositives; pos++)
            enrolledPositiveIDs[pos] = std::to_string(pos);
    }    

    // positive samples
    nPositives = positiveROIs.size();
    size_t dimsPositives[2]{ nPatches, nPositives };
    xstd::mvector<2, FeatureVector> positiveSamples(dimsPositives);     // [patch][positives](FeatureVector)

    // negative samples    
    size_t dimsNegatives[2]{ nPatches, 0 };                             // number of negatives unknown (loaded from file)
    xstd::mvector<2, FeatureVector> negativeSamples(dimsNegatives);

    // Exemplar-SVM    
    EoESVM = xstd::mvector<2, ESVM>(dimsPositives);                     // [patch][positive](ESVM)    

    // load positive target still images, extract features and normalize
    std::cout << "Loading positive image stills, extracting feature vectors and normalizing..." << std::endl;
    for (size_t pos = 0; pos < nPositives; pos++)
    {        
        std::vector<cv::Mat> patches = imPreprocess(positiveROIs[pos], imageSize, patchCounts);
        for (size_t p = 0; p < nPatches; p++)
            positiveSamples[p][pos] = normalizeAllFeatures(MIN_MAX, hog.compute(patches[p]), hogHardcodedFoundMin, hogHardcodedFoundMax);        
    }

    // load negative samples from pre-generated files for training (samples in files are pre-normalized)
    std::cout << "Loading negative samples from files..." << std::endl;
    for (size_t p = 0; p < nPatches; p++)
        ESVM::readSampleDataFile(negativesDir + "negatives-hog-patch" + std::to_string(p) +
                                 sampleFileExt, negativeSamples[p], sampleFileFormat);
    // training
    std::cout << "Training ESVM with positives and negatives..." << std::endl;
    for (size_t p = 0; p < nPatches; p++)
        for (size_t pos = 0; pos < nPositives; pos++)
            EoESVM[p][pos] = ESVM({ positiveSamples[p][pos] }, negativeSamples[p], enrolledPositiveIDs[pos] + "-patch" + std::to_string(p));

}

void esvmEnsemble::setConstants()
{
    imageSize = cv::Size(48, 48);
    patchCounts = cv::Size(3, 3); 
    blockSize = cv::Size(2, 2);
    blockStride  = cv::Size(2, 2);
    cellSize = cv::Size(2, 2);  
    nBins = 3;
    hog = FeatureExtractorHOG(imageSize, blockSize, blockStride, cellSize, nBins);

    hogHardcodedFoundMin = 0;               // Min found using 'FullChokePoint' test with SAMAN pre-generated files
    hogHardcodedFoundMax = 0.675058;        // Max found using 'FullChokePoint' test with SAMAN pre-generated files

    ///scoreHardcodedFoundMin = -1.578030;     // Min found using 'SimplifiedWorkingProcedure' test with SAMAN pre-generated files
    ///scoreHardcodedFoundMax = -0.478968;     // Max found using 'SimplifiedWorkingProcedure' test with SAMAN pre-generated files
    scoreHardcodedFoundMin = 0.085;         // Testing
    ///scoreHardcodedFoundMin = -0.638025;     // Min found using FAST-DT live test 
    scoreHardcodedFoundMax =  0.513050;     // Max found using FAST-DT live test 

    scoresHardCodedFoundMean = -1.26193;
    scoresHardCodedFoundStdDev = 0.247168;

    sampleFileExt = ".bin";
    sampleFileFormat = BINARY;
}

std::string esvmEnsemble::getPositiveID(int positiveIndex)
{
    size_t nPositives = getPositiveCount();
    return (nPositives != 0 && positiveIndex >= 0 && positiveIndex < nPositives) ? enrolledPositiveIDs[positiveIndex] : "";
}

/*
    Predicts the classification value for the specified roi using the trained ESVM model.
*/
std::vector<double> esvmEnsemble::predict(const cv::Mat roi) // this should be a feat vector
{
    size_t nPositives = getPositiveCount();
    size_t nPatches = getPatchCount();

    // load probe still images, extract features and normalize
    std::cout << "Loading probe images, extracting feature vectors and normalizing..." << std::endl;
    xstd::mvector<1, FeatureVector> probeSampleFeats(nPatches);
    std::vector<cv::Mat> patches = imPreprocess(roi, imageSize, patchCounts);
    for (size_t p = 0; p < nPatches; p++)
        probeSampleFeats[p] = normalizeAllFeatures(MIN_MAX, hog.compute(patches[p]), hogHardcodedFoundMin, hogHardcodedFoundMax);

    // testing, score fusion, normalization
    cout << "Testing probe samples against enrolled targets..." << std::endl;
    xstd::mvector<1, double> classificationScores(nPositives, 0.0);
    size_t dimsProbes[2]{ nPatches, nPositives }; 
    xstd::mvector<2, double> scores(dimsProbes, 0.0);
    for (size_t pos = 0; pos < nPositives; pos++) 
    {
        for (size_t p = 0; p < nPatches; p++)
        {                
            scores[p][pos] = EoESVM[p][pos].predict(probeSampleFeats[p]);
            classificationScores[pos] += scores[p][pos];                          // score accumulation for fusion
        }
        // average score fusion and normalization post-fusion
        classificationScores[pos] /= (double)nPatches;
        #if ESVM_SCORE_NORMALIZATION_MODE == 1
        classificationScores[pos] = normalize(MIN_MAX, classificationScores[pos], scoreHardcodedFoundMin, scoreHardcodedFoundMax);
        #elif ESVM_SCORE_NORMALIZATION_MODE == 2
        classificationScores[pos] = normalize(Z_SCORE, classificationScores[pos], scoresHardCodedFoundMean, scoresHardCodedFoundStdDev);
        #endif
    }

    return classificationScores;
}
