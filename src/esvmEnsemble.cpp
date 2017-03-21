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
    logstream logger(LOGGER_FILE);
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

    // Exemplar-SVM
    EoESVM = xstd::mvector<2, ESVM>(dimsPositives);                     // [patch][positive](ESVM)    

    // load positive target still images, extract features and normalize
    logger << "Loading positive image stills, extracting feature vectors and normalizing..." << std::endl;
    for (size_t pos = 0; pos < nPositives; pos++)
    {        
        std::vector<cv::Mat> patches = imPreprocess(positiveROIs[pos], imageSize, patchCounts, useHistEqual);
        for (size_t p = 0; p < nPatches; p++)
            positiveSamples[p][pos] = normalizeAllFeatures(MIN_MAX, hog.compute(patches[p]), hogHardcodedFoundMin, hogHardcodedFoundMax);        
    }

    // training
    logger << "Training ESVM with positives and negatives..." << std::endl;
    for (size_t p = 0; p < nPatches; p++)
    {
        /* note: 
                we re-assign the negative samples per patch individually and sequentially as loading them all 
                simultaneously can sometimes be hard on the available memory if a LOT of negatives are employed
        */
        std::vector<FeatureVector> negativePatchSamples;
        // load negative samples from pre-generated files for training (samples in files are pre-normalized)
        std::string negativeFileName = "negatives-patch" + std::to_string(p) + "-normROI-minmax" + sampleFileExt;
        ///std::string negativeFileName = "negatives-hog-patch" + std::to_string(p) + /*"-fullNorm" +*/ sampleFileExt;
        ESVM::readSampleDataFile(negativesDir + negativeFileName, negativePatchSamples, sampleFileFormat);
        logger << "Training ESVM with positives and negatives..." << std::endl;
        for (size_t pos = 0; pos < nPositives; pos++)
            EoESVM[p][pos] = ESVM({ positiveSamples[p][pos] }, negativePatchSamples, enrolledPositiveIDs[pos] + "-patch" + std::to_string(p));
    }
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

    useHistEqual = true; 

    // found min/max using 'FullChokePoint' test with SAMAN pre-generated files
    ///hogHardcodedFoundMin = 0;
    ///hogHardcodedFoundMax = 0.675058;

    // found min/max using 'create_negatives' procedure with all ChokePoint available ROIs that match the specified negative IDs (35276 samples)
    // feature extraction is executed using the same pre-process as on-line execution (with HistEqual = 0)
    ///hogHardcodedFoundMin = 0;
    ///hogHardcodedFoundMax = 0.682703;

    // found min/max using 'create_negatives' procedure with all ChokePoint available ROIs that match the specified negative IDs (11344 samples)
    // uses the 'LBP improved' localized face ROI refinement for more focused face details / less background noise
    // feature extraction is executed using the same pre-process as on-line execution (with HistEqual = 1)
    hogHardcodedFoundMin = 0;
    hogHardcodedFoundMax = 0.695519;

    // found min/max using 'SimplifiedWorkingProcedure' test with SAMAN pre-generated files
    ///scoresHardcodedFoundMin = -1.578030;
    ///scoresHardcodedFoundMax = -0.478968;
    
    // found min/max using FAST-DT live test 
    scoresHardcodedFoundMin = 0.085;         // Testing
    ///scoresHardcodedFoundMin = -0.638025;
    scoresHardcodedFoundMax =  0.513050;

    ///scoresHardCodedFoundMean = -1.26193;
    ///scoresHardCodedFoundStdDev = 0.247168;

    // values found using the LBP Improved run in 'live' and calculated over a large amount of samples
    /* found from test */
    //scoresHardcodedFoundMin = -4.69201; 
    //scoresHardcodedFoundMax = 1.46431;
    scoresHardcodedFoundMean = -1.61661;
    scoresHardcodedFoundStdDev = 0.712719;
    /* experimental adjustment */
    scoresHardcodedFoundMin = -2.929948; 
    scoresHardcodedFoundMax = -0.731181;

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
    logstream logger(LOGGER_FILE);
    size_t nPositives = getPositiveCount();
    size_t nPatches = getPatchCount();

    // load probe still images, extract features and normalize
    logger << "Loading probe images, extracting feature vectors and normalizing..." << std::endl;
    xstd::mvector<1, FeatureVector> probeSampleFeats(nPatches);
    std::vector<cv::Mat> patches = imPreprocess(roi, imageSize, patchCounts, useHistEqual);
    for (size_t p = 0; p < nPatches; p++)
        probeSampleFeats[p] = normalizeAllFeatures(MIN_MAX, hog.compute(patches[p]), hogHardcodedFoundMin, hogHardcodedFoundMax);

    // testing, score fusion, normalization
    logger << "Testing probe samples against enrolled targets..." << std::endl;
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
        classificationScores[pos] = normalize(MIN_MAX, classificationScores[pos], scoresHardcodedFoundMin, scoresHardcodedFoundMax);
        #elif ESVM_SCORE_NORMALIZATION_MODE == 2
        classificationScores[pos] = normalize(Z_SCORE, classificationScores[pos], scoresHardcodedFoundMean, scoresHardcodedFoundStdDev);
        #endif
    }
    return classificationScores;
}
