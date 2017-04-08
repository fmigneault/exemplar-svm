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
        {
            positiveSamples[p][pos] = hog.compute(patches[p]);
            #if ESVM_FEATURE_NORMALIZATION_MODE == 1
            positiveSamples[p][pos] = normalizeAllFeatures(MIN_MAX, positiveSamples[p][pos], hogRefMin, hogRefMax);
            #elif ESVM_FEATURE_NORMALIZATION_MODE == 2
            positiveSamples[p][pos] = normalizeAllFeatures(Z_SCORE, positiveSamples[p][pos], hogRefMean, hogRefStdDev);
            #elif ESVM_FEATURE_NORMALIZATION_MODE == 3
            positiveSamples[p][pos] = normalizePerFeature(MIN_MAX, positiveSamples[p][pos], hogRefMin, hogRefMax);
            #elif ESVM_FEATURE_NORMALIZATION_MODE == 4
            positiveSamples[p][pos] = normalizePerFeature(Z_SCORE, positiveSamples[p][pos], hogRefMean, hogRefStdDev);
            #endif/*ESVM_FEATURE_NORMALIZATION_MODE*/
        }
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

    /* --- Feature 'hardcoded' normalization values for on-line classification --- */

    #if ESVM_FEATURE_NORMALIZATION_MODE == 1    // Min-Max overall normalization
        // found min/max using 'FullChokePoint' test with SAMAN pre-generated files
        ///hogRefMin = 0;
        ///hogRefMax = 0.675058;

        // found min/max using 'create_negatives' procedure with all ChokePoint available ROIs that match the specified negative IDs (35276 samples)
        // feature extraction is executed using the same pre-process as on-line execution (with HistEqual = 0)
        ///hogRefMin = 0;
        ///hogRefMax = 0.682703;

        // found min/max using 'create_negatives' procedure with all ChokePoint available ROIs that match the specified negative IDs (11344 samples)
        // uses the 'LBP improved' localized face ROI refinement for more focused face details / less background noise
        // feature extraction is executed using the same pre-process as on-line execution (with HistEqual = 1)
        hogRefMin = 0;
        hogRefMax = 0.695519;
    #elif ESVM_FEATURE_NORMALIZATION_MODE == 2  // Z-Score overall normalization

    #elif ESVM_FEATURE_NORMALIZATION_MODE == 3  // Min-Max per feature normalization
        hogRefMin = {};
        hogRefMax = {};
    #elif ESVM_FEATURE_NORMALIZATION_MODE == 4  // Z-Score per feature normalization

    #endif/*ESVM_FEATURE_NORMALIZATION_MODE*/

    /* --- Score 'hardcoded' normalization values for on-line classification --- */

    #if ESVM_SCORE_NORMALIZATION_MODE == 1      // Min-Max normalization
        // found min/max using 'SimplifiedWorkingProcedure' test with SAMAN pre-generated files
        ///scoreRefMin = -1.578030;
        ///scoreRefMax = -0.478968;
    
        // found min/max using FAST-DT live test 
        scoreRefMin = 0.085;         // Testing
        ///scoreRefMin = -0.638025;
        scoreRefMax =  0.513050;

        // values found using the LBP Improved run in 'live' and calculated over a large amount of samples
        /* found from test */
        //scoreRefMin = -4.69201; 
        //scoreRefMax = 1.46431;
        /* experimental adjustment */
        scoreRefMin = -2.929948; 
        scoreRefMax = -0.731181;
    #elif ESVM_SCORE_NORMALIZATION_MODE == 2    // Z-Score normalization
        // found using FAST-DT live test 
        ///scoreRefMean = -1.26193;
        ///scoreRefStdDev = 0.247168;
        // values found using the LBP Improved run in 'live' and calculated over a large amount of samples
        scoreRefMean = -1.61661;
        scoreRefStdDev = 0.712719;
    #endif/*ESVM_SCORE_NORMALIZATION_MODE*/

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
    std::vector<cv::Mat> patches = imPreprocess(roi, imageSize, patchCounts, ESVM_USE_HISTOGRAM_EQUALIZATION);
    for (size_t p = 0; p < nPatches; p++) 
    {
        #if ESVM_FEATURE_NORMALIZATION_MODE == 1
        probeSampleFeats[p] = normalizeAllFeatures(MIN_MAX, hog.compute(patches[p]), hogRefMin, hogRefMax);
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 2
        probeSampleFeats[p] = normalizeAllFeatures(Z_SCORE, hog.compute(patches[p]), hogRefMean, hogRefStdDev);
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 3
        probeSampleFeats[p] = normalizePerFeature(MIN_MAX, hog.compute(patches[p]), hogRefMin, hogRefMax);
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 4
        probeSampleFeats[p] = normalizePerFeature(Z_SCORE, hog.compute(patches[p]), hogRefMean, hogRefStdDev);
        #endif
    }

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
        classificationScores[pos] = normalize(MIN_MAX, classificationScores[pos], scoreRefMin, scoreRefMax);
        #elif ESVM_SCORE_NORMALIZATION_MODE == 2
        classificationScores[pos] = normalize(Z_SCORE, classificationScores[pos], scoreRefMean, scoreRefStdDev);
        #endif
    }
    return classificationScores;
}
