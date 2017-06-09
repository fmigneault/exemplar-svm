#include "esvmEnsemble.h"
#include "esvmOptions.h"
#include "generic.h"
#include "norm.h"
#include "imgUtils.h"

#include <fstream>
#include <sstream>

/*
    Initializes an Ensemble of ESVM (EoESVM)
*/
esvmEnsemble::esvmEnsemble(const std::vector<std::vector<cv::Mat> >& positiveROIs, const std::string negativesDir,
                           const std::vector<std::string>& positiveIDs, const std::vector<cv::Mat>& additionalNegativeROIs)
{
    setConstants(negativesDir);
    size_t nPositives = positiveROIs.size();
    size_t nPatches = getPatchCount();
    if (positiveIDs.size() == nPositives)
        enrolledPositiveIDs = positiveIDs;
    else
    {
        enrolledPositiveIDs = std::vector<std::string>(nPositives);
        for (size_t pos = 0; pos < nPositives; ++pos)
            enrolledPositiveIDs[pos] = std::to_string(pos);
    }    

    // positive samples
    size_t dimsPositives[3]{ nPatches, nPositives, 0 };
    xstd::mvector<3, FeatureVector> posSamples(dimsPositives);          // [patch][positives][representation](FeatureVector)

    // Exemplar-SVM
    EoESVM = xstd::mvector<2, ESVM>(dimsPositives);                     // [patch][positive](ESVM)

    // load positive target still images, extract features and normalize
    for (size_t pos = 0; pos < nPositives; ++pos)
    {
        // apply operations for each positive target representation
        size_t nRepresentations = positiveROIs[pos].size();
        for (size_t p = 0; p < nPatches; ++p)
            posSamples[p][pos] = std::vector<FeatureVector>(nRepresentations);
        for (size_t r = 0; r < nRepresentations; ++r)
        {
            // apply pre-processing operation as required
            #if ESVM_ROI_PREPROCESS_MODE == 2
            cv::Mat roi = imCropByRatio(positiveROIs[pos][r], ESVM_ROI_CROP_RATIO, CENTER_MIDDLE);
            #else
            cv::Mat roi = positiveROIs[pos][r];
            #endif/*ESVM_ROI_PREPROCESS_MODE*/

            std::vector<cv::Mat> patches = imPreprocess(roi, imageSize, patchCounts, ESVM_USE_HISTOGRAM_EQUALIZATION);
            for (size_t p = 0; p < nPatches; ++p)
            {
                posSamples[p][pos][r] = hog.compute(patches[p]);
                #if ESVM_FEATURE_NORMALIZATION_MODE == 1
                posSamples[p][pos][r] = normalizeOverAll(MIN_MAX, posSamples[p][pos][r], hogRefMin, hogRefMax, ESVM_FEATURE_NORMALIZATION_CLIP);
                #elif ESVM_FEATURE_NORMALIZATION_MODE == 2
                posSamples[p][pos][r] = normalizeOverAll(Z_SCORE, posSamples[p][pos][r], hogRefMean, hogRefStdDev, ESVM_FEATURE_NORMALIZATION_CLIP);
                #elif ESVM_FEATURE_NORMALIZATION_MODE == 3
                posSamples[p][pos][r] = normalizePerFeature(MIN_MAX, posSamples[p][pos][r], hogRefMin, hogRefMax, ESVM_FEATURE_NORMALIZATION_CLIP);
                #elif ESVM_FEATURE_NORMALIZATION_MODE == 4
                posSamples[p][pos][r] = normalizePerFeature(Z_SCORE, posSamples[p][pos][r], hogRefMean, hogRefStdDev, ESVM_FEATURE_NORMALIZATION_CLIP);
                #elif ESVM_FEATURE_NORMALIZATION_MODE == 5
                posSamples[p][pos][r] = normalizeOverAll(MIN_MAX, posSamples[p][pos][r], hogRefMin[p], hogRefMax[p], ESVM_FEATURE_NORMALIZATION_CLIP);
                #elif ESVM_FEATURE_NORMALIZATION_MODE == 6
                posSamples[p][pos][r] = normalizeOverAll(Z_SCORE, posSamples[p][pos][r][r], hogRefMean[p], hogRefStdDev[p], ESVM_FEATURE_NORMALIZATION_CLIP);
                #elif ESVM_FEATURE_NORMALIZATION_MODE == 7
                posSamples[p][pos][r] = normalizePerFeature(MIN_MAX, posSamples[p][pos][r], hogRefMin[p], hogRefMax[p], ESVM_FEATURE_NORMALIZATION_CLIP);
                #elif ESVM_FEATURE_NORMALIZATION_MODE == 8
                posSamples[p][pos][r] = normalizePerFeature(Z_SCORE, posSamples[p][pos][r], hogRefMean[p], hogRefStdDev[p], ESVM_FEATURE_NORMALIZATION_CLIP);
                #endif/*ESVM_FEATURE_NORMALIZATION_MODE*/
            }
        }
    }

    // extract features and normalize from additional negatives if specified
    size_t nAdditionalNegatives = additionalNegativeROIs.size();
    size_t dimsPositives[2]{ nPatches, nAdditionalNegatives };
    xstd::mvector<2, FeatureVector> negSamples(dimsNegatives);    // [patch][negatives](FeatureVector)
    if (nAdditionalNegatives > 0) 
    {
        for (size_t neg = 0; neg < nAdditionalNegatives; ++neg) 
        {
            for (size_t p = 0; p < nPatches; ++p)
            {
                negSamples[p][neg] = hog.compute(patches[p]);
                #if ESVM_FEATURE_NORMALIZATION_MODE == 1
                negSamples[p][neg] = normalizeOverAll(MIN_MAX, negSamples[p][neg], hogRefMin, hogRefMax, ESVM_FEATURE_NORMALIZATION_CLIP);
                #elif ESVM_FEATURE_NORMALIZATION_MODE == 2
                negSamples[p][neg] = normalizeOverAll(Z_SCORE, negSamples[p][neg], hogRefMean, hogRefStdDev, ESVM_FEATURE_NORMALIZATION_CLIP);
                #elif ESVM_FEATURE_NORMALIZATION_MODE == 3
                negSamples[p][neg] = normalizePerFeature(MIN_MAX, negSamples[p][neg], hogRefMin, hogRefMax, ESVM_FEATURE_NORMALIZATION_CLIP);
                #elif ESVM_FEATURE_NORMALIZATION_MODE == 4
                negSamples[p][neg] = normalizePerFeature(Z_SCORE, negSamples[p][neg], hogRefMean, hogRefStdDev, ESVM_FEATURE_NORMALIZATION_CLIP);
                #elif ESVM_FEATURE_NORMALIZATION_MODE == 5
                negSamples[p][neg] = normalizeOverAll(MIN_MAX, negSamples[p][neg], hogRefMin[p], hogRefMax[p], ESVM_FEATURE_NORMALIZATION_CLIP);
                #elif ESVM_FEATURE_NORMALIZATION_MODE == 6
                negSamples[p][neg] = normalizeOverAll(Z_SCORE, negSamples[p][neg][r], hogRefMean[p], hogRefStdDev[p], ESVM_FEATURE_NORMALIZATION_CLIP);
                #elif ESVM_FEATURE_NORMALIZATION_MODE == 7
                negSamples[p][neg] = normalizePerFeature(MIN_MAX, negSamples[p][neg], hogRefMin[p], hogRefMax[p], ESVM_FEATURE_NORMALIZATION_CLIP);
                #elif ESVM_FEATURE_NORMALIZATION_MODE == 8
                negSamples[p][neg] = normalizePerFeature(Z_SCORE, negSamples[p][neg], hogRefMean[p], hogRefStdDev[p], ESVM_FEATURE_NORMALIZATION_CLIP);
                #endif/*ESVM_FEATURE_NORMALIZATION_MODE*/
            }
        }
    }

    // training
    for (size_t p = 0; p < nPatches; ++p)
    {
        /* note:
                we re-assign the negative samples per patch individually and sequentially clear them as loading them all
                simultaneously can sometimes be hard on the available memory if a LOT of negatives are employed
        */
        
        // load negative samples from pre-generated files for training (samples in files are pre-normalized)
        #if ESVM_FEATURE_NORMALIZATION_MODE == 0
        std::string negativeFileName = "negatives-patch" + std::to_string(p) + "-raw" + sampleFileExt;
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 1
        std::string negativeFileName = "negatives-patch" + std::to_string(p) + "-normROI-minmax-overAll" + sampleFileExt;
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 2
        std::string negativeFileName = "negatives-patch" + std::to_string(p) + "-normROI-zscore-overAll" + sampleFileExt;
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 3
        std::string negativeFileName = "negatives-patch" + std::to_string(p) + "-normROI-minmax-perFeat" + sampleFileExt;
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 4
        std::string negativeFileName = "negatives-patch" + std::to_string(p) + "-normROI-zscore-perFeat" + sampleFileExt;
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 5
        std::string negativeFileName = "negatives-patch" + std::to_string(p) + "-normPatch-minmax-overAll" + sampleFileExt;
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 6
        std::string negativeFileName = "negatives-patch" + std::to_string(p) + "-normPatch-zscore-overAll" + sampleFileExt;
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 7
        std::string negativeFileName = "negatives-patch" + std::to_string(p) + "-normPatch-minmax-perFeat" + sampleFileExt;
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 8
        std::string negativeFileName = "negatives-patch" + std::to_string(p) + "-normPatch-zscore-perFeat" + sampleFileExt;
        #endif/*ESVM_FEATURE_NORMALIZATION_MODE*/
        
        std::vector<FeatureVector> negFileSamples;  // reset on each patch
        ESVM::readSampleDataFile(negativesDir + negativeFileName, negFileSamples, sampleFileFormat);
        negSamples[p].insert(negSamples[p].end(), negFileSamples.begin(), negFileSamples.end());

        for (size_t pos = 0; pos < nPositives; ++pos)
            EoESVM[p][pos] = ESVM(posSamples[p][pos], negSamples[p], enrolledPositiveIDs[pos] + "-patch" + std::to_string(p));

        negSamples[p].clear();
    }
}

void esvmEnsemble::setConstants(std::string negativesDir)
{
    imageSize = cv::Size(48, 48);
    patchCounts = cv::Size(3, 3);
    blockSize = cv::Size(2, 2);
    blockStride = cv::Size(2, 2);
    cellSize = cv::Size(2, 2);
    nBins = 3;
    windowSize = cv::Size(imageSize.width / patchCounts.width, imageSize.height / patchCounts.height);
    hog = FeatureExtractorHOG(windowSize, blockSize, blockStride, cellSize, nBins);

    /* --- Feature 'hardcoded' normalization values for on-line classification --- */

    #if ESVM_FEATURE_NORMALIZATION_MODE == 1    // Min-Max overall normalization across patches
        // found min/max using 'FullChokePoint' test with SAMAN pre-generated files
    hogRefMin = 0;
    hogRefMax = 0.675058;
    // found min/max using 'FullGenerationAndTestProcess' test (loaded ROI from ChokePoint + Fast-DT ROI localized search)
    hogRefMin = 0;
    hogRefMax = 0.704711;

    // found min/max using 'proc_createNegativesSampleFiles' with 'PROC_ESVM_GENERATE_SAMPLE_FILES'=1 (S1 only)
    // generate AFTER fix of patch split / data pointer access of patches for HOG
    hogRefMin = 0;
    hogRefMax = 0.766903;

    // found min/max using 'create_negatives' procedure with all ChokePoint available ROIs that match the specified negative IDs (35276 samples)
    // feature extraction is executed using the same pre-process as on-line execution (with HistEqual = 0)
    ///hogRefMin = 0;
    ///hogRefMax = 0.682703;

    // found min/max using 'create_negatives' procedure with all ChokePoint available ROIs that match the specified negative IDs (11344 samples)
    // uses the 'LBP improved' localized face ROI refinement for more focused face details / less background noise
    // feature extraction is executed using the same pre-process as on-line execution (with HistEqual = 1)
    ///hogRefMin = 0;
    ///hogRefMax = 0.695519;
    #elif ESVM_FEATURE_NORMALIZATION_MODE == 2  // Z-Score overall normalization across patches
    THROW("Not set reference normalization values (ESVM_FEATURE_NORMALIZATION_MODE == 2)");
    #elif ESVM_FEATURE_NORMALIZATION_MODE == 3  // Min-Max per feature normalization across patches
    THROW("Not set reference normalization values (ESVM_FEATURE_NORMALIZATION_MODE == 3)");
    hogRefMin = {};
    hogRefMax = {};
    #elif ESVM_FEATURE_NORMALIZATION_MODE == 4  // Z-Score per feature normalization across patches
    THROW("Not set reference normalization values (ESVM_FEATURE_NORMALIZATION_MODE == 4)");
    #elif ESVM_FEATURE_NORMALIZATION_MODE == 5  // Min-Max overall normalization for each patch
    THROW("Not set reference normalization values (ESVM_FEATURE_NORMALIZATION_MODE == 5)");
    #elif ESVM_FEATURE_NORMALIZATION_MODE == 6  // Z-Score overall normalization for each patch
    THROW("Not set reference normalization values (ESVM_FEATURE_NORMALIZATION_MODE == 6)");
    #elif ESVM_FEATURE_NORMALIZATION_MODE == 7  // Min-Max per feature normalization for each patch
    ESVM::readSampleDataFile(negativesDir + "negatives-MIN-normPatch-minmax-perFeat.data", hogRefMin, LIBSVM);
    ESVM::readSampleDataFile(negativesDir + "negatives-MAX-normPatch-minmax-perFeat.data", hogRefMax, LIBSVM);
    #elif ESVM_FEATURE_NORMALIZATION_MODE == 8  // Z-Score per feature normalization for each patch
    THROW("Not set reference normalization values (ESVM_FEATURE_NORMALIZATION_MODE == 8)");
    #endif/*ESVM_FEATURE_NORMALIZATION_MODE*/

    /* --- Score 'hardcoded' normalization values for on-line classification --- */

    #if ESVM_SCORE_NORMALIZATION_MODE == 1      // Min-Max normalization
        // found min/max using 'SimplifiedWorkingProcedure' test with SAMAN pre-generated files
    scoreRefMin = -1.578030;
    scoreRefMax = -0.478968;
    // found min/max using 'FullGenerationAndTestProcess' test (loaded ROI from ChokePoint + Fast-DT ROI localized search)
    scoreRefMin = -5.15837;
    scoreRefMax = 0.156316;

    // found min/max using 'FullGenerationAndTestProcess' AFTER fix of patch split / data pointer access of patches for HOG
    //      S1 - Min / Max: -4.03879 / 0.366612
    //      S3 - Min / Max : -4.16766 / 0.200456
    //      EX - Min / Max : -4.28606 / 0.60522
    scoreRefMin = -4.28606;
    scoreRefMax = 0.60522;

    // found min/max using FAST-DT live test
    ///scoreRefMin = 0.085;         // Testing
    ///scoreRefMin = -0.638025;
    ///scoreRefMax =  0.513050;

    // values found using the LBP Improved run in 'live' and calculated over a large amount of samples
    /* found from test */
    //scoreRefMin = -4.69201;
    //scoreRefMax = 1.46431;
    /* experimental adjustment */
    ///scoreRefMin = -2.929948;
    ///scoreRefMax = -0.731181;
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
    Predicts the classification value for the specified roi using the trained Ensemble of ESVM model.
*/
std::vector<double> esvmEnsemble::predict(const cv::Mat& roi)
{
    size_t nPositives = getPositiveCount();
    size_t nPatches = getPatchCount();

    // apply pre-processing operation as required
    #if ESVM_ROI_PREPROCESS_MODE == 2
    cv::Mat procROI = imCropByRatio(roi, ESVM_ROI_CROP_RATIO, CENTER_MIDDLE);
    #else
    cv::Mat procROI = roi;
    #endif/*ESVM_ROI_PREPROCESS_MODE*/

    // load probe still images, extract features and normalize
    xstd::mvector<1, FeatureVector> probeSampleFeats(nPatches);
    std::vector<cv::Mat> patches = imPreprocess(procROI, imageSize, patchCounts, ESVM_USE_HISTOGRAM_EQUALIZATION);
    for (size_t p = 0; p < nPatches; p++)
    {
        probeSampleFeats[p] = hog.compute(patches[p]);
        #if ESVM_FEATURE_NORMALIZATION_MODE == 1
        probeSampleFeats[p] = normalizeOverAll(MIN_MAX, probeSampleFeats[p], hogRefMin, hogRefMax, ESVM_FEATURE_NORMALIZATION_CLIP);
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 2
        probeSampleFeats[p] = normalizeOverAll(Z_SCORE, probeSampleFeats[p], hogRefMean, hogRefStdDev, ESVM_FEATURE_NORMALIZATION_CLIP);
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 3
        probeSampleFeats[p] = normalizePerFeature(MIN_MAX, probeSampleFeats[p], hogRefMin, hogRefMax, ESVM_FEATURE_NORMALIZATION_CLIP);
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 4
        probeSampleFeats[p] = normalizePerFeature(Z_SCORE, probeSampleFeats[p], hogRefMean, hogRefStdDev, ESVM_FEATURE_NORMALIZATION_CLIP);
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 5
        probeSampleFeats[p] = normalizeOverAll(MIN_MAX, probeSampleFeats[p], hogRefMin[p], hogRefMax[p], ESVM_FEATURE_NORMALIZATION_CLIP);
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 6
        probeSampleFeats[p] = normalizeOverAll(Z_SCORE, probeSampleFeats[p], hogRefMean[p], hogRefStdDev[p], ESVM_FEATURE_NORMALIZATION_CLIP);
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 7
        probeSampleFeats[p] = normalizePerFeature(MIN_MAX, probeSampleFeats[p], hogRefMin[p], hogRefMax[p], ESVM_FEATURE_NORMALIZATION_CLIP);
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 8
        probeSampleFeats[p] = normalizePerFeature(Z_SCORE, probeSampleFeats[p], hogRefMean[p], hogRefStdDev[p], ESVM_FEATURE_NORMALIZATION_CLIP);
        #endif/*ESVM_FEATURE_NORMALIZATION_MODE*/
    }

    // testing, score fusion, normalization
    xstd::mvector<1, double> classificationScores(nPositives, 0.0);
    size_t dimsProbes[2]{ nPatches, nPositives };
    xstd::mvector<2, double> scores(dimsProbes, 0.0);
    for (size_t pos = 0; pos < nPositives; ++pos)
    {
        for (size_t p = 0; p < nPatches; ++p)
        {
            scores[p][pos] = EoESVM[p][pos].predict(probeSampleFeats[p]);
            classificationScores[pos] += scores[p][pos];                          // score accumulation for fusion
        }
        // average score fusion and normalization post-fusion
        classificationScores[pos] /= (double)nPatches;
        #if ESVM_SCORE_NORMALIZATION_MODE == 1
        classificationScores[pos] = normalize(MIN_MAX, classificationScores[pos], scoreRefMin, scoreRefMax, ESVM_SCORE_NORMALIZATION_CLIP);
        #elif ESVM_SCORE_NORMALIZATION_MODE == 2
        classificationScores[pos] = normalize(Z_SCORE, classificationScores[pos], scoreRefMean, scoreRefStdDev, ESVM_SCORE_NORMALIZATION_CLIP);
        #endif/*ESVM_SCORE_NORMALIZATION_MODE*/
    }
    return classificationScores;
}