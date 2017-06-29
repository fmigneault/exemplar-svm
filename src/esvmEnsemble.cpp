#include "esvmEnsemble.h"
#include "esvmOptions.h"

#include "imgUtils.h"
#include "logging.h"
#include "norm.h"

#include <fstream>
#include <sstream>

/*
    Initializes an Ensemble of ESVM (EoESVM)
*/
esvmEnsemble::esvmEnsemble(const std::vector<std::vector<cv::Mat> >& positiveROIs, const std::string negativesDir,
                           const std::vector<std::string>& positiveIDs, const std::vector<std::vector<cv::Mat> >& additionalNegativeROIs)
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

    // additional negative samples    
    size_t dimsNegatives[3]{ nPatches, nPositives, 0 };
    xstd::mvector<3, FeatureVector> negSamples(dimsNegatives);          // [patch][positives][negatives](FeatureVector)
    size_t nAdditionalNegatives = additionalNegativeROIs.size();

    // Ensemble of exemplar-SVM
    #if ESVM_RANDOM_SUBSPACE_METHOD > 0
    size_t dimsESVM[2]{ nPatches * ESVM_RANDOM_SUBSPACE_METHOD, nPositives };
    #else
    size_t dimsESVM[2]{ nPatches, nPositives };
    #endif/*ESVM_RANDOM_SUBSPACE_METHOD*/
    EoESVM = xstd::mvector<2, ESVM>(dimsESVM);                          // [patch|random-subspace][positive](ESVM)

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
                #if   ESVM_FEATURE_NORMALIZATION_MODE == 1
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

        // extract features and normalize from additional negatives if specified and matching positives to enroll
        if (nAdditionalNegatives == nPositives)
        {
            size_t nNegatives = additionalNegativeROIs[pos].size();
            for (size_t p = 0; p < nPatches; ++p)
                negSamples[p][pos] = std::vector<FeatureVector>(nNegatives);
            for (size_t neg = 0; neg < nNegatives; ++neg)
            {
                // apply pre-processing operation as required
                #if ESVM_ROI_PREPROCESS_MODE == 2
                cv::Mat roi = imCropByRatio(additionalNegativeROIs[pos][neg], ESVM_ROI_CROP_RATIO, CENTER_MIDDLE);
                #else
                cv::Mat roi = additionalNegativeROIs[pos][neg];
                #endif/*ESVM_ROI_PREPROCESS_MODE*/

                std::vector<cv::Mat> patches = imPreprocess(roi, imageSize, patchCounts, ESVM_USE_HISTOGRAM_EQUALIZATION);
                for (size_t p = 0; p < nPatches; ++p)
                {
                    negSamples[p][pos][neg] = hog.compute(patches[p]);
                    #if   ESVM_FEATURE_NORMALIZATION_MODE == 1
                    negSamples[p][pos][neg] = normalizeOverAll(MIN_MAX, negSamples[p][pos][neg], hogRefMin, hogRefMax, ESVM_FEATURE_NORMALIZATION_CLIP);
                    #elif ESVM_FEATURE_NORMALIZATION_MODE == 2
                    negSamples[p][pos][neg] = normalizeOverAll(Z_SCORE, negSamples[p][pos][neg], hogRefMean, hogRefStdDev, ESVM_FEATURE_NORMALIZATION_CLIP);
                    #elif ESVM_FEATURE_NORMALIZATION_MODE == 3
                    negSamples[p][pos][neg] = normalizePerFeature(MIN_MAX, negSamples[p][pos][neg], hogRefMin, hogRefMax, ESVM_FEATURE_NORMALIZATION_CLIP);
                    #elif ESVM_FEATURE_NORMALIZATION_MODE == 4
                    negSamples[p][pos][neg] = normalizePerFeature(Z_SCORE, negSamples[p][pos][neg], hogRefMean, hogRefStdDev, ESVM_FEATURE_NORMALIZATION_CLIP);
                    #elif ESVM_FEATURE_NORMALIZATION_MODE == 5
                    negSamples[p][pos][neg] = normalizeOverAll(MIN_MAX, negSamples[p][pos][neg], hogRefMin[p], hogRefMax[p], ESVM_FEATURE_NORMALIZATION_CLIP);
                    #elif ESVM_FEATURE_NORMALIZATION_MODE == 6
                    negSamples[p][pos][neg] = normalizeOverAll(Z_SCORE, negSamples[p][pos][neg][r], hogRefMean[p], hogRefStdDev[p], ESVM_FEATURE_NORMALIZATION_CLIP);
                    #elif ESVM_FEATURE_NORMALIZATION_MODE == 7
                    negSamples[p][pos][neg] = normalizePerFeature(MIN_MAX, negSamples[p][pos][neg], hogRefMin[p], hogRefMax[p], ESVM_FEATURE_NORMALIZATION_CLIP);
                    #elif ESVM_FEATURE_NORMALIZATION_MODE == 8
                    negSamples[p][pos][neg] = normalizePerFeature(Z_SCORE, negSamples[p][pos][neg], hogRefMean[p], hogRefStdDev[p], ESVM_FEATURE_NORMALIZATION_CLIP);
                    #endif/*ESVM_FEATURE_NORMALIZATION_MODE*/
                }
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
        #if   ESVM_FEATURE_NORMALIZATION_MODE == 0
        std::string negativeFileName = "negatives-raw-patch" + std::to_string(p) + sampleFileExt;
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 1
        std::string negativeFileName = "negatives-normROI-minmax-overAll-patch" + std::to_string(p) + sampleFileExt;
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 2
        std::string negativeFileName = "negatives-normROI-zscore-overAll-patch" + std::to_string(p) + sampleFileExt;
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 3
        std::string negativeFileName = "negatives-normROI-minmax-perFeat-patch" + std::to_string(p) + sampleFileExt;
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 4
        std::string negativeFileName = "negatives-normROI-zscore-perFeat-patch" + std::to_string(p) + sampleFileExt;
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 5
        std::string negativeFileName = "negatives-normPatch-minmax-overAll-patch" + std::to_string(p) + sampleFileExt;
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 6
        std::string negativeFileName = "negatives-normPatch-zscore-overAll-patch" + std::to_string(p) + sampleFileExt;
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 7
        std::string negativeFileName = "negatives-normPatch-minmax-perFeat-patch" + std::to_string(p) + sampleFileExt;
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 8
        std::string negativeFileName = "negatives-normPatch-zscore-perFeat-patch" + std::to_string(p) + sampleFileExt;
        #endif/*ESVM_FEATURE_NORMALIZATION_MODE*/
        
        std::vector<FeatureVector> negFileSamples;  // reset on each patch
        DataFile::readSampleDataFile(negativesDir + negativeFileName, negFileSamples, sampleFileFormat, ESVM_BINARY_HEADER_SAMPLES);        

        for (size_t pos = 0; pos < nPositives; ++pos) {
            negSamples[p][pos].insert(negSamples[p][pos].end(), negFileSamples.begin(), negFileSamples.end());
            std::string idESVM =  enrolledPositiveIDs[pos] + "-patch" + std::to_string(p);

            #if ESVM_RANDOM_SUBSPACE_METHOD == 0
            EoESVM[p][pos] = ESVM(posSamples[p][pos], negSamples[p][pos], idESVM);
            
            #else/*ESVM_RANDOM_SUBSPACE_METHOD*/
            
            // transfer features from random selection
            size_t nPosRS = posSamples[p][pos].size();
            size_t nNegRS = negSamples[p][pos].size();
            xstd::mvector<2, FeatureVector> posSamplesRS({ ESVM_RANDOM_SUBSPACE_METHOD, nPosRS });
            xstd::mvector<2, FeatureVector> negSamplesRS({ ESVM_RANDOM_SUBSPACE_METHOD, nNegRS });
            #pragma omp parallel for
            for (long rs = 0; rs < ESVM_RANDOM_SUBSPACE_METHOD; ++rs) {
                for (size_t f = 0; f < ESVM_RANDOM_SUBSPACE_FEATURES; ++f) {
                    for (size_t iPosRS = 0; iPosRS < nPosRS; ++iPosRS)
                        posSamplesRS[rs][iPosRS][f] = posSamples[p][pos][iPosRS][rsmFeatureIndexes[rs][f]];
                    for (size_t iNegRS = 0; iNegRS < nPosRS; ++iNegRS)
                        negSamplesRS[rs][iNegRS][f] = negSamples[p][pos][iNegRS][rsmFeatureIndexes[rs][f]];
                }
            }

            // train with random subspaces
            for (size_t rs = 0; rs < ESVM_RANDOM_SUBSPACE_METHOD; ++rs) {
                idESVM += "-rs" + std::to_string(rs);
                EoESVM[p * ESVM_RANDOM_SUBSPACE_METHOD + rs][pos] = ESVM(posSamplesRS[rs], negSamplesRS[rs], idESVM);
            }

            #endif/*ESVM_RANDOM_SUBSPACE_METHOD*/
            
            negSamples[p][pos].clear();
        }
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
    DataFile::readSampleDataFile(negativesDir + "negatives-normPatch-minmax-perFeat-MIN.data", hogRefMin, LIBSVM);
    DataFile::readSampleDataFile(negativesDir + "negatives-normPatch-minmax-perFeat-MAX.data", hogRefMax, LIBSVM);
    #elif ESVM_FEATURE_NORMALIZATION_MODE == 8  // Z-Score per feature normalization for each patch
    THROW("Not set reference normalization values (ESVM_FEATURE_NORMALIZATION_MODE == 8)");
    #endif/*ESVM_FEATURE_NORMALIZATION_MODE*/

    /* --- Random Subspace Method for feature selection and compact pool generation --- */

    #if ESVM_RANDOM_SUBSPACE_METHOD

    // read file containing indexes of features to employ from each corresponding random subspace
    // zero-value features are ignored, others are taken as part of the random subspace
    std::vector<FeatureVector> rsmIndexes;
    DataFile::readSampleDataFile("rsm-indexes.data", rsmIndexes, LIBSVM);
    ASSERT_THROW(rsmIndexes.size() == ESVM_RANDOM_SUBSPACE_METHOD, "Incorrect number of RSM subspaces");

    size_t dimsRSM[2]{ ESVM_RANDOM_SUBSPACE_METHOD, ESVM_RANDOM_SUBSPACE_FEATURES };
    rsmFeatureIndexes = xstd::mvector<2, int>(dimsRSM, 0);
    for (size_t rs = 0; rs < rsmIndexes.size(); ++rs) {
        size_t iFeat = 0;
        for (size_t f = 0; f < rsmIndexes[rs].size(); ++f) {
            if (rsmIndexes[rs][f] != 0) {
                rsmFeatureIndexes[rs][iFeat] = f;
                iFeat++;
            }
        }
        ASSERT_THROW(iFeat == ESVM_RANDOM_SUBSPACE_FEATURES, "Incorrect number of RSM features");
    }
    
    #endif/*ESVM_RANDOM_SUBSPACE_METHOD*/

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
    std::vector<FeatureVector> probeSamples(nPatches);
    std::vector<cv::Mat> patches = imPreprocess(procROI, imageSize, patchCounts, ESVM_USE_HISTOGRAM_EQUALIZATION);
    for (size_t p = 0; p < nPatches; p++)
    {
        probeSamples[p] = hog.compute(patches[p]);
        #if   ESVM_FEATURE_NORMALIZATION_MODE == 1
        probeSamples[p] = normalizeOverAll(MIN_MAX, probeSamples[p], hogRefMin, hogRefMax, ESVM_FEATURE_NORMALIZATION_CLIP);
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 2
        probeSamples[p] = normalizeOverAll(Z_SCORE, probeSamples[p], hogRefMean, hogRefStdDev, ESVM_FEATURE_NORMALIZATION_CLIP);
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 3
        probeSamples[p] = normalizePerFeature(MIN_MAX, probeSamples[p], hogRefMin, hogRefMax, ESVM_FEATURE_NORMALIZATION_CLIP);
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 4
        probeSamples[p] = normalizePerFeature(Z_SCORE, probeSamples[p], hogRefMean, hogRefStdDev, ESVM_FEATURE_NORMALIZATION_CLIP);
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 5
        probeSamples[p] = normalizeOverAll(MIN_MAX, probeSamples[p], hogRefMin[p], hogRefMax[p], ESVM_FEATURE_NORMALIZATION_CLIP);
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 6
        probeSamples[p] = normalizeOverAll(Z_SCORE, probeSamples[p], hogRefMean[p], hogRefStdDev[p], ESVM_FEATURE_NORMALIZATION_CLIP);
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 7
        probeSamples[p] = normalizePerFeature(MIN_MAX, probeSamples[p], hogRefMin[p], hogRefMax[p], ESVM_FEATURE_NORMALIZATION_CLIP);
        #elif ESVM_FEATURE_NORMALIZATION_MODE == 8
        probeSamples[p] = normalizePerFeature(Z_SCORE, probeSamples[p], hogRefMean[p], hogRefStdDev[p], ESVM_FEATURE_NORMALIZATION_CLIP);
        #endif/*ESVM_FEATURE_NORMALIZATION_MODE*/
    }

    // prepare test samples    
    # if !ESVM_RANDOM_SUBSPACE_METHOD
    size_t nESVM = nPatches;
    std::vector<FeatureVector> probeSampleTest = probeSamples;
    #else/*ESVM_RANDOM_SUBSPACE_METHOD*/
    size_t nESVM = nPatches * ESVM_RANDOM_SUBSPACE_METHOD;
    std::vector<FeatureVector> probeSampleTest(nESVM);
    #pragma omp parallel for
    for (long p = 0; p < nPatches; ++p)
        for (size_t rs = 0; rs < ESVM_RANDOM_SUBSPACE_METHOD; ++rs) {
            size_t iRS = p * ESVM_RANDOM_SUBSPACE_METHOD + rs;
            probeSampleTest[iRS] = FeatureVector(ESVM_RANDOM_SUBSPACE_FEATURES);
            for (size_t f = 0; f < ESVM_RANDOM_SUBSPACE_FEATURES; ++f)
                probeSampleTest[iRS][f] = probeSamples[p][rsmFeatureIndexes[rs][f]];
        }
    #endif/*ESVM_RANDOM_SUBSPACE_METHOD*/

    // testing, score fusion, normalization
    size_t dimsProbes[2]{ nESVM, nPositives };
    xstd::mvector<2, double> scores(dimsProbes, 0.0);
    xstd::mvector<1, double> classificationScores(nPositives, 0.0);
    for (size_t pos = 0; pos < nPositives; ++pos) {
        for (size_t svm = 0; svm < nESVM; ++svm) {
            scores[svm][pos] = EoESVM[svm][pos].predict(probeSampleTest[svm]);
            classificationScores[pos] += scores[svm][pos];  // score accumulation for fusion
        }        
        classificationScores[pos] /= (double)nESVM;         // average score fusion and normalization post-fusion
        #if ESVM_SCORE_NORMALIZATION_MODE == 1
        classificationScores[pos] = normalize(MIN_MAX, classificationScores[pos], scoreRefMin, scoreRefMax, ESVM_SCORE_NORMALIZATION_CLIP);
        #elif ESVM_SCORE_NORMALIZATION_MODE == 2
        classificationScores[pos] = normalize(Z_SCORE, classificationScores[pos], scoreRefMean, scoreRefStdDev, ESVM_SCORE_NORMALIZATION_CLIP);
        #endif/*ESVM_SCORE_NORMALIZATION_MODE*/
    }
    return classificationScores;
}