#include "ensembleEsvm.h"
#include "esvmOptions.h"
#include "generic.h"
#include "norm.h"
#include "imgUtils.h"

#include <fstream>
#include <sstream>

/*
    Initializes an ESVM Ensemble
*/
EnsembleESVM::EnsembleESVM(std::vector<cv::Mat> positiveROIs, std::string negativesDir, std::vector<std::string> positiveIDs)
{ 
    setContants();
    logstream logger("./output.txt");

    probeSampleFeats = std::vector<FeatureVector>(nPatches);
    hog = FeatureExtractorHOG(imageSize, blockSize, blockStride, cellSize, nBins);

    if (positiveIDs.size() != positiveROIs.size())
    {
        positiveIDs = std::vector<std::string>(positiveIDs.size());
        for (size_t pos = 0; pos < nPositives; pos++)
            positiveIDs[pos] = std::to_string(pos);
    }

    // positive samples
    nPositives = positiveROIs.size();
    size_t dimsPositives[3]{ nPatches, nPositives, 0 };
    size_t dimsEsvm[2]{ nPatches, nPositives };
    xstd::mvector<3, FeatureVector> positiveSamples(dimsPositives);     // [patch][positives][representation](FeatureVector)

    // negative samples    
    size_t dimsNegatives[2]{ nPatches, 0 };                             // number of negatives unknown (loaded from file)
    xstd::mvector<2, FeatureVector> negativeSamples(dimsNegatives);

    // Exemplar-SVM
    ESVM FileLoaderESVM;
    ensembleEsvm = xstd::mvector<2, ESVM>(dimsEsvm);               // [patch][positive](ESVM)    

    // load positive target still images, extract features and normalize
    logger << "Loading positive image stills, extracting feature vectors and normalizing..." << std::endl;
    for (size_t pos = 0; pos < nPositives; pos++)
    {        
        std::vector<cv::Mat> processedImages = imSyntheticGenerationScaleAndTranslation(positiveROIs[pos], 4, 4, 0.6);
        int nRepresentations = processedImages.size();
        for (size_t rep = 0; rep < nRepresentations; rep++)
        {
            std::vector<cv::Mat> patches = imPreprocess(processedImages[rep], imageSize, patchCounts);
            for (size_t p = 0; p < nPatches; p++)
                positiveSamples[p][pos].push_back(normalizeMinMaxAllFeatures(hog.compute(patches[p]), hogHardcodedFoundMin, hogHardcodedFoundMax));        
        }
    }

    // load negative samples from pre-generated files for training (samples in files are pre-normalized)
    logger << "Loading negative samples from files..." << std::endl;
    for (size_t p = 0; p < nPatches; p++)
        FileLoaderESVM.readSampleDataFile(negativesDir + "negatives-hog-patch" + std::to_string(p) +
                                          sampleFileExt, negativeSamples[p], sampleFileFormat);
    // training
    logger << "Training ESVM with positives and negatives..." << std::endl;
    for (size_t p = 0; p < nPatches; p++){
        for (size_t pos = 0; pos < nPositives; pos++){
            // xstd::mvector<2, FeatureVector> tempNegativeSamples(negativeSamples);
            // std::cout << "Size before: " << tempNegativeSamples[p].size() << std::endl;
            // for (size_t posInt = 0; posInt < nPositives; posInt++)
            //     if(pos != posInt)
            //         tempNegativeSamples[p].push_back(positiveSamples[p][posInt]);
            
            // std::cout << "Size after: " << tempNegativeSamples[p].size() << std::endl;
            ensembleEsvm[p][pos] = ESVM(positiveSamples[p][pos], negativeSamples[p], positiveIDs[pos] + "-patch" + std::to_string(p));
        }
    }

}

void EnsembleESVM::setContants()
{
    imageSize = cv::Size(48, 48);
    patchCounts = cv::Size(3, 3); 
    blockSize = cv::Size(2, 2);
    blockStride  = cv::Size(2, 2);
    cellSize = cv::Size(2, 2);  
    nBins = 3;
    nPatches = patchCounts.area();
    min = 999;
    max = -999;

    patchThreshold = std::vector<double>{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

    hogHardcodedFoundMin = 0;               // Min found using 'FullChokePoint' test with SAMAN pre-generated files
    hogHardcodedFoundMax = 0.675058;        // Max found using 'FullChokePoint' test with SAMAN pre-generated files

    // scoreHardcodedFoundMin = -1.578030;     // Min found using 'SimplifiedWorkingProcedure' test with SAMAN pre-generated files
    // scoreHardcodedFoundMax = -0.478968;     // Max found using 'SimplifiedWorkingProcedure' test with SAMAN pre-generated files
    ///scoreHardcodedFoundMin = -0.638025;     // Min found using FAST-DT live test 
    // scoreHardcodedFoundMin = 0.085;         // Testing
    // scoreHardcodedFoundMax =  0.513050;     // Max found using FAST-DT live test 
    // scoreHardcodedFoundMin = -2.578030;       
    // scoreHardcodedFoundMax =  1.0;    MIN: -0.205498 MAX: 0.516354
    scoreHardcodedFoundMin = -0.205498;     // Min found using 'SimplifiedWorkingProcedure' test with SAMAN pre-generated files
    scoreHardcodedFoundMax = 0.516354;     // Max found using 'SimplifiedWorkingProcedure' test with SAMAN pre-generated files
   
    sampleFileExt = ".bin";
    sampleFileFormat = BINARY;
}

/*
    Predicts the classification value for the specified roi using the trained ESVM model.
*/
std::vector<double> EnsembleESVM::predict(const cv::Mat roi) // this should be a feat vector
{
    logstream logger("./output.txt");
    
    // load probe still images, extract features and normalize
    logger << "Loading probe images, extracting feature vectors and normalizing..." << std::endl;
    std::vector<cv::Mat> patches = imPreprocess(roi, imageSize, patchCounts);
    for (size_t p = 0; p < nPatches; p++)
        probeSampleFeats[p] = normalizeMinMaxAllFeatures(hog.compute(patches[p]), hogHardcodedFoundMin, hogHardcodedFoundMax);        

    // testing, score fusion, normalization
    logger << "Testing probe samples against enrolled targets..." << std::endl;
    classificationScores = xstd::mvector<1, double>(nPositives, 0.0);
    size_t dimsProbes[2]{ nPatches, nPositives }; 
    scores = xstd::mvector<2, double>(dimsProbes, 0.0);
    for (size_t pos = 0; pos < nPositives; pos++) 
    {
        for (size_t p = 0; p < nPatches; p++)
        {                
            scores[p][pos] = ensembleEsvm[p][pos].predict(probeSampleFeats[p]);
            if(scores[p][pos] > patchThreshold[p]){
                // classificationScores[pos] += 1 ; 
            }
            else{
                // classificationScores[pos] += 0 ; 
            }

            classificationScores[pos] += scores[p][pos];                          // score accumulation for fusion
        }
        // logger << "Positive patches, pos: " << pos << " " << scores[0][pos] << " " << scores[1][pos] << " " << scores[2][pos] << " " << scores[3][pos] << " " << scores[4][pos] << " " << scores[5][pos] << " " << scores[6][pos] << " " << scores[7][pos] << " " << scores[8][pos] << endl;
        // average score fusion and normalization post-fusion
        classificationScores[pos] /= (double)nPatches; 

        if(classificationScores[pos] > max) {
            max = classificationScores[pos];
        } 
        if(classificationScores[pos] < min) {
            min = classificationScores[pos];
        }

        logger << "MIN: " << min << " MAX: " << max << std::endl;

        classificationScores[pos] = normalizeMinMax(classificationScores[pos], scoreHardcodedFoundMin, scoreHardcodedFoundMax);
    }

    return classificationScores;
}
