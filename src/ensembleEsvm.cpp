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
EnsembleESVM::EnsembleESVM(std::vector<cv::Mat> positiveRois, std::string negativesDir)
{ 
    setContants();

    probeSampleFeats = std::vector<FeatureVector>(nPatches);

    hog = FeatureExtractorHOG(imageSize, blockSize, blockStride, cellSize, nBins);

    // positive samples
    nPositives = positiveRois.size();    
    size_t dimsPositives[2]{ nPatches, nPositives };
    xstd::mvector<2, FeatureVector> positiveSamples(dimsPositives);     // [patch][positives](FeatureVector)

    // negative samples    
    size_t dimsNegatives[2]{ nPatches, 0 };                             // number of negatives unknown (loaded from file)
    xstd::mvector<2, FeatureVector> negativeSamples(dimsNegatives);

    // Exemplar-SVM
    ESVM FileLoaderESVM;
    ensembleEsvm = xstd::mvector<2, ESVM>(dimsPositives);                         // [patch][positive](ESVM)    

    // load positive target still images, extract features and normalize
    std::cout << "Loading positive image stills, extracting feature vectors and normalizing..." << std::endl;
    for (size_t pos = 0; pos < nPositives; pos++)
    {        
        std::vector<cv::Mat> patches = imPreprocess(positiveRois[pos], imageSize, patchCounts);
        for (size_t p = 0; p < nPatches; p++)
            positiveSamples[p][pos] = normalizeMinMaxAllFeatures(hog.compute(patches[p]), hogHardcodedFoundMin, hogHardcodedFoundMax);        
    }

    // load negative samples from pre-generated files for training (samples in files are pre-normalized)
    std::cout << "Loading negative samples from files..." << std::endl;
    for (size_t p = 0; p < nPatches; p++)
        FileLoaderESVM.readSampleDataFile(negativesDir + "negatives-hog-patch" + std::to_string(p) +
                                          sampleFileExt, negativeSamples[p], sampleFileFormat);
    // training
    std::cout << "Training ESVM with positives and negatives..." << std::endl;
    for (size_t p = 0; p < nPatches; p++)
        for (size_t pos = 0; pos < nPositives; pos++)
            ensembleEsvm[p][pos] = ESVM({ positiveSamples[p][pos] }, negativeSamples[p], std::to_string(pos) + "-patch" + std::to_string(p));

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

    hogHardcodedFoundMin = 0;            // Min found using 'FullChokePoint' test with SAMAN pre-generated files
    hogHardcodedFoundMax = 0.675058;     // Max found using 'FullChokePoint' test with SAMAN pre-generated files

    sampleFileExt = ".bin";
    sampleFileFormat = BINARY;
}


/*
    Predicts the classification value for the specified roi using the trained ESVM model.
*/
std::vector<double> EnsembleESVM::predict(const cv::Mat roi) // this should be a feat vector
{
    // load probe still images, extract features and normalize
    std::cout << "Loading probe images, extracting feature vectors and normalizing..." << std::endl;
    std::vector<cv::Mat> patches = imPreprocess(roi, imageSize, patchCounts);
    for (size_t p = 0; p < nPatches; p++)
        probeSampleFeats[p] = normalizeMinMaxAllFeatures(hog.compute(patches[p]), hogHardcodedFoundMin, hogHardcodedFoundMax);        

    // testing, score fusion, normalization
    cout << "Testing probe samples against enrolled targets..." << std::endl;
    classificationScores = xstd::mvector<1, double>(nPositives, 0.0);
    size_t dimsProbes[2]{ nPatches, nPositives }; 
    scores = xstd::mvector<2, double>(dimsProbes, 0.0);
    for (size_t pos = 0; pos < nPositives; pos++) 
    {
        for (size_t p = 0; p < nPatches; p++)
        {                
            scores[p][pos] = ensembleEsvm[p][pos].predict(probeSampleFeats[p]);
            classificationScores[pos] += scores[p][pos];                          // score accumulation
        }
        // std::cout << "ESVM Vals: pos:" << pos << " val: " << classificationScores[pos] << std::endl;
        classificationScores[pos] /= (double)nPatches;                                 // average score fusion
        // std::cout << "ESVM Vals: pos:" << pos << " val: " << classificationScores[pos] << std::endl;
        // classificationScores[pos] = normalizeMinMax(classificationScores[pos], hogHardcodedFoundMin, hogHardcodedFoundMax);      // score normalization post-fusion
        // std::cout << "ESVM Vals: pos:" << pos << " val: " << classificationScores[pos] << std::endl;
    }

    return classificationScores;
}
