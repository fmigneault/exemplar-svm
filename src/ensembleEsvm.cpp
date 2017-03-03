#include "ensembleEsvm.h"
#include "esvmOptions.h"
#include "generic.h"
#include "norm.h"
#include "imgUtils.h"

#include <fstream>
#include <sstream>

/*
    Initializes an ESVM
*/
EnsembleESVM::EnsembleESVM()
{ 
    imageSize = cv::Size(48, 48);
    patchCounts = cv::Size(3, 3); 
    blockSize = cv::Size(2, 2);
    blockStride  = cv::Size(2, 2);
    cellSize = cv::Size(2, 2);  
    nBins = 3;
    nPatches = patchCounts.area();
    probeSampleFeats = std::vector<FeatureVector>(nPatches);
    hog = FeatureExtractorHOG(imageSize, blockSize, blockStride, cellSize, nBins);
    for(int i = 0; i < patchCounts.area(); i++)
        ensembleEsvm.push_back(ESVM()); 

    std::string sampleFileExt = ".bin";
    FileFormat sampleFileFormat = BINARY;

    FeatureExtractorHOG hog(imageSize, blockSize, blockStride, cellSize, nBins);
    double hogHardcodedFoundMin = 0;            // Min found using 'FullChokePoint' test with SAMAN pre-generated files
    double hogHardcodedFoundMax = 0.675058;     // Max found using 'FullChokePoint' test with SAMAN pre-generated files

    // positive samples
    std::vector<std::string> positivesID = { "ID0003", "ID0005", "ID0006", "ID0010", "ID0024" };
    size_t nPositives = positivesID.size();    
    size_t dimsPositives[2]{ nPatches, nPositives };
    xstd::mvector<2, FeatureVector> positiveSamples(dimsPositives);     // [patch][positives](FeatureVector)

    // negative 1samples    
    size_t dimsNegatives[2]{ nPatches, 0 };                             // number of negatives unknown (loaded from file)
    xstd::mvector<2, FeatureVector> negativeSamples(dimsNegatives);

    // Exemplar-SVM
    ESVM FileLoaderESVM;
    xstd::mvector<2, ESVM> esvm(dimsPositives);                         // [patch][positive](ESVM)    

    // load positive target still images, extract features and normalize
    cout << "Loading positive image stills, extracting feature vectors and normalizing..." << std::endl;
    for (size_t pos = 0; pos < nPositives; pos++)
    {        
        std::vector<cv::Mat> patches = imPreprocess(refStillImagesPath + "roi" + positivesID[pos] + ".tif", imageSize, patchCounts);
        for (size_t p = 0; p < nPatches; p++){
            positiveSamples[p][pos] = normalizeMinMaxAllFeatures(hog.compute(patches[p]), hogHardcodedFoundMin, hogHardcodedFoundMax);        
        }
    }

    // load negative samples from pre-generated files for training (samples in files are pre-normalized)
    cout << "Loading negative samples from files..." << std::endl;
    for (size_t p = 0; p < nPatches; p++)
        FileLoaderESVM.readSampleDataFile(negativeSamplesDir + "negatives-hog-patch" + std::to_string(p) + 
                                          sampleFileExt, negativeSamples[p], sampleFileFormat);
 // training
    cout << "Training ESVM with positives and negatives..." << std::endl;
    for (size_t p = 0; p < nPatches; p++)
        for (size_t pos = 0; pos < nPositives; pos++)
            esvm[p][pos] = ESVM({ positiveSamples[p][pos] }, negativeSamples[p], positivesID[pos] + "-patch" + std::to_string(p));

}

/*
    Predicts the classification value for the specified feature vector sample using the trained ESVM model.
*/
std::vector<double> EnsembleESVM::predict(const cv::Mat roi) // this should be a feat vector
{
    xstd::mvector<1, ESVM> esvm(nPatches); 

    // load positive target still images, extract features and normalize
    cout << "Extracting feature vectors and normalizing..." << std::endl;
    std::cout  << "HELLO1" << std::endl;
    std::vector<cv::Mat> patches = imPreprocess(roi, imageSize, patchCounts);
    std::cout  << "HELLO2" << std::endl;
    for (size_t p = 0; p < nPatches; p++)
        probeSampleFeats[p] = normalizeMinMaxAllFeatures(hog.compute(patches[p]), hogHardcodedFoundMin, hogHardcodedFoundMax);        
    std::cout  << "HELLO3" << std::endl;

    // testing, score fusion, normalization
    cout << "Testing probe samples against enrolled targets..." << std::endl;
    classificationScores = xstd::mvector<1, double>(1, 0.0);
    scores = xstd::mvector<1, double>(9, 0.0);
    for (size_t p = 0; p < nPatches; p++)
    {                
        scores.push_back(esvm[p].predict(probeSampleFeats[p]) );
        classificationScores[0] += scores[p];                          // score accumulation
    }
    classificationScores = normalizeMinMaxClassScores(classificationScores);      // score normalization post-fusion

    FeatureVector feats;
    std::vector<double> predictions;
    for(int i = 0; i < 9; i++)
        predictions.push_back(patchCounts.area()); // ensembleEsvm[i].predict(feats)

    return predictions;
}
