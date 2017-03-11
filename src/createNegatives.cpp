#include "createNegatives.h"

#include "boost/filesystem.hpp"
namespace bfs = boost::filesystem;
using namespace std;

void load_pgm_images_from_directory(std::string dir, xstd::mvector<2, cv::Mat>& imgVector){
    size_t nPatches = 9;
    cv::Size imageSize = cv::Size(48, 48);
    cv::Size patchCounts = cv::Size(3, 3);
    bool useHistEqual = false;
    bfs::directory_iterator endDir;

    if (bfs::is_directory(dir))
    {
        for (bfs::directory_iterator itDir(dir); itDir != endDir; ++itDir)
        {
            if (bfs::is_regular_file(*itDir) && itDir->path().extension() == ".png")
            {
                size_t neg = imgVector.size();
                imgVector.push_back(xstd::mvector<1, cv::Mat>(nPatches));
                std::vector<cv::Mat> patches = imPreprocess(itDir->path().string(), imageSize, patchCounts,
                                                            useHistEqual, "WINDOW_NAME", cv::IMREAD_GRAYSCALE);
                for (size_t p = 0; p < nPatches; p++)
                    imgVector[neg][p] = patches[p];
            }                  
        }
    }
}

int create_negatives(){
    size_t nPatches = 9;
    xstd::mvector<2, cv::Mat> matNegativeSamples;   
    bfs::directory_iterator endDir;                            // [negative][patch](Mat[x,y])
    std::vector<PORTAL_TYPE> types = { ENTER, LEAVE };
    cv::Size patchCounts = cv::Size(3, 3);
    cv::Size imageSize = cv::Size(48, 48);
    std::vector<std::string> negativeSamplesID;                                 // [negative](string)
    xstd::mvector<2, FeatureVector> fvNegativeSamples;                      // [patch][descriptor][negative](FeatureVector)

    // Hog init
    cv::Size patchSize = cv::Size(imageSize.width / patchCounts.width, imageSize.height / patchCounts.height);
    cv::Size blockSize = cv::Size(2, 2);
    cv::Size blockStride = cv::Size(2, 2);
    cv::Size cellSize = cv::Size(2, 2);
    int nBins = 3;
    FeatureExtractorHOG hog;
    hog.initialize(patchSize, blockSize, blockStride, cellSize, nBins);

    // ChokePoint Path
    std::string rootChokePointPath = std::string(std::getenv("CHOKEPOINT_ROOT")) + "/";       // ChokePoint dataset root
    std::string roiChokePointCroppedFacePath = rootChokePointPath + "cropped_faces/";   // Path of extracted 96x96 ROI from all videos 

    std::vector<std::string> negativesID = { "0001", "0002", "0007", "0009", "0011",
                                             "0013", "0014", "0016", "0017", "0018",
                                             "0019", "0020", "0021", "0022", "0025" };
    bool useHistEqual = true;

    for (int sn = 1; sn <= SESSION_QUANTITY; sn++)
    {
        for (int pn = 1; pn <= PORTAL_QUANTITY; pn++) {
        for (auto it = types.begin(); it != types.end(); ++it) {
        for (int cn = 1; cn <= CAMERA_QUANTITY; cn++)
        {     

            string seq = buildChokePointSequenceString(pn, *it, sn, cn);


            // Add ROI to corresponding sample vectors according to individual IDs            
            for (int id = 1; id <= INDIVIDUAL_QUANTITY; id++)
            {
                std::string dirPath = roiChokePointCroppedFacePath + buildChokePointSequenceString(pn, *it, sn, cn, id) + "/";
                cout << "Loading negative and probe images for sequence " << dirPath << "...: " << std::endl;
                if (bfs::is_directory(dirPath))
                {
                    for (bfs::directory_iterator itDir(dirPath); itDir != endDir; ++itDir)
                    {
                        if (bfs::is_regular_file(*itDir) && itDir->path().extension() == ".pgm")
                        {
                            std::string strID = buildChokePointIndividualID(id);
                            if (contains(negativesID, strID))
                            {
                                size_t neg = matNegativeSamples.size();
                                matNegativeSamples.push_back(xstd::mvector<1, cv::Mat>(nPatches));
                                std::vector<cv::Mat> patches = imPreprocess(itDir->path().string(), imageSize, patchCounts,
                                                                            useHistEqual, "WINDOW_NAME", cv::IMREAD_GRAYSCALE);
                                for (size_t p = 0; p < nPatches; p++)
                                    matNegativeSamples[neg][p] = patches[p];

                                negativeSamplesID.push_back(strID);
                            }
                        }                  
                    }
                }                        
            }


        } } }
    }

    size_t dimsNegatives[2] = { nPatches, 0 };              // [patch][negative]
    fvNegativeSamples = xstd::mvector<2, FeatureVector>(dimsNegatives);
    size_t nNegatives = matNegativeSamples.size();

    for (size_t p = 0; p < nPatches; p++)
        for (size_t neg = 0; neg < nNegatives; neg++)
            fvNegativeSamples[p].push_back(hog.compute(matNegativeSamples[neg][p]));


    for (size_t p = 0; p < nPatches; p++)
    {    
        double minAll, maxAll;
        findMinMaxOverall(fvNegativeSamples[p], &minAll, &maxAll);
        std::cout << "Patch Number: " << p << " Min: " << minAll << " Max: " << maxAll << std::endl;

        for (size_t neg = 0; neg < nNegatives; neg++)
            fvNegativeSamples[p][neg] = normalizeMinMaxAllFeatures(fvNegativeSamples[p][neg], minAll, maxAll);
    }

    // ofstream outputFile;
    // outputFile.open ("example1.txt");

    // cout << "nNegatives: " << nNegatives << " nPatches: " << nPatches << endl;
    // for (size_t p = 0; p < nPatches; p++)
    //     for (size_t neg = 0; neg < nNegatives; neg++){
    //         for (size_t i = 0; i < fvNegativeSamples[p][neg].size(); i++){
    //             outputFile << fvNegativeSamples[p][neg][i];
    //         }
    //         outputFile << endl;
    //     }

    // outputFile.close();

    std::cout << "DONE!" << std::endl;
}

int create_probes(std::string positives, std::string negatives){
    double hogHardcodedFoundMin = 0;            // Min found using 'FullChokePoint' test with SAMAN pre-generated files
    double hogHardcodedFoundMax = 0.675058;     // Max found using 'FullChokePoint' test with SAMAN pre-generated files
    size_t nPatches = 9;
    xstd::mvector<2, cv::Mat> matPositiveSamples, matNegativeSamples;
    std::vector<PORTAL_TYPE> types = { ENTER, LEAVE };
    cv::Size patchCounts = cv::Size(3, 3);
    cv::Size imageSize = cv::Size(48, 48);
    xstd::mvector<2, FeatureVector> fvNegativeSamples, fvPositiveSamples;                   // [patch][descriptor][negative](FeatureVector)

    // Hog init
    cv::Size patchSize = cv::Size(imageSize.width / patchCounts.width, imageSize.height / patchCounts.height);
    cv::Size blockSize = cv::Size(2, 2);
    cv::Size blockStride = cv::Size(2, 2);
    cv::Size cellSize = cv::Size(2, 2);
    int nBins = 3;
    FeatureExtractorHOG hog;
    hog.initialize(patchSize, blockSize, blockStride, cellSize, nBins);

    // ChokePoint Path
    std::string rootChokePointPath = std::string(std::getenv("CHOKEPOINT_ROOT")) + "/";         // ChokePoint dataset root
    std::string roiChokePointCroppedFacePath = rootChokePointPath + "cropped_faces/";           // Path of extracted 96x96 ROI from all videos 

    // Add ROI to corresponding sample vectors according to individual IDs
    cout << "Loading probe images for sequence " << positives << "...: " << std::endl;
    load_pgm_images_from_directory(positives, matPositiveSamples);
    cout << "Loading probe images for sequence " << negatives << "...: " << std::endl;
    load_pgm_images_from_directory(negatives, matNegativeSamples);

    size_t nPositives = matPositiveSamples.size();
    size_t nNegatives = matNegativeSamples.size();

    size_t dims[2] = { nPatches, 0 };                             // [patch][negative/positive]
    fvNegativeSamples = xstd::mvector<2, FeatureVector>(dims);
    fvPositiveSamples = xstd::mvector<2, FeatureVector>(dims);

    // Calculate Feature Vectors
    for (size_t p = 0; p < nPatches; p++){
        for (size_t pos = 0; pos < nPositives; pos++)
            fvPositiveSamples[p].push_back(hog.compute(matPositiveSamples[pos][p]));
        for (size_t neg = 0; neg < nNegatives; neg++)
            fvNegativeSamples[p].push_back(hog.compute(matNegativeSamples[neg][p]));
    }

    for (size_t p = 0; p < nPatches; p++){
        for (size_t pos = 0; pos < nPositives; pos++)
            fvPositiveSamples[p][pos] = normalizeMinMaxAllFeatures(fvPositiveSamples[p][pos], hogHardcodedFoundMin, hogHardcodedFoundMax);
        for (size_t neg = 0; neg < nNegatives; neg++)
            fvNegativeSamples[p][neg] = normalizeMinMaxAllFeatures(fvNegativeSamples[p][neg], hogHardcodedFoundMin, hogHardcodedFoundMax);
    }

    for (size_t p = 0; p < nPatches; p++)
        fvPositiveSamples[p].insert(fvPositiveSamples[p].end(), fvNegativeSamples[p].begin(), fvNegativeSamples[p].end());

    std::vector<int> targetOutputs(nPositives, 1);
    std::vector<int> targetOutputsNeg(nNegatives, -1);
    targetOutputs.insert(targetOutputs.end(), targetOutputsNeg.begin(), targetOutputsNeg.end());

    cout << "Size check - pos: " << targetOutputs.size() << " neg: " << targetOutputsNeg.size() << std::endl;

    ESVM esvm;

    for (size_t p = 0; p < nPatches; p++)
        esvm.writeSampleDataFile("ID0003-probes-hog-patch" + std::to_string(p) + ".bin", fvPositiveSamples[p], targetOutputs, BINARY);

    // ofstream outputFile;
    // outputFile.open ("example1.txt");

    // cout << "nNegatives: " << nNegatives << " nPatches: " << nPatches << endl;
    // for (size_t p = 0; p < nPatches; p++)
    //     for (size_t neg = 0; neg < nNegatives; neg++){
    //         for (size_t i = 0; i < fvNegativeSamples[p][neg].size(); i++){
    //             outputFile << fvNegativeSamples[p][neg][i];
    //         }
    //         outputFile << endl;
    //     }

    // outputFile.close();

    std::cout << "DONE!" << std::endl;
}
