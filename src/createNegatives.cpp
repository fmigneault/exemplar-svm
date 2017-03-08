#include "createNegatives.h"

#include "boost/filesystem.hpp"
namespace bfs = boost::filesystem;
using namespace std;

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

        for (size_t neg = 0; neg < nNegatives; neg++)
            fvNegativeSamples[p][neg] = normalizeMinMaxAllFeatures(fvNegativeSamples[p][neg], minAll, maxAll);
    }

    ofstream outputFile;
    outputFile.open ("example.txt");

    cout << "nNegatives: " << nNegatives << " nPatches: " << nPatches << endl;
    for (size_t p = 0; p < nPatches; p++)
        for (size_t neg = 0; neg < nNegatives; neg++){
            for (size_t i = 0; i < fvNegativeSamples[p][neg].size(); i++){
                outputFile << fvNegativeSamples[p][neg][i];
            }
            outputFile << endl;
        }

    outputFile.close();

    std::cout << "DONE!" << std::endl;
}

