#include "createSampleFiles.h"

#include "boost/filesystem.hpp"
namespace bfs = boost::filesystem;
using namespace std;

void load_pgm_images_from_directory(std::string dir, xstd::mvector<2, cv::Mat>& imgVector){
    size_t nPatches = 9;
    cv::Size imageSize = cv::Size(48, 48);
    cv::Size patchCounts = cv::Size(3, 3);
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
                                                            ESVM_USE_HISTOGRAM_EQUALIZATION, "WINDOW_NAME", cv::IMREAD_GRAYSCALE);
                for (size_t p = 0; p < nPatches; p++)
                    imgVector[neg][p] = patches[p];
            }                  
        }
    }
}

int create_negatives()
{
    #if PROC_ESVM_GENERATE_SAMPLE_FILES

    logstream logger("negatives-output.txt");
    std::string tab = "    ";

    // outputs
    bool writeBinaryFormat = true;
    bool writeLibsvmFormat = false;
    std::string windowNameOriginal = "WINDOW_ORIGINAL";     // display oringal 'cropped_face' ROIs
    std::string windowNameROI = "WINDOW_ROI";               // display localized ROIs from LBP improved (if activated)
    int delayShowROI = 1;                                   // [ms] - show found LBP improved ROIs with a delay (for visual inspection)
    bool keepAllFoundROI = false;                           // keep all the found ROIs (if applicable), or only the first one (if applicable)
    assert(writeBinaryFormat || writeLibsvmFormat);
    assert(delayShowROI > 0);

    // general parameters    
    size_t nPatches = 9;
    cv::Size patchCounts = cv::Size(3, 3);
    cv::Size imageSize = cv::Size(48, 48);

    // improved LBP face detection parameters (try to focus roi on more descriptive part of the face)
    #if PROC_ESVM_GENERATE_SAMPLE_FILES_MODE == 1
        double scaleFactor = 1.1;
        int nmsThreshold = 1;                           // 0 generates multiple detections, >0 usually returns only 1 face on 'cropped_faces' ROIs
        cv::Size minSize(20, 20), maxSize = imageSize;
        cv::CascadeClassifier faceCascade;
        std::string faceCascadeFilePath = sourcesOpenCV + "data/lbpcascades/lbpcascade_frontalface_improved.xml";
        assert(bfs::is_regular_file(faceCascadeFilePath));
        assert(faceCascade.load(faceCascadeFilePath));
    #endif/*PROC_ESVM_GENERATE_SAMPLE_FILES_MODE*/

    // feature extraction HOG parameters
    cv::Size patchSize = cv::Size(imageSize.width / patchCounts.width, imageSize.height / patchCounts.height);
    cv::Size blockSize = cv::Size(2, 2);
    cv::Size blockStride = cv::Size(2, 2);
    cv::Size cellSize = cv::Size(2, 2);
    int nBins = 3;

    // Negatives to employ
    std::vector<std::string> negativesID = { "0001", "0002", "0007", "0009", "0011",
                                             "0013", "0014", "0016", "0017", "0018",
                                             "0019", "0020", "0021", "0022", "0025" };

    // init containers / classes with parameters
    xstd::mvector<2, cv::Mat> matNegativeSamples;               // [negative][patch](Mat[x,y])
    std::vector<std::string> negativeSamplesID;                 // [negative](string)
    FeatureExtractorHOG hog;
    hog.initialize(patchSize, blockSize, blockStride, cellSize, nBins);

    // Loop for all ChokePoint cropped faces
    std::vector<int> perSessionNegatives(SESSION_QUANTITY, 0);
    std::vector<int> perSequenceNegatives(PORTAL_QUANTITY * SESSION_QUANTITY * PORTAL_TYPE_QUANTITY * CAMERA_QUANTITY, 0);
    int seqIdx = 0;
    std::vector<PORTAL_TYPE> types = { ENTER, LEAVE };
    bfs::directory_iterator endDir;
    for (int sn = 1; sn <= SESSION_QUANTITY; sn++) {
    for (int pn = 1; pn <= PORTAL_QUANTITY; pn++) {
    for (auto it = types.begin(); it != types.end(); ++it) {
    for (int cn = 1; cn <= CAMERA_QUANTITY; cn++)
    {     
        string seq = buildChokePointSequenceString(pn, *it, sn, cn);

        // Add ROI to corresponding sample vectors according to individual IDs            
        for (int id = 1; id <= INDIVIDUAL_QUANTITY; id++)
        {
            std::string dirPath = roiChokePointCroppedFacePath + buildChokePointSequenceString(pn, *it, sn, cn, id) + "/";
            logger << "Loading negative from directory: '" << dirPath << "'" << std::endl;
            if (bfs::is_directory(dirPath))
            {
                for (bfs::directory_iterator itDir(dirPath); itDir != endDir; ++itDir)
                {
                    if (bfs::is_regular_file(*itDir) && itDir->path().extension() == ".pgm")
                    {
                        std::string strID = buildChokePointIndividualID(id);
                        if (contains(negativesID, strID))
                        {
                            std::string imgPath = itDir->path().string();
                            cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
                            cv::Mat roi;
                            std::vector<cv::Rect> faces;

                            #if PROC_ESVM_GENERATE_SAMPLE_FILES_MODE == 0       // directly use 'cropped_faces'
                                roi = img;
                            #elif PROC_ESVM_GENERATE_SAMPLE_FILES_MODE == 1     // LBP improved localized ROI refinement                                
                                cv::imshow(windowNameOriginal, img);
                                cv::waitKey(1);
                                faceCascade.detectMultiScale(img, faces, scaleFactor, nmsThreshold, cv::CASCADE_SCALE_IMAGE, minSize, maxSize);
                                size_t nFaces = faces.size();
                                if (nFaces > 0)
                                {
                                    logger << "Found " << faces.size() << " face(s)" << std::endl;
                                    for (size_t iFace = 0; iFace < nFaces; ++iFace) {
                                        if (keepAllFoundROI || iFace == 0)              // update kept ROI according to setting
                                            roi = img(faces[iFace]);
                                        logger << tab << "face[" << iFace << "] = " << faces[iFace] << std::endl;
                                        cv::imshow(windowNameROI, img(faces[iFace]));   // always display all found ROIs
                                        cv::waitKey(delayShowROI);
                                    }
                                }
                                else
                                {
                                    logger << "Did not find face on cropped image: '" << imgPath << "'" << std::endl;
                                    continue;   // skip if not found any face
                                }
                            #elif PROC_ESVM_GENERATE_SAMPLE_FILES_MODE == 2     // pre-cropping of ROI

                            #endif/*PROC_ESVM_GENERATE_SAMPLE_FILES_MODE*/

                            std::vector<cv::Mat> patches = imPreprocess(roi, imageSize, patchCounts,
                                                                        ESVM_USE_HISTOGRAM_EQUALIZATION, windowNameROI, cv::IMREAD_GRAYSCALE);
                            size_t neg = matNegativeSamples.size();
                            matNegativeSamples.push_back(xstd::mvector<1, cv::Mat>(nPatches));
                            for (size_t p = 0; p < nPatches; p++)
                                matNegativeSamples[neg][p] = patches[p];
                            negativeSamplesID.push_back(strID);
                            perSessionNegatives[sn-1]++;
                            perSequenceNegatives[seqIdx]++;
                        }
                    }
                }
            }
        }
        seqIdx++;

    } } } } // end ChokePoint loops

    cv::destroyAllWindows();

    size_t nNegatives = matNegativeSamples.size();
    size_t dimsNegatives[2] = { nPatches, nNegatives };             // [patch][negative]
    xstd::mvector<2, FeatureVector> fvNegRaw(dimsNegatives);        // [patch][negative](FeatureVector)

    // feature extraction HOG
    for (size_t p = 0; p < nPatches; p++)
        for (size_t neg = 0; neg < nNegatives; neg++)
            fvNegRaw[p][neg] = hog.compute(matNegativeSamples[neg][p]);

    // find + apply normalization values
    size_t hogFeatCount = hog.getFeatureCount();
    double minAllROIOverAll = DBL_MAX, maxAllROIOverAll = -DBL_MAX, meanAllROIOverAll = -DBL_MAX, stdDevAllROIOverAll = -DBL_MAX;
    FeatureVector minAllROIPerFeat(hogFeatCount, DBL_MAX), maxAllROIPerFeat(hogFeatCount, -DBL_MAX),
                  meanAllROIPerFeat(hogFeatCount, -DBL_MAX), stdDevAllROIPerFeat(hogFeatCount, -DBL_MAX);
    std::vector<double> minPatchOverAll(nPatches, DBL_MAX), maxPatchOverAll(nPatches, -DBL_MAX), 
                        meanPatchOverAll(nPatches, -DBL_MAX), stdDevPatchOverAll(nPatches, -DBL_MAX);
    std::vector<FeatureVector> minPatchPerFeat(nPatches, FeatureVector(hogFeatCount, DBL_MAX)), 
                               maxPatchPerFeat(nPatches, FeatureVector(hogFeatCount, -DBL_MAX)),
                               meanPatchPerFeat(nPatches, FeatureVector(hogFeatCount, -DBL_MAX)),
                               stdDevPatchPerFeat(nPatches, FeatureVector(hogFeatCount, -DBL_MAX));
    xstd::mvector<2, FeatureVector> fvNegMinMaxPatchOverAll(dimsNegatives), fvNegMinMaxROIOverAll(dimsNegatives),
                                    fvNegZScorePatchOverAll(dimsNegatives), fvNegZScoreROIOverAll(dimsNegatives),
                                    fvNegMinMaxPatchPerFeat(dimsNegatives), fvNegMinMaxROIPerFeat(dimsNegatives),
                                    fvNegZScorePatchPerFeat(dimsNegatives), fvNegZScoreROIPerFeat(dimsNegatives);

    for (size_t p = 0; p < nPatches; p++)
    {
        // find per patch normalization paramters
        findNormParamsOverAll(MIN_MAX, fvNegRaw[p], minPatchOverAll[p], maxPatchOverAll[p]);
        findNormParamsOverAll(Z_SCORE, fvNegRaw[p], meanPatchOverAll[p], stdDevPatchOverAll[p]);
        findNormParamsPerFeature(MIN_MAX, fvNegRaw[p], minPatchPerFeat[p], maxPatchPerFeat[p]);
        findNormParamsPerFeature(Z_SCORE, fvNegRaw[p], meanPatchPerFeat[p], stdDevPatchPerFeat[p]);

        logger << "Patch Number: " << p << std::endl;
        logger << tab << "OverAll: " << std::endl 
               << tab << tab << "Min: " << minPatchOverAll[p] << std::endl
               << tab << tab << "Max: " << maxPatchOverAll[p] << std::endl
               << tab << tab << "Mean: " << meanPatchOverAll[p] << std::endl
               << tab << tab << "StdDev: " << stdDevPatchOverAll[p] << std::endl;
        logger << tab << "PerFeat: " << std::endl 
               << tab << tab << "Min: " << minPatchPerFeat[p] << std::endl
               << tab << tab << "Max: " << maxPatchPerFeat[p] << std::endl
               << tab << tab << "Mean: " << meanPatchPerFeat[p] << std::endl
               << tab << tab << "StdDev: " << stdDevPatchPerFeat[p] << std::endl;

        // apply found normalization parameters
        for (size_t neg = 0; neg < nNegatives; neg++) {
            fvNegMinMaxPatchOverAll[p][neg] = normalizeOverAll(MIN_MAX, fvNegRaw[p][neg], minPatchOverAll[p], maxPatchOverAll[p], true);
            fvNegZScorePatchOverAll[p][neg] = normalizeOverAll(Z_SCORE, fvNegRaw[p][neg], meanPatchOverAll[p], stdDevPatchOverAll[p], true);
            fvNegMinMaxPatchPerFeat[p][neg] = normalizePerFeature(MIN_MAX, fvNegRaw[p][neg], minPatchPerFeat[p], maxPatchPerFeat[p], true);
            fvNegZScorePatchPerFeat[p][neg] = normalizePerFeature(Z_SCORE, fvNegRaw[p][neg], meanPatchPerFeat[p], stdDevPatchPerFeat[p], true);
        }

        // update across all patches min-max normalization parameters
        if (minAllROIOverAll > minPatchOverAll[p])
            minAllROIOverAll = minPatchOverAll[p];
        if (maxAllROIOverAll < maxPatchOverAll[p])
            maxAllROIOverAll = maxPatchOverAll[p];
        for (size_t f = 0; f < hogFeatCount; ++f) {
            if (minAllROIPerFeat[f] > minPatchPerFeat[p][f])
                minAllROIPerFeat[f] = minPatchPerFeat[p][f];
            if (maxAllROIPerFeat[f] < maxPatchPerFeat[p][f])
                maxAllROIPerFeat[f] = maxPatchPerFeat[p][f];
        }
    }

    // update across all patches z-score normalization parameters
    std::vector<FeatureVector> fvNegAllPatches;
    fvNegAllPatches.reserve(nPatches * nNegatives);
    for (size_t p = 0; p < nPatches; p++)
        fvNegAllPatches.insert(fvNegAllPatches.end(), fvNegRaw[p].begin(), fvNegRaw[p].end());
    findNormParamsOverAll(Z_SCORE, fvNegAllPatches, meanAllROIOverAll, stdDevAllROIOverAll);
    findNormParamsPerFeature(Z_SCORE, fvNegAllPatches, meanAllROIPerFeat, stdDevAllROIPerFeat);

    // apply found across all patches normalization parameters
    for (size_t p = 0; p < nPatches; p++) {
        for (size_t neg = 0; neg < nNegatives; neg++) {
            fvNegMinMaxROIOverAll[p][neg] = normalizeOverAll(MIN_MAX, fvNegRaw[p][neg], minAllROIOverAll, maxAllROIOverAll, true);
            fvNegZScoreROIOverAll[p][neg] = normalizeOverAll(Z_SCORE, fvNegRaw[p][neg], meanAllROIOverAll, stdDevAllROIOverAll, true);
            fvNegMinMaxROIPerFeat[p][neg] = normalizePerFeature(MIN_MAX, fvNegRaw[p][neg], minAllROIPerFeat, maxAllROIPerFeat, true);
            fvNegZScoreROIPerFeat[p][neg] = normalizePerFeature(Z_SCORE, fvNegRaw[p][neg], meanAllROIPerFeat, stdDevAllROIPerFeat, true);
        }
    }

    // write resulting sample files
    std::vector<int> negClass(nNegatives, ESVM_NEGATIVE_CLASS);
    for (size_t p = 0; p < nPatches; p++)
    {
        std::string fileStart = "negatives-patch" + std::to_string(p);
        if (writeBinaryFormat) {
            ESVM::writeSampleDataFile(fileStart + "-raw.bin",                       fvNegRaw[p],                negClass, BINARY);
            ESVM::writeSampleDataFile(fileStart + "-normPatch-minmax-overAll.bin",  fvNegMinMaxPatchOverAll[p], negClass, BINARY);
            ESVM::writeSampleDataFile(fileStart + "-normROI-minmax-overAll.bin",    fvNegMinMaxROIOverAll[p],   negClass, BINARY);
            ESVM::writeSampleDataFile(fileStart + "-normPatch-zcore-overAll.bin",   fvNegZScorePatchOverAll[p], negClass, BINARY);
            ESVM::writeSampleDataFile(fileStart + "-normROI-zcore-overAll.bin",     fvNegZScoreROIOverAll[p],   negClass, BINARY);
            ESVM::writeSampleDataFile(fileStart + "-normPatch-minmax-perFeat.bin",  fvNegMinMaxPatchPerFeat[p], negClass, BINARY);
            ESVM::writeSampleDataFile(fileStart + "-normROI-minmax-perFeat.bin",    fvNegMinMaxROIPerFeat[p],   negClass, BINARY);
            ESVM::writeSampleDataFile(fileStart + "-normPatch-zcore-perFeat.bin",   fvNegZScorePatchPerFeat[p], negClass, BINARY);
            ESVM::writeSampleDataFile(fileStart + "-normROI-zcore-perFeat.bin",     fvNegZScoreROIPerFeat[p],   negClass, BINARY);
        }
        if (writeLibsvmFormat) {
            ESVM::writeSampleDataFile(fileStart + "-raw.data",                      fvNegRaw[p],                negClass, LIBSVM);
            ESVM::writeSampleDataFile(fileStart + "-normPatch-minmax-overAll.data", fvNegMinMaxPatchOverAll[p], negClass, LIBSVM);
            ESVM::writeSampleDataFile(fileStart + "-normROI-minmax-overAll.data",   fvNegMinMaxROIOverAll[p],   negClass, LIBSVM);
            ESVM::writeSampleDataFile(fileStart + "-normPatch-zcore-overAll.data",  fvNegZScorePatchOverAll[p], negClass, LIBSVM);
            ESVM::writeSampleDataFile(fileStart + "-normROI-zcore-overAll.data",    fvNegZScoreROIOverAll[p],   negClass, LIBSVM);
            ESVM::writeSampleDataFile(fileStart + "-normPatch-minmax-perFeat.data", fvNegMinMaxPatchPerFeat[p], negClass, LIBSVM);
            ESVM::writeSampleDataFile(fileStart + "-normROI-minmax-perFeat.data",   fvNegMinMaxROIPerFeat[p],   negClass, LIBSVM);
            ESVM::writeSampleDataFile(fileStart + "-normPatch-zcore-perFeat.data",  fvNegZScorePatchPerFeat[p], negClass, LIBSVM);
            ESVM::writeSampleDataFile(fileStart + "-normROI-zcore-perFeat.data",    fvNegZScoreROIPerFeat[p],   negClass, LIBSVM);
        }
    }
    
    std::string str_minPatchPerFeat, str_maxPatchPerFeat, str_meanPatchPerFeat, str_stdDevPatchPerFeat;
    for (size_t p = 0; p < nPatches; ++p) {
        str_minPatchPerFeat += "\n" + tab + tab + featuresToVectorString(minPatchPerFeat[p]);
        str_maxPatchPerFeat += "\n" + tab + tab + featuresToVectorString(maxPatchPerFeat[p]);
        str_meanPatchPerFeat += "\n" + tab + tab + featuresToVectorString(meanPatchPerFeat[p]);
        str_stdDevPatchPerFeat += "\n" + tab + tab + featuresToVectorString(stdDevPatchPerFeat[p]);
    }

    // write configs employed (traceback)
    logstream logSampleConfig("negatives-output-config.txt");
    logSampleConfig << "negativeIDs:      " << negativesID << std::endl
                    << "nNegatives:       " << nNegatives << std::endl
                    << "perSessionNeg:    " << perSessionNegatives << std::endl
                    << "perSeqNeg         " << perSequenceNegatives << std::endl
                    << "histEqual:        " << ESVM_USE_HISTOGRAM_EQUALIZATION << std::endl
                    << "generation mode:  " << PROC_ESVM_GENERATE_SAMPLE_FILES_MODE << std::endl
                    #if PROC_ESVM_GENERATE_SAMPLE_FILES_MODE == 1       // using LBP improved localized ROI refinement
                    << "scaleFactor:      " << scaleFactor << std::endl
                    << "nmsThreshold:     " << nmsThreshold << std::endl
                    << "CC minSize:       " << minSize << std::endl
                    << "CC maxSize:       " << maxSize << std::endl
                    #elif PROC_ESVM_GENERATE_SAMPLE_FILES_MODE == 2     // using pre-cropped ROI refinement

                    #endif/*PROC_ESVM_GENERATE_SAMPLE_FILES_MODE*/
                    << "imageSize:        " << imageSize << std::endl
                    << "nPatches:         " << nPatches << std::endl
                    << "patchCounts:      " << patchCounts << std::endl
                    << "patchSize:        " << patchSize << std::endl
                    << "blockSize:        " << blockSize << std::endl
                    << "blockStride:      " << blockStride << std::endl
                    << "cellSize:         " << cellSize << std::endl
                    << "nBins:            " << nBins << std::endl 
                    << "nFeatures:        " << hogFeatCount << std::endl
                    << "BINARY fmt?:      " << writeBinaryFormat << std::endl
                    << "LIBSVM fmt?:      " << writeLibsvmFormat << std::endl
                    << "OverAll Norm:     " << std::endl
                    << tab << "minROI[p]:    " << minPatchOverAll << std::endl
                    << tab << "maxROI[p]:    " << maxPatchOverAll << std::endl
                    << tab << "meanROI[p]:   " << meanPatchOverAll << std::endl
                    << tab << "stdDevROI[p]: " << stdDevPatchOverAll << std::endl
                    << tab << "minAllROI:    " << minAllROIOverAll << std::endl
                    << tab << "maxAllROI:    " << maxAllROIOverAll << std::endl
                    << tab << "meanAllROI:   " << meanAllROIOverAll << std::endl
                    << tab << "stdDevAllROI: " << stdDevAllROIOverAll << std::endl
                    << "PerFeat Norm:     " << std::endl
                    << tab << "minROI[p]:    " << str_minPatchPerFeat << std::endl
                    << tab << "maxROI[p]:    " << str_maxPatchPerFeat << std::endl
                    << tab << "meanROI[p]:   " << str_meanPatchPerFeat << std::endl
                    << tab << "stdDevROI[p]: " << str_stdDevPatchPerFeat << std::endl
                    << tab << "minAllROI:    " << minAllROIPerFeat << std::endl
                    << tab << "maxAllROI:    " << maxAllROIPerFeat << std::endl
                    << tab << "meanAllROI:   " << meanAllROIPerFeat << std::endl
                    << tab << "stdDevAllROI: " << stdDevAllROIPerFeat << std::endl
                    << "all Neg IDs:      " << negativeSamplesID << std::endl;

    #else/*PROC_ESVM_GENERATE_SAMPLE_FILES*/
    return passThroughDisplayTestStatus(__func__, SKIPPED);
    #endif/*PROC_ESVM_GENERATE_SAMPLE_FILES*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

int create_probes(std::string positives, std::string negatives)
{
    logstream logger("probes-output.txt");

    double hogRefMin = 0;            // Min found using 'FullChokePoint' test with SAMAN pre-generated files
    double hogRefMax = 0.675058;     // Max found using 'FullChokePoint' test with SAMAN pre-generated files
    size_t nPatches = 9;
    xstd::mvector<2, cv::Mat> matPositiveSamples, matNegativeSamples;
    std::vector<PORTAL_TYPE> types = { ENTER, LEAVE };
    cv::Size patchCounts = cv::Size(3, 3);
    cv::Size imageSize = cv::Size(48, 48);
    xstd::mvector<2, FeatureVector> fvNeg, fvPositiveSamples;                   // [patch][descriptor][negative](FeatureVector)

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
    logger << "Loading probe images for sequence " << positives << "...: " << std::endl;
    load_pgm_images_from_directory(positives, matPositiveSamples);
    logger << "Loading probe images for sequence " << negatives << "...: " << std::endl;
    load_pgm_images_from_directory(negatives, matNegativeSamples);

    size_t nPositives = matPositiveSamples.size();
    size_t nNegatives = matNegativeSamples.size();

    size_t dims[2] = { nPatches, 0 };                             // [patch][negative/positive]
    fvNeg = xstd::mvector<2, FeatureVector>(dims);
    fvPositiveSamples = xstd::mvector<2, FeatureVector>(dims);

    // Calculate Feature Vectors
    for (size_t p = 0; p < nPatches; p++){
        for (size_t pos = 0; pos < nPositives; pos++)
            fvPositiveSamples[p].push_back(hog.compute(matPositiveSamples[pos][p]));
        for (size_t neg = 0; neg < nNegatives; neg++)
            fvNeg[p].push_back(hog.compute(matNegativeSamples[neg][p]));
    }

    for (size_t p = 0; p < nPatches; p++){
        for (size_t pos = 0; pos < nPositives; pos++)
            fvPositiveSamples[p][pos] = normalizeOverAll(MIN_MAX, fvPositiveSamples[p][pos], hogRefMin, hogRefMax);
        for (size_t neg = 0; neg < nNegatives; neg++)
            fvNeg[p][neg] = normalizeOverAll(MIN_MAX, fvNeg[p][neg], hogRefMin, hogRefMax);
    }

    for (size_t p = 0; p < nPatches; p++)
        fvPositiveSamples[p].insert(fvPositiveSamples[p].end(), fvNeg[p].begin(), fvNeg[p].end());

    std::vector<int> targetOutputs(nPositives, 1);
    std::vector<int> targetOutputsNeg(nNegatives, -1);
    targetOutputs.insert(targetOutputs.end(), targetOutputsNeg.begin(), targetOutputsNeg.end());

    logger << "Size check - pos: " << targetOutputs.size() << " neg: " << targetOutputsNeg.size() << std::endl;

    for (size_t p = 0; p < nPatches; p++)
        ESVM::writeSampleDataFile("ID0003-probes-hog-patch" + std::to_string(p) + ".bin", fvPositiveSamples[p], targetOutputs, BINARY);

    // ofstream outputFile;
    // outputFile.open ("example1.txt");

    // logger << "nNegatives: " << nNegatives << " nPatches: " << nPatches << endl;
    // for (size_t p = 0; p < nPatches; p++)
    //     for (size_t neg = 0; neg < nNegatives; neg++){
    //         for (size_t i = 0; i < fvNeg[p][neg].size(); i++){
    //             outputFile << fvNeg[p][neg][i];
    //         }
    //         outputFile << endl;
    //     }

    // outputFile.close();

    logger << "DONE!" << std::endl;
    return 0;
}
