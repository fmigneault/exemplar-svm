#include "createSampleFiles.h"

#include "boost/filesystem.hpp"
namespace bfs = boost::filesystem;
using namespace std;

xstd::mvector<2, cv::Mat> loadAndProcessImages(std::string dirPath, std::string imageExtension)
{
    size_t nPatches = 9;
    cv::Size imageSize = cv::Size(48, 48);
    cv::Size patchCounts = cv::Size(3, 3);
    bfs::directory_iterator endDir;
    xstd::mvector<2, cv::Mat> processedImagePatches;

    if (bfs::is_directory(dirPath))
    {
        for (bfs::directory_iterator itDir(dirPath); itDir != endDir; ++itDir)
        {
            if (bfs::is_regular_file(*itDir) && itDir->path().extension() == imageExtension)
            {
                size_t neg = processedImagePatches.size();
                processedImagePatches.push_back(xstd::mvector<1, cv::Mat>(nPatches));
                std::vector<cv::Mat> patches = imPreprocess(itDir->path().string(), imageSize, patchCounts,
                                                            ESVM_USE_HISTOGRAM_EQUALIZATION, "WINDOW_NAME", cv::IMREAD_GRAYSCALE);
                for (size_t p = 0; p < nPatches; ++p)
                    processedImagePatches[neg][p] = patches[p];
            }                  
        }
    }
    return processedImagePatches;
}

int proc_generateConvertedImageTypes()
{
    #if PROC_ESVM_GENERATE_CONVERTED_IMAGES

    std::string parentDir = "C:/Users/Francis/Programs/DEVELOPMENT/Face Recognition/Face Databases/ChokePoint Dataset/Cropped_Faces/P1E_S1_C1/";
    std::string outputDir = parentDir + "tif/";
    std::string imgExt_To = ".tif";
    std::string imgExt_From = ".pgm";
    imConvert(parentDir, imgExt_To, imgExt_From, outputDir);

    #else/*!PROC_ESVM_GENERATE_CONVERTED_IMAGES*/
    return passThroughDisplayTestStatus(__func__, SKIPPED);
    #endif/*PROC_ESVM_GENERATE_CONVERTED_IMAGES*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

std::vector<std::string> getReplicationNegativeIDs()
{
    return std::vector<std::string>
    #if !defined(PROC_ESVM_GENERATE_SAMPLE_FILES_REPLICATION) || PROC_ESVM_GENERATE_SAMPLE_FILES_REPLICATION == 0
    { "0001", "0002", "0007", "0009", "0011", "0013", "0014", "0016", "0017", "0018", "0019", "0020", "0021", "0022", "0025" };
    #elif PROC_ESVM_GENERATE_SAMPLE_FILES_REPLICATION == 1
    { "0001", "0002", "0003", "0004", "0005", "0006", "0007", "0012", "0014", "0016", "0018", "0019", "0023", "0024", "0027" };
    #elif PROC_ESVM_GENERATE_SAMPLE_FILES_REPLICATION == 2
    { "0001", "0004", "0007", "0009", "0010", "0011", "0012", "0014", "0016", "0017", "0021", "0024", "0026", "0027", "0030" };
    #elif PROC_ESVM_GENERATE_SAMPLE_FILES_REPLICATION == 3
    { "0001", "0002", "0006", "0007", "0009", "0010", "0013", "0014", "0016", "0020", "0021", "0022", "0023", "0026", "0027" };
    #elif PROC_ESVM_GENERATE_SAMPLE_FILES_REPLICATION == 4
    { "0002", "0003", "0005", "0006", "0007", "0009", "0010", "0011", "0013", "0017", "0021", "0022", "0025", "0026", "0027" };
    #elif PROC_ESVM_GENERATE_SAMPLE_FILES_REPLICATION == 5
    { "0001", "0002", "0005", "0006", "0010", "0011", "0016", "0018", "0019", "0021", "0023", "0025", "0027", "0028", "0029" };
    #elif PROC_ESVM_GENERATE_SAMPLE_FILES_REPLICATION == 6
    { "0002", "0007", "0009", "0010", "0011", "0012", "0013", "0014", "0017", "0022", "0024", "0027", "0028", "0029", "0030" };
    #elif PROC_ESVM_GENERATE_SAMPLE_FILES_REPLICATION == 7
    { "0001", "0002", "0004", "0005", "0007", "0010", "0011", "0017", "0019", "0020", "0022", "0023", "0024", "0028", "0030" };
    #elif PROC_ESVM_GENERATE_SAMPLE_FILES_REPLICATION == 8
    { "0001", "0003", "0004", "0006", "0007", "0010", "0011", "0012", "0017", "0021", "0023", "0026", "0027", "0029", "0030" };
    #elif PROC_ESVM_GENERATE_SAMPLE_FILES_REPLICATION == 9
    { "0001", "0002", "0007", "0009", "0010", "0011", "0015", "0016", "0017", "0018", "0020", "0022", "0023", "0025", "0030" };
    #elif PROC_ESVM_GENERATE_SAMPLE_FILES_REPLICATION == 10
    { "0001", "0003", "0006", "0015", "0016", "0017", "0018", "0019", "0022", "0023", "0024", "0026", "0028", "0029", "0030" };
    #endif/*PROC_ESVM_GENERATE_SAMPLE_FILES_REPLICATION*/
}

int proc_createNegativesSampleFiles()
{
    #if PROC_ESVM_GENERATE_SAMPLE_FILES
    logstream logger(LOGGER_FILE);
    logstream logNeg("negatives-output.txt");
    logger << "Running '" << __func__ << "' test..." << std::endl;    
    std::string tab = "    ";

    // outputs
    std::string windowNameOriginal = "WINDOW_ORIGINAL";     // display oringal 'cropped_face' ROIs
    std::string windowNameROI = "WINDOW_ROI";               // display localized ROIs from LBP improved (if activated)
    int delayShowROI = 1;                                   // [ms] - show found LBP improved ROIs with a delay (for visual inspection)
    bool keepAllFoundROI = false;                           // keep all the found ROIs (if applicable), or only the first one (if applicable)
    ASSERT_LOG(PROC_ESVM_GENERATE_SAMPLE_FILES_BINARY || PROC_ESVM_GENERATE_SAMPLE_FILES_LIBSVM, 
               "Either 'PROC_ESVM_GENERATE_SAMPLE_FILES_BINARY' or 'PROC_ESVM_GENERATE_SAMPLE_FILES_LIBSVM' must be enabled for file generation");
    ASSERT_LOG(delayShowROI > 0, "Delay to display ROI during file generation must be greater than zero");
    ASSERT_LOG(PROC_ESVM_GENERATE_SAMPLE_FILES_SESSION >= 0 && PROC_ESVM_GENERATE_SAMPLE_FILES_SESSION <= SESSION_QUANTITY,
               "Undefined value '" + std::to_string(PROC_ESVM_GENERATE_SAMPLE_FILES_SESSION) + 
               "' specified for 'PROC_ESVM_GENERATE_SAMPLE_FILES_SESSION', must be in range [0" + std::to_string(SESSION_QUANTITY) + "]");

    // general parameters    
    size_t nPatches = 9;
    cv::Size patchCounts = cv::Size(3, 3);
    cv::Size imageSize = cv::Size(48, 48);

    // improved LBP face detection parameters (try to focus roi on more descriptive part of the face)
    #if ESVM_ROI_PREPROCESS_MODE == 1
        double scaleFactor = 1.1;
        int nmsThreshold = 1;                           // 0 generates multiple detections, >0 usually returns only 1 face on 'cropped_faces' ROIs
        cv::Size minSize(20, 20), maxSize = imageSize;
        cv::CascadeClassifier faceCascade;
        std::string faceCascadeFilePath = sourcesOpenCV + "data/lbpcascades/lbpcascade_frontalface_improved.xml";
        assert(bfs::is_regular_file(faceCascadeFilePath));
        assert(faceCascade.load(faceCascadeFilePath));
    #endif/*ESVM_ROI_PREPROCESS_MODE*/

    // feature extraction HOG parameters
    cv::Size patchSize = cv::Size(imageSize.width / patchCounts.width, imageSize.height / patchCounts.height);
    cv::Size blockSize = cv::Size(2, 2);
    cv::Size blockStride = cv::Size(2, 2);
    cv::Size cellSize = cv::Size(2, 2);
    int nBins = 3;
    FeatureExtractorHOG hog(patchSize, blockSize, blockStride, cellSize, nBins);

    // Negatives to employ
    std::vector<std::string> negativesID = getReplicationNegativeIDs();
    logNeg << "Using negative IDs: " << negativesID << std::endl;

    // init containers / classes with parameters
    size_t nNegatives = 0;
    size_t dimsNegatives[2] = { nPatches, nNegatives };         // [patch][negative]
    xstd::mvector<2, FeatureVector> fvNegRaw(dimsNegatives);    // [patch][negative](FeatureVector)    
    std::vector<std::string> negativeSamplesID;                 // [negative](string)    

    // Loop for all ChokePoint cropped faces
    int totalSeq = PROC_ESVM_GENERATE_SAMPLE_FILES_SESSION == 0 ? TOTAL_SEQUENCES : TOTAL_SEQUENCES / SESSION_QUANTITY;
    std::vector<int> perSessionNegatives(SESSION_QUANTITY, 0);
    std::vector<int> perSequenceNegatives(totalSeq, 0);
    int seqIdx = 0;
    std::vector<PORTAL_TYPE> types = { ENTER, LEAVE };
    bfs::directory_iterator endDir;
    #if PROC_ESVM_GENERATE_SAMPLE_FILES_SESSION == 0
    for (int sn = 1; sn <= SESSION_QUANTITY; ++sn) {            // session number
    #else /*PROC_ESVM_GENERATE_SAMPLE_FILES_SESSION == [1-4]*/
    int sn = PROC_ESVM_GENERATE_SAMPLE_FILES_SESSION; {
    #endif/*PROC_ESVM_GENERATE_SAMPLE_FILES_SESSION*/
    for (int pn = 1; pn <= PORTAL_QUANTITY; ++pn) {             // portal number
    for (auto pt = types.begin(); pt != types.end(); ++pt) {    // portal type
    for (int cn = 1; cn <= CAMERA_QUANTITY; ++cn)               // camera number
    {     
        string seq = buildChokePointSequenceString(pn, *pt, sn, cn);

        // Add ROI to corresponding sample vectors according to individual IDs            
        for (int id = 1; id <= INDIVIDUAL_QUANTITY; id++)
        {
            std::string dirPath = roiChokePointCroppedFacePath + buildChokePointSequenceString(pn, *pt, sn, cn, id) + "/";
            logNeg << "Loading negative from directory: '" << dirPath << "'" << std::endl;
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

                            #if ESVM_ROI_PREPROCESS_MODE == 0           // directly use 'cropped_faces'
                                roi = img;
                            #elif ESVM_ROI_PREPROCESS_MODE == 1         // LBP improved localized ROI refinement                                
                                cv::imshow(windowNameOriginal, img);
                                cv::waitKey(1);
                                faceCascade.detectMultiScale(img, faces, scaleFactor, nmsThreshold, cv::CASCADE_SCALE_IMAGE, minSize, maxSize);
                                size_t nFaces = faces.size();
                                if (nFaces > 0)
                                {
                                    logNeg << "Found " << faces.size() << " face(s)" << std::endl;
                                    for (size_t iFace = 0; iFace < nFaces; ++iFace) {
                                        if (keepAllFoundROI || iFace == 0)              // update kept ROI according to setting
                                            roi = img(faces[iFace]);
                                        logNeg << tab << "face[" << iFace << "] = " << faces[iFace] << std::endl;
                                        cv::imshow(windowNameROI, img(faces[iFace]));   // always display all found ROIs
                                        cv::waitKey(delayShowROI);
                                    }
                                }
                                else
                                {
                                    logNeg << "Did not find face on cropped image: '" << imgPath << "'" << std::endl;
                                    continue;   // skip if not found any face
                                }
                            #elif ESVM_ROI_PREPROCESS_MODE == 2         // pre-cropping of ROI
                                cv::imshow(windowNameOriginal, img);
                                cv::waitKey(1);
                                roi = imCropByRatio(img, ESVM_ROI_CROP_RATIO);
                            #endif/*ESVM_ROI_PREPROCESS_MODE*/

                            // feature extraction HOG and update sample counts/ids
                            std::vector<cv::Mat> patches = imPreprocess(roi, imageSize, patchCounts, ESVM_USE_HISTOGRAM_EQUALIZATION,
                                                                        windowNameROI, cv::IMREAD_GRAYSCALE);                            
                            for (size_t p = 0; p < nPatches; ++p)
                                fvNegRaw[p].push_back(hog.compute(patches[p]));

                            negativeSamplesID.push_back(strID);
                            perSessionNegatives[sn-1]++;
                            perSequenceNegatives[seqIdx]++;
                            nNegatives++;        
        } } } } } // end get samples only from negative individual images
        seqIdx++;
    } } } } // end ChokePoint loops
    cv::destroyAllWindows();
    dimsNegatives[1] = nNegatives;  // update loaded negative count

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

    bool clip = ESVM_FEATURE_NORMALIZATION_CLIP;
    for (size_t p = 0; p < nPatches; ++p)
    {
        // find per patch normalization paramters
        findNormParamsOverAll(MIN_MAX, fvNegRaw[p], minPatchOverAll[p], maxPatchOverAll[p]);
        findNormParamsOverAll(Z_SCORE, fvNegRaw[p], meanPatchOverAll[p], stdDevPatchOverAll[p]);
        findNormParamsPerFeature(MIN_MAX, fvNegRaw[p], minPatchPerFeat[p], maxPatchPerFeat[p]);
        findNormParamsPerFeature(Z_SCORE, fvNegRaw[p], meanPatchPerFeat[p], stdDevPatchPerFeat[p]);

        logNeg << "Patch Number: " << p << std::endl;
        logNeg << tab << "OverAll: " << std::endl
               << tab << tab << "Min: " << minPatchOverAll[p] << std::endl
               << tab << tab << "Max: " << maxPatchOverAll[p] << std::endl
               << tab << tab << "Mean: " << meanPatchOverAll[p] << std::endl
               << tab << tab << "StdDev: " << stdDevPatchOverAll[p] << std::endl;
        logNeg << tab << "PerFeat: " << std::endl
               << tab << tab << "Min: " << minPatchPerFeat[p] << std::endl
               << tab << tab << "Max: " << maxPatchPerFeat[p] << std::endl
               << tab << tab << "Mean: " << meanPatchPerFeat[p] << std::endl
               << tab << tab << "StdDev: " << stdDevPatchPerFeat[p] << std::endl;

        // apply found normalization parameters
        for (size_t neg = 0; neg < nNegatives; ++neg) {
            fvNegMinMaxPatchOverAll[p][neg] = normalizeOverAll(MIN_MAX,    fvNegRaw[p][neg], minPatchOverAll[p],  maxPatchOverAll[p],    clip);
            fvNegZScorePatchOverAll[p][neg] = normalizeOverAll(Z_SCORE,    fvNegRaw[p][neg], meanPatchOverAll[p], stdDevPatchOverAll[p], clip);
            fvNegMinMaxPatchPerFeat[p][neg] = normalizePerFeature(MIN_MAX, fvNegRaw[p][neg], minPatchPerFeat[p],  maxPatchPerFeat[p],    clip);
            fvNegZScorePatchPerFeat[p][neg] = normalizePerFeature(Z_SCORE, fvNegRaw[p][neg], meanPatchPerFeat[p], stdDevPatchPerFeat[p], clip);
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
    for (size_t p = 0; p < nPatches; ++p)
        fvNegAllPatches.insert(fvNegAllPatches.end(), fvNegRaw[p].begin(), fvNegRaw[p].end());
    findNormParamsOverAll(Z_SCORE, fvNegAllPatches, meanAllROIOverAll, stdDevAllROIOverAll);
    findNormParamsPerFeature(Z_SCORE, fvNegAllPatches, meanAllROIPerFeat, stdDevAllROIPerFeat);

    // apply found across all patches normalization parameters
    for (size_t p = 0; p < nPatches; ++p) {
        for (size_t neg = 0; neg < nNegatives; ++neg) {
            fvNegMinMaxROIOverAll[p][neg] = normalizeOverAll(MIN_MAX,    fvNegRaw[p][neg], minAllROIOverAll,  maxAllROIOverAll,    clip);
            fvNegZScoreROIOverAll[p][neg] = normalizeOverAll(Z_SCORE,    fvNegRaw[p][neg], meanAllROIOverAll, stdDevAllROIOverAll, clip);
            fvNegMinMaxROIPerFeat[p][neg] = normalizePerFeature(MIN_MAX, fvNegRaw[p][neg], minAllROIPerFeat,  maxAllROIPerFeat,    clip);
            fvNegZScoreROIPerFeat[p][neg] = normalizePerFeature(Z_SCORE, fvNegRaw[p][neg], meanAllROIPerFeat, stdDevAllROIPerFeat, clip);
        }
    }

    // write resulting sample files
    std::vector<int> negClass(nNegatives, ESVM_NEGATIVE_CLASS);
    for (size_t p = 0; p < nPatches; ++p)
    {
        std::string fileStart = "negatives-patch" + std::to_string(p);
        #if PROC_ESVM_GENERATE_SAMPLE_FILES_BINARY
            ESVM::writeSampleDataFile(fileStart + "-raw.bin",                       fvNegRaw[p],                negClass, BINARY);
            ESVM::writeSampleDataFile(fileStart + "-normPatch-minmax-overAll.bin",  fvNegMinMaxPatchOverAll[p], negClass, BINARY);
            ESVM::writeSampleDataFile(fileStart + "-normROI-minmax-overAll.bin",    fvNegMinMaxROIOverAll[p],   negClass, BINARY);
            ESVM::writeSampleDataFile(fileStart + "-normPatch-zcore-overAll.bin",   fvNegZScorePatchOverAll[p], negClass, BINARY);
            ESVM::writeSampleDataFile(fileStart + "-normROI-zcore-overAll.bin",     fvNegZScoreROIOverAll[p],   negClass, BINARY);
            ESVM::writeSampleDataFile(fileStart + "-normPatch-minmax-perFeat.bin",  fvNegMinMaxPatchPerFeat[p], negClass, BINARY);
            ESVM::writeSampleDataFile(fileStart + "-normROI-minmax-perFeat.bin",    fvNegMinMaxROIPerFeat[p],   negClass, BINARY);
            ESVM::writeSampleDataFile(fileStart + "-normPatch-zcore-perFeat.bin",   fvNegZScorePatchPerFeat[p], negClass, BINARY);
            ESVM::writeSampleDataFile(fileStart + "-normROI-zcore-perFeat.bin",     fvNegZScoreROIPerFeat[p],   negClass, BINARY);
        #endif/*PROC_ESVM_GENERATE_SAMPLE_FILES_BINARY*/
        #if PROC_ESVM_GENERATE_SAMPLE_FILES_LIBSVM
            ESVM::writeSampleDataFile(fileStart + "-raw.data",                      fvNegRaw[p],                negClass, LIBSVM);
            ESVM::writeSampleDataFile(fileStart + "-normPatch-minmax-overAll.data", fvNegMinMaxPatchOverAll[p], negClass, LIBSVM);
            ESVM::writeSampleDataFile(fileStart + "-normROI-minmax-overAll.data",   fvNegMinMaxROIOverAll[p],   negClass, LIBSVM);
            ESVM::writeSampleDataFile(fileStart + "-normPatch-zcore-overAll.data",  fvNegZScorePatchOverAll[p], negClass, LIBSVM);
            ESVM::writeSampleDataFile(fileStart + "-normROI-zcore-overAll.data",    fvNegZScoreROIOverAll[p],   negClass, LIBSVM);
            ESVM::writeSampleDataFile(fileStart + "-normPatch-minmax-perFeat.data", fvNegMinMaxPatchPerFeat[p], negClass, LIBSVM);
            ESVM::writeSampleDataFile(fileStart + "-normROI-minmax-perFeat.data",   fvNegMinMaxROIPerFeat[p],   negClass, LIBSVM);
            ESVM::writeSampleDataFile(fileStart + "-normPatch-zcore-perFeat.data",  fvNegZScorePatchPerFeat[p], negClass, LIBSVM);
            ESVM::writeSampleDataFile(fileStart + "-normROI-zcore-perFeat.data",    fvNegZScoreROIPerFeat[p],   negClass, LIBSVM);
        #endif/*PROC_ESVM_GENERATE_SAMPLE_FILES_LIBSVM*/
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
                    << "feat norm clip:   " << ESVM_FEATURE_NORMALIZATION_CLIP << std::endl
                    << "generation mode:  " << ESVM_ROI_PREPROCESS_MODE << std::endl
                    #if ESVM_ROI_PREPROCESS_MODE == 1                   // using LBP improved localized ROI refinement
                    << "scaleFactor:      " << scaleFactor << std::endl
                    << "nmsThreshold:     " << nmsThreshold << std::endl
                    << "CC minSize:       " << minSize << std::endl
                    << "CC maxSize:       " << maxSize << std::endl
                    #elif ESVM_ROI_PREPROCESS_MODE == 2                 // using pre-cropped ROI refinement
                    << "pre-crop ratio:   " << ESVM_ROI_CROP_RATIO << std::endl
                    #endif/*ESVM_ROI_PREPROCESS_MODE*/
                    << "imageSize:        " << imageSize << std::endl
                    << "nPatches:         " << nPatches << std::endl
                    << "patchCounts:      " << patchCounts << std::endl
                    << "patchSize:        " << patchSize << std::endl
                    << "blockSize:        " << blockSize << std::endl
                    << "blockStride:      " << blockStride << std::endl
                    << "cellSize:         " << cellSize << std::endl
                    << "nBins:            " << nBins << std::endl 
                    << "nFeatures:        " << hogFeatCount << std::endl
                    << "BINARY fmt?:      " << PROC_ESVM_GENERATE_SAMPLE_FILES_BINARY << std::endl
                    << "LIBSVM fmt?:      " << PROC_ESVM_GENERATE_SAMPLE_FILES_LIBSVM << std::endl
                    << "SESSION mode:     " << PROC_ESVM_GENERATE_SAMPLE_FILES_SESSION << std::endl
                    << "REPLICATION:      " << PROC_ESVM_GENERATE_SAMPLE_FILES_REPLICATION << std::endl
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

    #else/*!PROC_ESVM_GENERATE_SAMPLE_FILES*/
    return passThroughDisplayTestStatus(__func__, SKIPPED);
    #endif/*PROC_ESVM_GENERATE_SAMPLE_FILES*/
    return passThroughDisplayTestStatus(__func__, PASSED);
}

int proc_createProbesSampleFiles(std::string positivesImageDirPath, std::string negativesImageDirPath)
{
    logstream logger(LOGGER_FILE);
    logstream logPrb("probes-output.txt");
    logger << "Running '" << __func__ << "' test..." << std::endl;

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
    logPrb << "Loading probe images from '" << positivesImageDirPath << "'...: " << std::endl;
    matPositiveSamples = loadAndProcessImages(positivesImageDirPath, ".png");
    logPrb << "Loading probe images from '" << negativesImageDirPath << "'...: " << std::endl;
    matNegativeSamples = loadAndProcessImages(negativesImageDirPath, ".png");

    size_t nPositives = matPositiveSamples.size();
    size_t nNegatives = matNegativeSamples.size();

    size_t dims[2] = { nPatches, 0 };                             // [patch][negative/positive]
    fvNeg = xstd::mvector<2, FeatureVector>(dims);
    fvPositiveSamples = xstd::mvector<2, FeatureVector>(dims);

    // Calculate Feature Vectors
    for (size_t p = 0; p < nPatches; ++p) {
        for (size_t pos = 0; pos < nPositives; ++pos)
            fvPositiveSamples[p].push_back(hog.compute(matPositiveSamples[pos][p]));
        for (size_t neg = 0; neg < nNegatives; ++neg)
            fvNeg[p].push_back(hog.compute(matNegativeSamples[neg][p]));
    }

    for (size_t p = 0; p < nPatches; ++p) {
        for (size_t pos = 0; pos < nPositives; ++pos)
            fvPositiveSamples[p][pos] = normalizeOverAll(MIN_MAX, fvPositiveSamples[p][pos], hogRefMin, hogRefMax);
        for (size_t neg = 0; neg < nNegatives; ++neg)
            fvNeg[p][neg] = normalizeOverAll(MIN_MAX, fvNeg[p][neg], hogRefMin, hogRefMax);
    }

    for (size_t p = 0; p < nPatches; ++p)
        fvPositiveSamples[p].insert(fvPositiveSamples[p].end(), fvNeg[p].begin(), fvNeg[p].end());

    std::vector<int> targetOutputs(nPositives, 1);
    std::vector<int> targetOutputsNeg(nNegatives, -1);
    targetOutputs.insert(targetOutputs.end(), targetOutputsNeg.begin(), targetOutputsNeg.end());

    logPrb << "Size check - pos: " << targetOutputs.size() << " neg: " << targetOutputsNeg.size() << std::endl;

    for (size_t p = 0; p < nPatches; ++p)
        ESVM::writeSampleDataFile("ID0003-probes-hog-patch" + std::to_string(p) + ".bin", fvPositiveSamples[p], targetOutputs, BINARY);

    // ofstream outputFile;
    // outputFile.open ("example1.txt");

    // logPrb << "nNegatives: " << nNegatives << " nPatches: " << nPatches << endl;
    // for (size_t p = 0; p < nPatches; ++p)
    //     for (size_t neg = 0; neg < nNegatives; ++neg) {
    //         for (size_t i = 0; i < fvNeg[p][neg].size(); ++i) {
    //             outputFile << fvNeg[p][neg][i];
    //         }
    //         outputFile << endl;
    //     }

    // outputFile.close();

    return 0;
}
