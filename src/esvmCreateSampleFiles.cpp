#include "esvmCreateSampleFiles.h"

#include "datasetChokePoint.h"
#include "testing.h"

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
                std::vector<cv::Mat> patches = imPreprocess<cv::Mat>(itDir->path().string(), imageSize, patchCounts,
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
        #if     PROC_ESVM_GENERATE_SAMPLE_FILES_NEGATIVE_COUNT == 10 
        { "0001", "0002", "0007", "0009", "0011", "0013", "0014", "0016", "0017", "0018" };
        #elif   PROC_ESVM_GENERATE_SAMPLE_FILES_NEGATIVE_COUNT == 15 
        { "0001", "0002", "0007", "0009", "0011", "0013", "0014", "0016", "0017", "0018", "0019", "0020", "0021", "0022", "0025" }; 
        #else
        ();
        #endif
    #elif PROC_ESVM_GENERATE_SAMPLE_FILES_REPLICATION == 1
        #if     PROC_ESVM_GENERATE_SAMPLE_FILES_NEGATIVE_COUNT == 10 
        { "0018", "0001", "0023", "0014", "0005", "0019", "0016", "0004", "0012", "0027" };
        #elif   PROC_ESVM_GENERATE_SAMPLE_FILES_NEGATIVE_COUNT == 15 
        { "0018", "0001", "0023", "0014", "0005", "0019", "0016", "0004", "0012", "0027", "0003", "0006", "0007", "0024", "0002" };
        #else
        ();
        #endif
    #elif PROC_ESVM_GENERATE_SAMPLE_FILES_REPLICATION == 2
        #if     PROC_ESVM_GENERATE_SAMPLE_FILES_NEGATIVE_COUNT == 10 
        { "0026", "0021", "0004", "0007", "0027", "0024", "0009", "0016", "0030", "0011" };
        #elif   PROC_ESVM_GENERATE_SAMPLE_FILES_NEGATIVE_COUNT == 15
        { "0026", "0021", "0004", "0007", "0027", "0024", "0009", "0016", "0030", "0011", "0012", "0010", "0002", "0017", "0014" };
        #else
        ();
        #endif
    #elif PROC_ESVM_GENERATE_SAMPLE_FILES_REPLICATION == 3
        #if     PROC_ESVM_GENERATE_SAMPLE_FILES_NEGATIVE_COUNT == 10 
        { "0023", "0007", "0027", "0006", "0021", "0020", "0013", "0022", "0014", "0026" };
        #elif   PROC_ESVM_GENERATE_SAMPLE_FILES_NEGATIVE_COUNT == 15
        { "0023", "0007", "0027", "0006", "0021", "0020", "0013", "0022", "0014", "0026", "0016", "0009", "0002", "0001", "0010" };
        #else
        ();
        #endif
    #elif PROC_ESVM_GENERATE_SAMPLE_FILES_REPLICATION == 4
        #if     PROC_ESVM_GENERATE_SAMPLE_FILES_NEGATIVE_COUNT == 10 
        { "0027", "0006", "0022", "0005", "0010", "0009", "0003", "0011", "0021", "0017" };
        #elif   PROC_ESVM_GENERATE_SAMPLE_FILES_NEGATIVE_COUNT == 15
        { "0027", "0006", "0022", "0005", "0010", "0009", "0003", "0011", "0021", "0017", "0026", "0007", "0025", "0002", "0013" };
        #else
        ();
        #endif
    #elif PROC_ESVM_GENERATE_SAMPLE_FILES_REPLICATION == 5
        #if     PROC_ESVM_GENERATE_SAMPLE_FILES_NEGATIVE_COUNT == 10 
        { "0005", "0019", "0021", "0023", "0002", "0016", "0011", "0027", "0018", "0029" };
        #elif   PROC_ESVM_GENERATE_SAMPLE_FILES_NEGATIVE_COUNT == 15
        { "0005", "0019", "0021", "0023", "0002", "0016", "0011", "0027", "0018", "0029", "0001", "0006", "0028", "0010", "0025" };
        #else
        ();
        #endif
    #elif PROC_ESVM_GENERATE_SAMPLE_FILES_REPLICATION == 6
        #if     PROC_ESVM_GENERATE_SAMPLE_FILES_NEGATIVE_COUNT == 10 
        { "0029", "0030", "0002", "0013", "0007", "0012", "0010", "0009", "0017", "0014" };
        #elif   PROC_ESVM_GENERATE_SAMPLE_FILES_NEGATIVE_COUNT == 15
        { "0029", "0030", "0002", "0013", "0007", "0012", "0010", "0009", "0017", "0014", "0011", "0028", "0024", "0027", "0022" };
        #else
        ();
        #endif
    #elif PROC_ESVM_GENERATE_SAMPLE_FILES_REPLICATION == 7
        #if     PROC_ESVM_GENERATE_SAMPLE_FILES_NEGATIVE_COUNT == 10 
        { "0010", "0004", "0017", "0001", "0011", "0030", "0022", "0005", "0020", "0023" };
        #elif   PROC_ESVM_GENERATE_SAMPLE_FILES_NEGATIVE_COUNT == 15
        { "0010", "0004", "0017", "0001", "0011", "0030", "0022", "0005", "0020", "0023", "0028", "0007", "0024", "0002", "0019" };
        #else
        ();
        #endif
    #elif PROC_ESVM_GENERATE_SAMPLE_FILES_REPLICATION == 8
        #if     PROC_ESVM_GENERATE_SAMPLE_FILES_NEGATIVE_COUNT == 10 
        { "0007", "0004", "0010", "0001", "0012", "0011", "0021", "0006", "0003", "0030" };
        #elif   PROC_ESVM_GENERATE_SAMPLE_FILES_NEGATIVE_COUNT == 15
        { "0007", "0004", "0010", "0001", "0012", "0011", "0021", "0006", "0003", "0030", "0017", "0026", "0023", "0029", "0027" };
        #else
        ();
        #endif
    #elif PROC_ESVM_GENERATE_SAMPLE_FILES_REPLICATION == 9
        #if     PROC_ESVM_GENERATE_SAMPLE_FILES_NEGATIVE_COUNT == 10 
        { "0020", "0025", "0015", "0030", "0007", "0023", "0011", "0017", "0002", "0009" };
        #elif   PROC_ESVM_GENERATE_SAMPLE_FILES_NEGATIVE_COUNT == 15
        { "0020", "0025", "0015", "0030", "0007", "0023", "0011", "0017", "0002", "0009", "0018", "0022", "0001", "0010", "0016" };
        #else
        ();
        #endif
    #elif PROC_ESVM_GENERATE_SAMPLE_FILES_REPLICATION == 10
        #if     PROC_ESVM_GENERATE_SAMPLE_FILES_NEGATIVE_COUNT == 10 
        { "0028", "0001", "0023", "0017", "0022", "0016", "0030", "0003", "0018", "0029" };
        #elif   PROC_ESVM_GENERATE_SAMPLE_FILES_NEGATIVE_COUNT == 15
        { "0001", "0003", "0006", "0015", "0016", "0017", "0018", "0019", "0022", "0023", "0024", "0026", "0028", "0029", "0030" };
        #else
        ();
        #endif
    #else
        ();
    #endif/*PROC_ESVM_GENERATE_SAMPLE_FILES_REPLICATION*/
}

int proc_createNegativesSampleFiles()
{
    #if PROC_ESVM_GENERATE_SAMPLE_FILES
    logstream logger(LOGGER_FILE);
    logstream logNeg("negatives-output.txt");
    logger << "Running '" << __func__ << "' test..." << std::endl;
    std::string tab = "    ";
    ChokePoint cp;

    // outputs
    std::string windowNameOriginal = "WINDOW_ORIGINAL";     // display oringal 'cropped_face' ROIs
    std::string windowNameROI = "WINDOW_ROI";               // display localized ROIs from LBP improved (if activated)
    int delayShowROI = 1;                                   // [ms] - show found LBP improved ROIs with a delay (for visual inspection)
    bool keepAllFoundROI = false;                           // keep all the found ROIs (if applicable), or only the first one (if applicable)
    ASSERT_LOG(PROC_ESVM_GENERATE_SAMPLE_FILES_BINARY || PROC_ESVM_GENERATE_SAMPLE_FILES_LIBSVM, 
               "Either 'PROC_ESVM_GENERATE_SAMPLE_FILES_BINARY' or 'PROC_ESVM_GENERATE_SAMPLE_FILES_LIBSVM' must be enabled for file generation");
    ASSERT_LOG(delayShowROI > 0, "Delay to display ROI during file generation must be greater than zero");
    ASSERT_LOG(PROC_ESVM_GENERATE_SAMPLE_FILES_SESSION >= 0 && PROC_ESVM_GENERATE_SAMPLE_FILES_SESSION <= cp.SESSION_QUANTITY,
               "Undefined value '" + std::to_string(PROC_ESVM_GENERATE_SAMPLE_FILES_SESSION) + 
               "' specified for 'PROC_ESVM_GENERATE_SAMPLE_FILES_SESSION', must be in range [0" + std::to_string(cp.SESSION_QUANTITY) + "]");

    // Negatives to employ
    std::vector<std::string> negativesID = getReplicationNegativeIDs();
    ASSERT_LOG(negativesID.size() > 0, "Negative IDs cannot be empty");
    logNeg << "Using negative IDs: " << negativesID << std::endl;

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

    // init containers / classes with parameters
    size_t nNegatives = 0;
    size_t dimsNegatives[2] = { nPatches, nNegatives };         // [patch][negative]
    xstd::mvector<2, FeatureVector> fvNegRaw(dimsNegatives);    // [patch][negative](FeatureVector)    
    std::vector<std::string> negativeSamplesID;                 // [negative](string)    

    // Loop for all ChokePoint cropped faces
    int totalSeq = PROC_ESVM_GENERATE_SAMPLE_FILES_SESSION == 0 ? cp.TOTAL_SEQUENCES : cp.TOTAL_SEQUENCES / cp.SESSION_QUANTITY;
    std::vector<int> perSessionNegatives(cp.SESSION_QUANTITY, 0);
    std::vector<int> perSequenceNegatives(totalSeq, 0);
    int seqIdx = 0;
    
    std::vector<ChokePoint::PortalType> types = { ChokePoint::PortalType::ENTER, ChokePoint::PortalType::LEAVE };
    bfs::directory_iterator endDir;
    #if PROC_ESVM_GENERATE_SAMPLE_FILES_SESSION == 0
    for (int sn = 1; sn <= cp.SESSION_QUANTITY; ++sn) {         // session number
    #else /*PROC_ESVM_GENERATE_SAMPLE_FILES_SESSION == [1-4]*/
    int sn = PROC_ESVM_GENERATE_SAMPLE_FILES_SESSION; {
    #endif/*PROC_ESVM_GENERATE_SAMPLE_FILES_SESSION*/
    for (int pn = 1; pn <= cp.PORTAL_QUANTITY; ++pn) {             // portal number
    for (auto pt = types.begin(); pt != types.end(); ++pt) {    // portal type
    for (int cn = 1; cn <= cp.CAMERA_QUANTITY; ++cn)            // camera number
    {       
        string seq = cp.getSequenceString(pn, *pt, sn, cn);

        // Add ROI to corresponding sample vectors according to individual IDs            
        for (int id = 1; id <= cp.INDIVIDUAL_QUANTITY; ++id)
        {
            std::string strID = cp.getIndividualID(id);
            if (!contains(negativesID, strID))
                logNeg << "Skipping non-negative: '" << strID << "'" << std::endl;
            else
            {
                std::string dirPath = roiChokePointCroppedFacePath + cp.getSequenceString(pn, *pt, sn, cn, id) + "/";
                logNeg << "Loading negative from directory: '" << dirPath << "'" << std::endl;
                if (bfs::is_directory(dirPath))
                {
                    for (bfs::directory_iterator itDir(dirPath); itDir != endDir; ++itDir)
                    {
                        if (bfs::is_regular_file(*itDir) && itDir->path().extension() == ".pgm")
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
    std::vector<int> negClass(nNegatives, ESVM_NEGATIVE_CLASS);
    
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
        #pragma omp parallel for
        for (long neg = 0; neg < nNegatives; ++neg) {
            fvNegMinMaxPatchOverAll[p][neg] = normalizeOverAll(MIN_MAX,    fvNegRaw[p][neg], minPatchOverAll[p],  maxPatchOverAll[p],    clip);
            fvNegZScorePatchOverAll[p][neg] = normalizeOverAll(Z_SCORE,    fvNegRaw[p][neg], meanPatchOverAll[p], stdDevPatchOverAll[p], clip);
            fvNegMinMaxPatchPerFeat[p][neg] = normalizePerFeature(MIN_MAX, fvNegRaw[p][neg], minPatchPerFeat[p],  maxPatchPerFeat[p],    clip);
            fvNegZScorePatchPerFeat[p][neg] = normalizePerFeature(Z_SCORE, fvNegRaw[p][neg], meanPatchPerFeat[p], stdDevPatchPerFeat[p], clip);
        }
        
        #if PROC_ESVM_GENERATE_SAMPLE_FILES_BINARY
        std::string h = ESVM_BINARY_HEADER_SAMPLES;
        #endif

        // write resulting sample files gradually (per patch) to distribute memory allocation
        std::string strPatch = "-patch" + std::to_string(p);
        #if PROC_ESVM_GENERATE_SAMPLE_FILES_BINARY
            DataFile::writeSampleDataFile("negatives-raw"                      + strPatch + ".bin",  fvNegRaw[p],                negClass, BINARY, h);
            DataFile::writeSampleDataFile("negatives-normPatch-minmax-overAll" + strPatch + ".bin",  fvNegMinMaxPatchOverAll[p], negClass, BINARY, h);
            DataFile::writeSampleDataFile("negatives-normPatch-zcore-overAll"  + strPatch + ".bin",  fvNegZScorePatchOverAll[p], negClass, BINARY, h);
            DataFile::writeSampleDataFile("negatives-normPatch-minmax-perFeat" + strPatch + ".bin",  fvNegMinMaxPatchPerFeat[p], negClass, BINARY, h);
            DataFile::writeSampleDataFile("negatives-normPatch-zcore-perFeat"  + strPatch + ".bin",  fvNegZScorePatchPerFeat[p], negClass, BINARY, h);
        #endif/*PROC_ESVM_GENERATE_SAMPLE_FILES_BINARY*/
        #if PROC_ESVM_GENERATE_SAMPLE_FILES_LIBSVM
            DataFile::writeSampleDataFile("negatives-raw"                      + strPatch + ".data", fvNegRaw[p],                negClass, LIBSVM);
            DataFile::writeSampleDataFile("negatives-normPatch-minmax-overAll" + strPatch + ".data", fvNegMinMaxPatchOverAll[p], negClass, LIBSVM);
            DataFile::writeSampleDataFile("negatives-normPatch-zcore-overAll"  + strPatch + ".data", fvNegZScorePatchOverAll[p], negClass, LIBSVM);
            DataFile::writeSampleDataFile("negatives-normPatch-minmax-perFeat" + strPatch + ".data", fvNegMinMaxPatchPerFeat[p], negClass, LIBSVM);
            DataFile::writeSampleDataFile("negatives-normPatch-zcore-perFeat"  + strPatch + ".data", fvNegZScorePatchPerFeat[p], negClass, LIBSVM);
        #endif/*PROC_ESVM_GENERATE_SAMPLE_FILES_LIBSVM*/

        // free vectors not required anymore
        fvNegMinMaxPatchOverAll[p].clear();
        fvNegZScorePatchOverAll[p].clear();
        fvNegMinMaxPatchPerFeat[p].clear();
        fvNegZScorePatchPerFeat[p].clear();

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

        // write resulting sample files gradually (per patch) to distribute memory allocation
        std::string strPatch = "-patch" + std::to_string(p);
        #if PROC_ESVM_GENERATE_SAMPLE_FILES_BINARY
            DataFile::writeSampleDataFile("negatives-normROI-minmax-overAll" + strPatch + ".bin",  fvNegMinMaxROIOverAll[p], negClass, BINARY, h);
            DataFile::writeSampleDataFile("negatives-normROI-zcore-overAll"  + strPatch + ".bin",  fvNegZScoreROIOverAll[p], negClass, BINARY, h);
            DataFile::writeSampleDataFile("negatives-normROI-minmax-perFeat" + strPatch + ".bin",  fvNegMinMaxROIPerFeat[p], negClass, BINARY, h);
            DataFile::writeSampleDataFile("negatives-normROI-zcore-perFeat"  + strPatch + ".bin",  fvNegZScoreROIPerFeat[p], negClass, BINARY, h);
        #endif/*PROC_ESVM_GENERATE_SAMPLE_FILES_BINARY*/
        #if PROC_ESVM_GENERATE_SAMPLE_FILES_LIBSVM
            DataFile::writeSampleDataFile("negatives-normROI-minmax-overAll" + strPatch + ".data", fvNegMinMaxROIOverAll[p], negClass, LIBSVM);
            DataFile::writeSampleDataFile("negatives-normROI-zcore-overAll"  + strPatch + ".data", fvNegZScoreROIOverAll[p], negClass, LIBSVM);
            DataFile::writeSampleDataFile("negatives-normROI-minmax-perFeat" + strPatch + ".data", fvNegMinMaxROIPerFeat[p], negClass, LIBSVM);
            DataFile::writeSampleDataFile("negatives-normROI-zcore-perFeat"  + strPatch + ".data", fvNegZScoreROIPerFeat[p], negClass, LIBSVM);
        #endif/*PROC_ESVM_GENERATE_SAMPLE_FILES_LIBSVM*/

        // free vectors not required anymore
        fvNegMinMaxROIOverAll[p].clear();
        fvNegZScoreROIOverAll[p].clear();
        fvNegMinMaxROIPerFeat[p].clear();
        fvNegZScoreROIPerFeat[p].clear();
        fvNegRaw[p].clear();
    }

    // write normalization result files
    std::vector<int> negAllROIOupput(1, ESVM_NEGATIVE_CLASS);
    std::vector<int> negPatchOutputs(nPatches, ESVM_NEGATIVE_CLASS);
    std::vector<FeatureVector> minAllROIPerFeatVec{ minAllROIPerFeat };
    std::vector<FeatureVector> maxAllROIPerFeatVec{ maxAllROIPerFeat };
    std::vector<FeatureVector> meanAllROIPerFeatVec{ meanAllROIPerFeat };
    std::vector<FeatureVector> stdDevAllROIPerFeatVec{ stdDevAllROIPerFeat };
    DataFile::writeSampleDataFile("negatives-normROI-minmax-perFeat-MIN.data",      minAllROIPerFeatVec,    negAllROIOupput, LIBSVM);
    DataFile::writeSampleDataFile("negatives-normROI-minmax-perFeat-MAX.data",      maxAllROIPerFeatVec,    negAllROIOupput, LIBSVM);
    DataFile::writeSampleDataFile("negatives-normROI-zscore-perFeat-MEAN.data",     meanAllROIPerFeatVec,   negAllROIOupput, LIBSVM);
    DataFile::writeSampleDataFile("negatives-normROI-zscore-perFeat-STDDEV.data",   stdDevAllROIPerFeatVec, negAllROIOupput, LIBSVM);
    DataFile::writeSampleDataFile("negatives-normPath-minmax-perFeat-MIN.data",     minPatchPerFeat,        negPatchOutputs, LIBSVM);
    DataFile::writeSampleDataFile("negatives-normPatch-minmax-perFeat-MAX.data",    maxPatchPerFeat,        negPatchOutputs, LIBSVM);
    DataFile::writeSampleDataFile("negatives-normPatch-zscore-perFeat-MEAN.data",   meanPatchPerFeat,       negPatchOutputs, LIBSVM);
    DataFile::writeSampleDataFile("negatives-normPatch-zscore-perFeat-STDDEV.data", stdDevPatchPerFeat,     negPatchOutputs, LIBSVM);
    
    // prepare list of feature vector normalization values
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
    std::vector<ChokePoint::PortalType> types = { ChokePoint::PortalType::ENTER, ChokePoint::PortalType::LEAVE };
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
        DataFile::writeSampleDataFile("ID0003-probes-hog-patch" + std::to_string(p) + ".bin", 
                                      fvPositiveSamples[p], targetOutputs, BINARY, ESVM_BINARY_HEADER_SAMPLES);

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
