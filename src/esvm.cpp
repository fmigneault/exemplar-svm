#include "esvm.h"
#include "esvmOptions.h"
#include "generic.h"

#include <sys/stat.h>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "boost/filesystem.hpp"
namespace bfs = boost::filesystem;

/*
    Initializes and trains an ESVM using list of positive and negative feature vectors
*/
ESVM::ESVM(std::vector<FeatureVector> positives, std::vector<FeatureVector> negatives, std::string id)
{
    ASSERT_THROW(positives.size() > 0 && negatives.size() > 0, "Exemplar-SVM cannot train without both positive and negative feature vectors");
        
    int posSamples = positives.size();
    int negSamples = negatives.size();
    targetID = id;

    std::vector<int> targets(posSamples + negSamples, ESVM_NEGATIVE_CLASS);
    for (int s = 0; s < posSamples; s++)
        targets[s] = ESVM_POSITIVE_CLASS;

    std::vector<FeatureVector> samples;
    samples.insert(samples.end(), positives.begin(), positives.end());
    samples.insert(samples.end(), negatives.begin(), negatives.end());

    // train with penalty weights according to specified mode
    // greater penalty attributed to incorrectly classifying a positive vs the many negatives 
    std::vector<double> weights = calcClassWeightsFromMode(posSamples, negSamples);
    trainModel(samples, targets, weights);    
}

/*
    Initializes and trains an ESVM using a combined list of positive and negative feature vectors with corresponding labels    
*/
ESVM::ESVM(std::vector<FeatureVector> samples, std::vector<int> targetOutputs, std::string id)
{
    int Np = std::count(targetOutputs.begin(), targetOutputs.end(), ESVM_POSITIVE_CLASS);
    int Nn = std::count(targetOutputs.begin(), targetOutputs.end(), ESVM_NEGATIVE_CLASS);
    targetID = id;

    // train with penalty weights according to specified mode
    // greater penalty attributed to incorrectly classifying a positive vs the many negatives 
    std::vector<double> weights = calcClassWeightsFromMode(Np, Nn);
    trainModel(samples, targetOutputs, weights);
}

/*
    Initializes and trains an ESVM using a pre-generated file of feature vectors and correponding labels
    The file must be saved in the LIBSVM sample data format
*/
ESVM::ESVM(std::string trainingSamplesFilePath, std::string id)
{
    // get samples
    std::vector<FeatureVector> samples;
    std::vector<int> targets;
    readSampleDataFile(trainingSamplesFilePath, samples, targets);

    int Np = std::count(targets.begin(), targets.end(), ESVM_POSITIVE_CLASS);
    int Nn = std::count(targets.begin(), targets.end(), ESVM_NEGATIVE_CLASS);
    targetID = id;

    // train using loaded samples
    std::vector<double> weights = calcClassWeightsFromMode(Np, Nn);
    trainModel(samples, targets, weights);    
}

/* 
    Initializes and trains an ESVM using a pre-loaded and pre-trained SVM model
    Model can be saved with 'saveModelFile' method in LIBSVM format and retrieved with 'svm_load_model'
*/
ESVM::ESVM(svm_model* trainedModel, std::string id)
{
    checkModelParameters_assert(trainedModel);
    resetModel(trainedModel);
    targetID = id;
}

/*ESVM::~ESVM()
{
    logstream logger(LOGGER_FILE);
    logger << "BEFORE FREE MODEL REF: " << isModelTrained() << std::endl;
    if (isModelTrained())
        svm_free_model_content(model.p);
    logger << "AFTER CONTENT FREE - DELETE MODEL" << std::endl;
    delete model;
}*/

void ESVM::resetModel(svm_model* model)
{
    if (esvmModel)
        svm_free_model_content(esvmModel.get());
    esvmModel.reset(model);
}

void ESVM::logModelParameters(bool displaySV)
{
    logModelParameters(esvmModel.get(), targetID, displaySV);
}

void ESVM::logModelParameters(svm_model *model, std::string id, bool displaySV)
{
    logstream logger(LOGGER_FILE);
    if (!model)
    {
        logger << "ESVM model is 'null'" << std::endl;
        return;
    }

    std::vector<double> svCoefVector;
    if (model->sv_coef)
        svCoefVector = std::vector<double>(model->sv_coef[0], model->sv_coef[0] + model->l);

    logger << "ESVM model trained with parameters:" << std::endl
           << "   targetID:     " << id << std::endl
           << "   C:            " << model->param.C << std::endl
           << "   eps:          " << model->param.eps << std::endl           
           << "   shrinking:    " << model->param.shrinking << std::endl
           << "   probability:  " << model->param.probability << std::endl;
    if (model->param.probability)
    {
        logger << "   probA:        " << (model->probA == nullptr ? "'null'" : std::to_string(model->probA[0])) << std::endl
               << "   probB:        " << (model->probB == nullptr ? "'null'" : std::to_string(model->probB[0])) << std::endl;
    }
    logger << "   W mode:       " << ESVM_WEIGHTS_MODE << std::endl
           << "   nr W:         " << model->param.nr_weight << std::endl
           #if ESVM_WEIGHTS_MODE
           << "   W pos:         " << model->param.weight[0] << std::endl
           << "   W neg:         " << model->param.weight[1] << std::endl
           << "   W pos label:   " << model->param.weight_label[0] << std::endl
           << "   W neg label:   " << model->param.weight_label[1] << std::endl
           #endif/*ESVM_WEIGHTS_MODE*/
           << "   free sv:      " << model->free_sv << std::endl
           << "   pos label:    " << model->label[0] << std::endl
           << "   neg label:    " << model->label[1] << std::endl
           << "   nr class:     " << model->nr_class << std::endl
           << "   pos SV:       " << model->nSV[0] << std::endl
           << "   neg SV:       " << model->nSV[1] << std::endl
           << "   total SV:     " << model->l << std::endl
           << "   SV:           " << (displaySV ? model->SV ? "" : "'null'" : "'displaySV=false'") << std::endl;
    if (displaySV && model->SV)
        for (int sv = 0; sv < model->l; ++sv)
            logger << "      " << featuresToVectorString(getFeatureVector(model->SV[sv])) << std::endl;
    logger << "   rho:          " << (model->rho ? std::to_string(model->rho[0]) : "'null'") << std::endl
           << "   SV coef:      " << featuresToVectorString(svCoefVector) << std::endl;
}

/*
    Verifies that the specified 'svm_model' parameters are adequately set to be employed by the ESVM class
*/
bool ESVM::checkModelParameters(svm_model* model) 
{ 
    try { checkModelParameters_assert(model); }
    catch (...) { return false; }
    return true;
}

/*
    Verifies that the specified 'svm_model' parameters are adequately set to be employed by the ESVM class
    This is the version employed internally by the class, other method (no '_assert') is publicly available for quick validation test
*/
void ESVM::checkModelParameters_assert(svm_model* model)
{
    ASSERT_THROW(model != nullptr, "No SVM model reference specified");
    ASSERT_THROW(model->param.svm_type == C_SVC, "ESVM model must be a C-SVM classifier");
    ASSERT_THROW(model->param.kernel_type == LINEAR, "ESVM model must have a LINEAR kernel");
    ASSERT_THROW(model->nr_class == 2, "ESVM model must have two classes (positives, negatives)");
    ASSERT_THROW(model->l > 1, "Number of samples must be greater than one (at least 1 positive and 1 negative)");
    ASSERT_THROW(model->nSV[0] > 0, "Number of positive ESVM support vector must be greater than zero");
    ASSERT_THROW(model->nSV[1] > 0, "Number of negative ESVM support vector must be greater than zero");
    ASSERT_THROW((model->label[0] == ESVM_POSITIVE_CLASS && model->label[1] == ESVM_NEGATIVE_CLASS) ||
                 (model->label[1] == ESVM_POSITIVE_CLASS && model->label[0] == ESVM_NEGATIVE_CLASS),
                 "ESVM model labels must be set to expected distinct positive and negative class values");

    #if ESVM_USE_PREDICT_PROBABILITY
    ASSERT_THROW(model->param.probability == 1, "Probability option disabled when it should be set to '1'");
    ASSERT_THROW(model->probA != nullptr, "Reference of probability estimate parameter 'probA' not specified for ESVM using probability prediction");
    ASSERT_THROW(model->probB != nullptr, "Reference of probability estimate parameter 'probB' not specified for ESVM using probability prediction");
    #else
    ASSERT_THROW(model->param.probability == 0, "Probability option enabled when it should be set to '0'");
    ASSERT_THROW(model->probA == nullptr, "Reference of probability estimate parameter 'probA' not null for ESVM not using probability prediction");
    ASSERT_THROW(model->probB == nullptr, "Reference of probability estimate parameter 'probB' not null for ESVM not using probability prediction");
    #endif/*ESVM_USE_PREDICT_PROBABILITY*/

    if (model->free_sv == 0)        // trained from samples
    {
        ASSERT_THROW(model->param.C > 0, "ESVM model cost must be greater than zero");
        int nWeights = model->param.nr_weight;
        ASSERT_THROW(nWeights == 0 || nWeights == 2, "ESVM model must have either two weights (positive, negative) or none");
        if (nWeights == 2)
        {
            ASSERT_THROW(model->param.weight[0] > 0, "ESVM model positive class weight must be greater than zero");
            ASSERT_THROW(model->param.weight[1] > 0, "ESVM model negative class weight must be greater than zero");
            ASSERT_THROW(model->param.weight_label[0] == model->label[0], "ESVM model weight label [0] must match label [0]");
            ASSERT_THROW(model->param.weight_label[1] == model->label[1], "ESVM model weight label [1] must match label [1]");
        }
    }
    else if (model->free_sv == 1)   // loaded from pre-trained file
    {
        ASSERT_THROW(model->rho != nullptr, "ESVM model constant for decision function must be specified");
        ASSERT_THROW(model->sv_coef != nullptr, "ESVM model coefficients container for decision functions must be specified");
        ASSERT_THROW(model->sv_coef[0] != nullptr, "ESVM model specific coefficients for unique decision function must be specified");
        ASSERT_THROW(model->SV != nullptr, "ESVM model support vector container must be specified");
        for (int sv = 0; sv < model->l; ++sv)
            ASSERT_THROW(model->SV[sv] != nullptr, "ESVM model specific support vectors must be specified");
    }
    else
        throw std::runtime_error("Unsupported model 'free_sv' mode");
}

/*
    Verifies if the specified header can be found at the start of a binary file
*/
bool ESVM::checkBinaryHeader(std::ifstream& binaryFileStream, std::string header)
{
    if (!binaryFileStream.is_open()) return false;
    int headerLength = header.size();
    char *headerCheck = new char[headerLength + 1];     // +1 for the terminating '\0'
    binaryFileStream.read(headerCheck, headerLength);
    headerCheck[headerLength] = '\0';                   // avoids comparing different strings because '\0' is not found
    bool isFound = (header == std::string(headerCheck));
    delete headerCheck;
    return isFound;
}

/*
    Loads an ESVM model file form the specified model format
*/
bool ESVM::loadModelFile(std::string modelFilePath, FileFormat format, std::string id)
{
    targetID = (id == "") ? modelFilePath : id;

    if (format == LIBSVM)
        resetModel(svm_load_model(modelFilePath.c_str()));
    else if (format == BINARY)
        loadModelFile_binary(modelFilePath);
    else
        throw std::runtime_error("Unsupported file format");

    return isModelTrained();
}

/*
    Reads and updates the ESVM from a pre-trained LIBSVM model file (if possible)
    This function can easily break if the model doesn't meet all requirements.
    Calling the 'checkModelParameters' method is suggested before calling to limit potential errors.
*/
void ESVM::loadModelFile_libsvm(std::string filePath)
{
    std::ifstream modelFile(filePath, std::ios::in | std::ios::binary);
    bool isBinary = checkBinaryHeader(modelFile, ESVM_BINARY_HEADER_MODEL);
    if (modelFile.is_open()) 
        modelFile.close();
    if (!isBinary)
    {
        resetModel(svm_load_model(filePath.c_str()));
        ASSERT_THROW(esvmModel == nullptr, "Model loaded from LIBSVM file should either be uninitialized or set as loaded from 'svm_load'");
    }
}

/*
    Reads and updates the ESVM from a pre-trained BINARY model file
    (see writing function for expected format)
*/
void ESVM::loadModelFile_binary(std::string filePath)
{
    // check for opened file
    std::ifstream modelFile(filePath, std::ios::in | std::ios::binary);
    ASSERT_THROW(modelFile.is_open(), "Failed to open the specified model BINARY file: '" + filePath + "'");

    svm_model* model;
    try
    {
        // check for header
        ASSERT_THROW(checkBinaryHeader(modelFile, ESVM_BINARY_HEADER_MODEL), "Expected BINARY file header was not found");

        // set assumed parameters and prepare containers
        resetModel(new svm_model);
        svm_parameter param;        
        param.svm_type = C_SVC;
        param.kernel_type = LINEAR;
        model->param = param;
        model->nr_class = 2;        
        model->rho = new double[1];                 // 1 decision function parameter
        model->sv_coef = new double*[1];            // 1 x N sv coefficients for 1 decision function
        model->label = new int[model->nr_class];
        model->nSV = new int[model->nr_class];

        // labels required to determine/ensure of the order of positives/negatives SV saved to file
        modelFile.read(reinterpret_cast<char*>(&model->rho[0]), sizeof(model->rho[0]));
        modelFile.read(reinterpret_cast<char*>(&model->label[0]), model->nr_class * sizeof(model->label[0]));
        ASSERT_THROW((model->label[0] == ESVM_POSITIVE_CLASS && model->label[1] == ESVM_NEGATIVE_CLASS) ||
                     (model->label[1] == ESVM_POSITIVE_CLASS && model->label[0] == ESVM_NEGATIVE_CLASS),
                     "Read labels are not set to expected distinct positive and negative class values");

        // read general parameters
        int nFeatures = 0;
        modelFile.read(reinterpret_cast<char*>(&model->nSV[0]), model->nr_class * sizeof(model->nSV[0]));   // positive/negative SV
        modelFile.read(reinterpret_cast<char*>(&nFeatures), sizeof(nFeatures));                             // features count for each SV
        model->l = model->nSV[0] + model->nSV[1];                                                           // total number of support vectors
        ASSERT_THROW(model->l > 0, "Read total number of support vectors should be greater than zero");
        ASSERT_THROW(model->nSV[0] > 0, "Read number of positive support vectors should be greater than zero");
        ASSERT_THROW(model->nSV[1] > 0, "Read number of negative support vectors should be greater than zero");
        ASSERT_THROW(nFeatures > 0, "Read number of features should be greater than zero");

        // read support vectors and decision function coefficients
        std::vector<FeatureVector> sampleSV(model->l);
        model->sv_coef[0] = new double[model->l];
        modelFile.read(reinterpret_cast<char*>(&model->sv_coef[0][0]), model->l * sizeof(model->sv_coef[0][0]));
        model->SV = new svm_node*[model->l];
        for (int sv = 0; sv < model->l; sv++)
        {
            sampleSV[sv] = FeatureVector(nFeatures);
            modelFile.read(reinterpret_cast<char*>(&sampleSV[sv][0]), nFeatures * sizeof(&sampleSV[sv][0]));
            model->SV[sv] = getFeatureNodes(sampleSV[sv]);
            ASSERT_THROW(modelFile.good(), "Invalid file stream status when reading model");
        }

        model->param.probability = ESVM_USE_PREDICT_PROBABILITY;

        model->free_sv = 1;     // flag model obtained from pre-trained file instead of trained from samples
        modelFile.close();
        checkModelParameters_assert(model);
        resetModel(model);
    }
    catch (std::exception& ex)
    {
        if (modelFile.is_open())
            modelFile.close();
        if (model)
            svm_free_model_content(model);
        throw ex;
    }    
}

/*
    Saves the ESVM model file in the specified model format
*/
bool ESVM::saveModelFile(std::string modelFilePath, FileFormat format)
{
    ASSERT_THROW(isModelTrained(), "Cannot save an untrained model");

    if (format == LIBSVM)
        return svm_save_model(modelFilePath.c_str(), esvmModel.get()) == 0;     // 0 if success, -1 otherwise
    else if (format == BINARY)
    {        
        saveModelFile_binary(modelFilePath);
        struct stat buffer;
        return stat(modelFilePath.c_str(), &buffer) != 1;   // quickly checks if a file exists
    }
    else
        throw std::runtime_error("Unsupported file format");
}

/*
    Writes the ESVM model to a BINARY model file
*/
void ESVM::saveModelFile_binary(std::string filePath)
{
    /*
    Expected data format and order:    <all reinterpreted as char*>

        TYPE          QUANTITY                VALUE
        ========================================
        (char)      | len(header)           | 'ESVM_BINARY_HEADER_MODEL'
        (double)    | 1                     | rho (constant in decision function - only one since 2 classes)
        (int)       | 2                     | class labels (corresponding to following support vectors order)      
        (int)       | 2                     | nSV (number of support vectors per corresponding class label)
        (int)       | 1                     | nFeatures (number of features forming support vectors)
        (double)    | sum(nSV)              | coefficients of decision function for corresponding support vectors
        (double)    | sum(nSV) * nFeatures  | support vector features
    */

    std::ofstream modelFile(filePath, std::ios::out | std::ios::binary);
    ASSERT_THROW(modelFile.is_open(), "Failed to open the specified model BINARY file: '" + filePath + "'");

    try
    {
        // get support vector and feature counts (expected valid dimensions from properly trained model)
        int nFeatures = getFeatureVector(esvmModel->SV[0]).size();
        ASSERT_THROW(esvmModel->l > 0, "Cannot save a model that doesn't contain any support vector");
        ASSERT_THROW(nFeatures > 0, "Cannot save a model with support vectors not containing any feature");

        // write header and counts for later reading
        std::string headerStr = ESVM_BINARY_HEADER_MODEL;
        const char *headerChar = headerStr.c_str();
        modelFile.write(headerChar, headerStr.size());
        modelFile.write(reinterpret_cast<const char*>(&esvmModel->rho[0]), sizeof(esvmModel->rho[0]));
        modelFile.write(reinterpret_cast<const char*>(&esvmModel->label[0]), esvmModel->nr_class * sizeof(esvmModel->label[0]));
        modelFile.write(reinterpret_cast<const char*>(&esvmModel->nSV[0]), esvmModel->nr_class * sizeof(esvmModel->nSV[0]));
        modelFile.write(reinterpret_cast<const char*>(&nFeatures), sizeof(nFeatures));

        // write support vectors and decision function coefficients    
        modelFile.write(reinterpret_cast<const char*>(&esvmModel->sv_coef[0][0]), esvmModel->l * sizeof(esvmModel->sv_coef[0][0]));
        for (int sv = 0; sv < esvmModel->l; sv++)
            modelFile.write(reinterpret_cast<const char*>(&esvmModel->SV[sv][0]), nFeatures * sizeof(esvmModel->SV[sv][0]));
        modelFile.close();
    }
    catch(std::exception& ex)
    {
        if (modelFile.is_open())
            modelFile.close();
        throw ex;
    }
}

/*
    Reads feature vectors from a data sample file
*/
void ESVM::readSampleDataFile(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors, FileFormat format)
{
    std::vector<int> dummyOutputTargets;
    readSampleDataFile(filePath, sampleFeatureVectors, dummyOutputTargets, format);
}

/*
    Reads feature vectors and corresponding target output class from a data sample file
*/
void ESVM::readSampleDataFile(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors, 
                              std::vector<int>& targetOutputs, FileFormat format)
{
    if (format == BINARY)
        readSampleDataFile_binary(filePath, sampleFeatureVectors, targetOutputs);
    else if (format == LIBSVM)
        readSampleDataFile_libsvm(filePath, sampleFeatureVectors, targetOutputs);
    else
        throw std::runtime_error("Unsupported file format");
}

/*
    Reads feature vectors from a BINARY data sample file
    (see writing function for expected format)
*/
void ESVM::readSampleDataFile_binary(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors, std::vector<int>& targetOutputs)
{    
    // check for opened file
    std::ifstream samplesFile(filePath, std::ios::in | std::ios::binary);
    ASSERT_THROW(samplesFile.is_open(), "Failed to open the specified samples data BINARY file: '" + filePath + "'");

    try
    {
        // check for header
        ASSERT_THROW(checkBinaryHeader(samplesFile, ESVM_BINARY_HEADER_SAMPLES), "Expected BINARY file header was not found");

        // check for samples and feature counts
        int nSamples = 0;   // warning: format 'int' required, not 'size_t' for matching binary dimension
        int nFeatures = 0;  // warning: format 'int' required, not 'size_t' for matching binary dimension
        samplesFile.read(reinterpret_cast<char*>(&nSamples), sizeof(nSamples));
        samplesFile.read(reinterpret_cast<char*>(&nFeatures), sizeof(nFeatures));
        ASSERT_THROW(nSamples > 0, "Read number of samples should be greater than zero");
        ASSERT_THROW(nFeatures > 0, "Read number of features should be greater than zero");

        // retrieve sample features and target outputs
        sampleFeatureVectors = std::vector<FeatureVector>(nSamples);
        targetOutputs = std::vector<int>(nSamples);
        samplesFile.read(reinterpret_cast<char*>(&targetOutputs[0]), nSamples * sizeof(targetOutputs[0]));
        for (int s = 0; s < nSamples; s++)
        {
            sampleFeatureVectors[s] = FeatureVector(nFeatures);        
            samplesFile.read(reinterpret_cast<char*>(&sampleFeatureVectors[s][0]), nFeatures * sizeof(sampleFeatureVectors[s][0]));
            ASSERT_THROW(samplesFile.good(), "Invalid file stream status when reading samples");
        }
        samplesFile.close();
    }
    catch (std::exception& ex)
    {
        // avoid locked file from assert failure
        if (samplesFile.is_open())
            samplesFile.close();
        throw ex;   // re-throw
    }
}

/*
    Reads feature vectors and corresponding target output class from a LIBSVM formatted data sample file
*/
void ESVM::readSampleDataFile_libsvm(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors, std::vector<int>& targetOutputs)
{
    std::ifstream samplesFile(filePath);
    ASSERT_THROW(samplesFile, "Could not open specified ESVM samples data file: '" + filePath + "'");

    try
    {
        std::vector<FeatureVector> samples;
        std::vector<int> targets;
        size_t nFeatures = 0;
        static std::string delimiter = ":";
        static int offDelim = delimiter.length();

        // loop each line
        while (samplesFile)
        {
            std::string line;
            if (!getline(samplesFile, line)) break;

            bool firstPart = true;
            std::istringstream ssline(line);
            std::string spart;
            int prev = 0;
            int index = 0;
            double value = 0;
            FeatureVector features;
            
            // loop each part delimited by a space
            while (ssline)
            {
                if (!getline(ssline, spart, ' ')) break;
                if (firstPart)
                {
                    // Reading label
                    ASSERT_THROW(spart.find(delimiter) == std::string::npos, "Could not find class label before sample of specified file");
                    
                    #if ESVM_READ_LIBSVM_PARSER_MODE == 0
                    int target = 0;
                    std::istringstream(spart) >> target;
                    #elif ESVM_READ_LIBSVM_PARSER_MODE == 1
                    int target = std::strtol(spart.c_str(), NULL, 10);
                    #elif ESVM_READ_LIBSVM_PARSER_MODE == 2
                    int target = parse(spart.c_str());
                    #else
                    throw std::runtime_error("Undefined parser mode");
                    #endif/*ESVM_READ_LIBSVM_PARSER_MODE*/

                    ASSERT_THROW(target == ESVM_POSITIVE_CLASS || target == ESVM_NEGATIVE_CLASS, "Invalid class label specified in file for ESVM");
                    targets.push_back(target);
                    firstPart = false;
                }
                else
                {
                    // Reading features
                    size_t offset = spart.find(delimiter);
                    ASSERT_THROW(offset != std::string::npos, "Failed to find feature 'index:value' delimiter");
                    
                    #if ESVM_READ_LIBSVM_PARSER_MODE == 0
                    std::istringstream(spart.substr(0, offset)) >> index;
                    std::istringstream(spart.erase(0, offset + offDelim)) >> value;
                    #elif ESVM_READ_LIBSVM_PARSER_MODE == 1
                    index = std::strtol(spart.substr(0, offset).c_str(), NULL, 10);
                    value = std::strtod(spart.erase(0, offset + offDelim).c_str(), NULL);
                    #elif ESVM_READ_LIBSVM_PARSER_MODE == 2
                    index = parse(spart.substr(0, offset).c_str());
                    value = parse(spart.erase(0, offset + offDelim).c_str());
                    logstream logger(LOGGER_FILE);
                    logger << "INDEX: " << std::setprecision(12) << index << std::endl;
                    logger << "value: " << std::setprecision(12) << value << std::endl;
                    #else
                    throw std::runtime_error("Undefined parser mode");
                    #endif/*ESVM_READ_LIBSVM_PARSER_MODE*/

                    // end reading index:value if termination index found (-1), otherwise check if still valid index
                    if (index == -1) break;
                    ASSERT_THROW(index - prev > 0, "Feature indexes must be in ascending order");

                    // Add omitted sparse features (zero value features)
                    while (index - prev > 1)
                    {
                        features.push_back(0);
                        prev++;
                    }
                    features.push_back(value);
                    prev++;
                }
            }

            if (nFeatures == 0)
                nFeatures = features.size();
            else
                ASSERT_THROW(nFeatures == features.size(), "Loaded feature vectors must have a consistent dimension");
            samples.push_back(features);
        }
        ASSERT_THROW(samplesFile.eof(), "Reading ESVM samples file finished without reaching EOF");

        sampleFeatureVectors = samples;
        targetOutputs = targets;
    }
    catch (std::exception& ex)
    {
        // avoid locked file from assert failure
        if (samplesFile.is_open())
            samplesFile.close();
        throw ex;   // re-throw
    }
}

/*
    Writes feature vectors and corresponding target output class to a data sample file
*/
void ESVM::writeSampleDataFile(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors,
                               std::vector<int>& targetOutputs, FileFormat format)
{
    size_t nSamples = sampleFeatureVectors.size();
    ASSERT_THROW(nSamples > 0, "Number of samples must be greater than zero");
    ASSERT_THROW(nSamples == targetOutputs.size(), "Number of samples must match number of corresponding target outputs");
    size_t nFeatures = sampleFeatureVectors[0].size();
    ASSERT_THROW(nFeatures > 0, "Number of features in samples must be greater than zero");

    for (size_t s = 0; s < nSamples; ++s)
        ASSERT_THROW(sampleFeatureVectors[s].size() == nFeatures, "Inconsistent number of features in samples");

    if (format == BINARY)
        writeSampleDataFile_binary(filePath, sampleFeatureVectors, targetOutputs);
    else if (format == LIBSVM)
        writeSampleDataFile_libsvm(filePath, sampleFeatureVectors, targetOutputs);
    else
        throw std::runtime_error("Unsupported file format");
}

/*
    Writes feature vectors and corresponding target output class to a BINARY data sample file
*/
void ESVM::writeSampleDataFile_binary(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors, std::vector<int>& targetOutputs)
{
    /*
    Expected data format and order:    <all reinterpreted as char*>

        TYPE          QUANTITY                VALUE
        =======================================================
        (char)      | len(header)           | 'ESVM_BINARY_HEADER_SAMPLES'
        (int)       | 1                     | nSamples
        (int)       | 1                     | nFeatures
        (int)       | nSamples              | Targets
        (double)    | nSamples * nFeatures  | Samples Features
    */

    // check opened file
    std::ofstream samplesFile(filePath, std::ios::out | std::ios::binary);
    ASSERT_THROW(samplesFile.is_open(), "Failed to open the specified samples data BINARY file: '" + filePath + "'");
    
    // get sample and feature counts (already checked valid dimensions from calling function)
    int nSamples = sampleFeatureVectors.size();     // warning: format 'int' required, not 'size_t' for matching binary dimension
    int nFeatures = sampleFeatureVectors[0].size(); // warning: format 'int' required, not 'size_t' for matching binary dimension

    // write header and counts for later reading
    std::string headerStr = ESVM_BINARY_HEADER_SAMPLES;
    const char *headerChar = headerStr.c_str();
    samplesFile.write(headerChar, headerStr.size());
    samplesFile.write(reinterpret_cast<const char*>(&nSamples), sizeof(nSamples));
    samplesFile.write(reinterpret_cast<const char*>(&nFeatures), sizeof(nFeatures));

    // write target outputs and sample features
    samplesFile.write(reinterpret_cast<const char*>(&targetOutputs[0]), nSamples * sizeof(targetOutputs[0]));
    for (int s = 0; s < nSamples; s++)    
        samplesFile.write(reinterpret_cast<const char*>(&sampleFeatureVectors[s][0]), nFeatures * sizeof(sampleFeatureVectors[s][0]));    

    samplesFile.close();
}

/*
    Writes feature vectors and corresponding target output class to a LIBSVM formatted data sample file
*/
void ESVM::writeSampleDataFile_libsvm(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors, std::vector<int>& targetOutputs)
{
    std::ofstream samplesFile(filePath);
    ASSERT_THROW(samplesFile, "Could not open specified ESVM sample data file: '" + filePath + "'");

    // get sample and feature counts (already checked valid dimensions from calling function)
    size_t nSamples = sampleFeatureVectors.size();
    size_t nFeatures = sampleFeatureVectors[0].size();

    for (size_t s = 0; s < nSamples; ++s)
    {
        ASSERT_THROW(targetOutputs[s] == ESVM_POSITIVE_CLASS || targetOutputs[s] == ESVM_NEGATIVE_CLASS, 
                     "Target output value must correspond to either positive or negative class");
        samplesFile << targetOutputs[s];
        for (size_t f = 0; f < nFeatures; ++f)
            samplesFile << " " << (f + 1) << ":" << sampleFeatureVectors[s][f];
        samplesFile << std::endl;
    }
}

/*
    Trains the ESVM using the sample feature vectors and their corresponding target outpus
*/
void ESVM::trainModel(std::vector<FeatureVector> samples, std::vector<int> targetOutputs, std::vector<double> classWeights)
{    
    ASSERT_THROW(samples.size() > 1, "Number of samples must be greater than one (at least 1 positive and 1 negative)");
    ASSERT_THROW(samples.size() == targetOutputs.size(), "Number of samples must match number of corresponding target outputs");
    ASSERT_THROW(classWeights.size() == 2, "Exemplar-SVM expects two weigths (positive, negative)");

    logstream logger(LOGGER_FILE);

    svm_problem prob;
    prob.l = samples.size();    // number of training data        
    
    // convert and assign training vectors and corresponding target values for classification 
    prob.y = new double[prob.l];
    prob.x = new svm_node*[prob.l];
    /// ############################################# #pragma omp parallel for
    for (int s = 0; s < prob.l; s++)
    {
        prob.y[s] = targetOutputs[s];
        prob.x[s] = getFeatureNodes(samples[s]);
    }
        
    /// ################################################ DEBUG
    /// logger << "'trainModel' samples converted to 'svm_node'" << std::endl;
    /*
    logger << "ESVM training samples | outputs" << std::endl;
    for (int s = 0; s < samples.size(); s++)
        logger << "      " << featuresToVectorString(samples[s]) << " | " << prob.y[s] << std::endl;
    */
    /// ################################################ DEBUG

    // set training parameters    
    svm_parameter param;
    param.C = 1;                // cost constraint violation used for w*C
    param.svm_type = C_SVC;     // cost classifier SVM
    param.kernel_type = LINEAR; // linear kernel
    param.eps = 0.001;          // stopping optimization criterion
    param.probability = 0;      // possibility to use probability outputs instead of (+1,-1) classes (adds extra training time)
    param.shrinking = 1;        // use problem shrinking heuristics
    param.cache_size = 100;     // size in MB

    /* unused default values
          libsvm 'svm_check_parameter' sometimes returns an error if some parameters don't pass verifications values althought
          these parameters are not employed by the current SVM/kernel types, simply set valid values to avoid random errors
    */
    param.coef0 = 0;            // coefficient of { POLY, SIGMOID } kernels
    param.degree = 0;           // degree of POLY kernel 
    param.gamma = 0;            // gamma of { POLY, RBF, SIGMOID } kernels
    param.nu = 0;               // nu for { NU_SVC, ONE_CLASS, NU_SVR } SVMs
    param.p = 0.1;              // epsilon in epsilon-insensitive loss function of EPSILON_SVR    

    #if ESVM_WEIGHTS_MODE == 0
    param.nr_weight = 0;
    param.weight = nullptr;
    param.weight_label = nullptr;
    #else/*ESVM_WEIGHTS_MODE != 0*/
    param.nr_weight = 2;                                                            // number of weights
    param.weight = new double[2] { classWeights[0], classWeights[1] };              // class weights (positive, negative)
    param.weight_label = new int[2] { ESVM_POSITIVE_CLASS, ESVM_NEGATIVE_CLASS };   // class labels    
    #endif/*ESVM_WEIGHTS_MODE*/

    // validate parameters and train models    
    try
    {
        const char* msg = svm_check_parameter(&prob, &param);
        ASSERT_THROW(msg == nullptr, "Failure message from 'svm_check_parameter': " + std::string(msg) + "\n");
    }
    catch(std::exception& ex)
    {
        logger << "Exception occurred during parameter check: " << ex.what() << std::endl;
        throw ex;
    }

    logger << "ESVM training..." << std::endl;
    try
    {
        resetModel(svm_train(&prob, &param));
    }
    catch (std::exception& ex)
    {
        logger << "Exception occurred during ESVM training: " << ex.what() << std::endl;
        throw ex;
    }

    // free problem 
    // cannot free inner x[] 'svm_node' as they are shared with 'model->SV'
    delete[] prob.y;
    delete[] prob.x;
    /*for (int s = 0; s < prob.l; ++s)
        delete[] prob.x[s];
    */
    /// ################################################ DEBUG
    logModelParameters(esvmModel.get(), targetID, false);
    /// ################################################ DEBUG
}

bool ESVM::isModelTrained()
{

    logModelParameters(true);   /// TODO REMOVE


    /// TODO - check multiple parameters or not?
    return (esvmModel != nullptr);  /// && model->SV != nullptr && model->sv_coef != nullptr && model->
}

/*
    Calculates positive and negative class weights (Wp, Wn) according to the specified weighting mode.

    ESVM_WEIGHTS_MODE:
        0: (Wp = 0, Wn = 0)         unused
        1: (Wp = 1, Wn = 0.01)      enforced values
        2: (Wp = 100, Wn = 1)       enforced values
        3: (Wp = N/Np, Wn = N/Nn)   ratio of sample counts
        4: (Wp = 1, Wn = Np/Nn)     ratio of sample counts normalized for positives (Np/Nn = [N/Nn]/[N/Np])
*/
std::vector<double> ESVM::calcClassWeightsFromMode(int positivesCount, int negativesCount)
{
    int Np = positivesCount;
    int Nn = negativesCount;
    int N = Nn + Np;
    ASSERT_THROW(Np > 0, "Number of positives must be greater than zero");
    ASSERT_THROW(Nn > 0, "Number of negatives must be greater than zero");

    #if ESVM_WEIGHTS_MODE == 0
    double Wp = 0;
    double Wn = 0;
    #elif ESVM_WEIGHTS_MODE == 1
    double Wp = 1;
    double Wn = 0.01;
    #elif ESVM_WEIGHTS_MODE == 2
    double Wp = 100;
    double Wn = 1;
    #elif ESVM_WEIGHTS_MODE == 3
    double Wp = (double)N / (double)Np;
    double Wn = (double)N / (double)Nn;
    #elif ESVM_WEIGHTS_MODE == 4
    double Wp = 1;
    double Wn = (double)Np / (double)Nn;
    #endif/*ESVM_WEIGHTS_MODE*/

    /// ################################################ DEBUG
    logstream logger(LOGGER_FILE);
    logger << "ESVM weight initialization" << std::endl
           << "   N:  " << N  << std::endl
           << "   Np: " << Np << std::endl
           << "   Nn: " << Nn << std::endl
           << "   Wp: " << Wp << std::endl
           << "   Wn: " << Wn << std::endl;
    /// ################################################ DEBUG

    return { Wp, Wn };
}

/*
    Predicts the classification value for the specified feature vector sample using the trained ESVM model.
*/
double ESVM::predict(FeatureVector probeSample)
{
    ASSERT_THROW(isModelTrained(), "Cannot predict with not trained ESVM model");

    #if ESVM_USE_PREDICT_PROBABILITY
    if (model->param.probability)
    {
        /*
        double* probEstimates = (double *)malloc(model->nr_class * sizeof(double)); // = new double[model->nr_class];
        double p = svm_predict_probability(model, getFeatureVector(probeSample), probEstimates);
        */
        double* probEstimates = new double[model->nr_class];
        svm_predict_probability(model, getFeatureVector(probeSample), probEstimates);

        /// ################################################ DEBUG
        logstream logger(LOGGER_FILE);
        logger << "ESVM predict" << std::endl;
        for (int s = 0; s < model->nr_class; s++)
        {
            logger << "   probEstimates " << s << ": " << probEstimates[s] << std::endl;
        }
        /// ################################################ DEBUG

        //return p;
        return probEstimates[0];
    }
    #endif/*ESVM_USE_PREDICT_PROBABILITY*/
    
    // Obtain decision values directly (instead of predicted label/probability from 'svm_predict'/'svm_predict_probability')
    // Since the number of decision values of each class combination is calculated with [ nr_class*(nr_class-1)/2 ],
    // and that we have only 2 classes, we have only one decision value (positive vs. negative)    
    double* decisionValues = new double[esvmModel->nr_class * (esvmModel->nr_class - 1) / 2]; 
    svm_predict_values(esvmModel.get(), getFeatureNodes(probeSample), decisionValues);
    return decisionValues[0];
}

/*
    Predicts the classification values for the specified list of feature vector samples using the trained ESVM model.
*/
std::vector<double> ESVM::predict(std::vector<FeatureVector> probeSamples)
{
    int nPredictions = probeSamples.size();
    std::vector<double> outputs(nPredictions);
    for (int p = 0; p < nPredictions; p++)
        outputs[p] = predict(probeSamples[p]);
    return outputs;
}

/*
    Predicts all classification values for each of the feature vector samples within the file using the trained ESVM model.
    The file must be saved in the LIBSVM sample data format.
    Ground truth class read from the file are returned using 'probeGroundTruths' if specified. 
*/
std::vector<double> ESVM::predict(std::string probeSamplesFilePath, std::vector<int>* probeGroundTruths)
{
    std::vector<int> classGroundTruths;
    std::vector<FeatureVector> samples;
    readSampleDataFile(probeSamplesFilePath, samples, classGroundTruths);
    if (probeGroundTruths != nullptr)
        *probeGroundTruths = classGroundTruths;
    return predict(samples);
}

/*
    Converts an array of LIBSVM 'svm_node' to a feature vector
    Assumes that the last feature node is (-1,?), but it is not inclued in the feature vector
*/
FeatureVector ESVM::getFeatureVector(svm_node* features)
{
    FeatureVector fv;
    int f = 0;
    while (features[f].index != -1)    
        fv.push_back(features[f++].value);
    return fv;
}

/*
    Converts a feature vector to an array of LIBSVM 'svm_node'   
*/
svm_node* ESVM::getFeatureNodes(FeatureVector features)
{
    return getFeatureNodes(&features[0], features.size());
}

/*
    Converts an array of 'double' features to an array of LIBSVM 'svm_node'
*/
svm_node* ESVM::getFeatureNodes(double* features, int featureCount)
{
    svm_node* fv = new svm_node[featureCount + 1];
    /// ############################################# #pragma omp parallel for
    for (int f = 0; f < featureCount; f++)
    {
        fv[f].index = f + 1;        // indexes should be one based
        fv[f].value = features[f];
    }
    fv[featureCount].index = -1;    // Additional feature value must be (-1,?) to end the vector (see LIBSVM README)
    return fv;
}
