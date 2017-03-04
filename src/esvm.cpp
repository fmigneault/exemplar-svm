#include "esvm.h"
#include "esvmOptions.h"
#include "generic.h"

#include <sys/stat.h>
#include <fstream>
#include <sstream>

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
    logstream logger(LOGGER_FILE);///TODO REMOVE
    logger << "DEBUG -- esvm 0" << std::endl; ///TODO REMOVE

    ASSERT_THROW(trainedModel != nullptr, "No SVM model reference specified to intialize ESVM");
    logger << "DEBUG -- esvm 0.1" << std::endl; ///TODO REMOVE
    ASSERT_THROW(trainedModel->param.svm_type == C_SVC, "ESVM model must be a C-SVM classifier");
    logger << "DEBUG -- esvm 0.2" << std::endl; ///TODO REMOVE
    ASSERT_THROW(trainedModel->param.kernel_type == LINEAR, "ESVM model must have a LINEAR kernel");    
    logger << "DEBUG -- esvm 0.3" << std::endl; ///TODO REMOVE
    ASSERT_THROW(trainedModel->nr_class == 2, "ESVM model must have two classes (positives, negatives)");
    logger << "DEBUG -- esvm 0.4" << std::endl; ///TODO REMOVE
    ASSERT_THROW(trainedModel->l > 1, "Number of samples must be greater than one (at least 1 positive and 1 negative)");
    logger << "DEBUG -- esvm 0.5" << std::endl; ///TODO REMOVE
    ASSERT_THROW(trainedModel->nSV[0] > 0, "Number of positive ESVM support vector must be greater than zero");
    logger << "DEBUG -- esvm 0.6" << std::endl; ///TODO REMOVE
    ASSERT_THROW(trainedModel->nSV[1] > 0, "Number of negative ESVM support vector must be greater than zero");
    logger << "DEBUG -- esvm 0.7" << std::endl; ///TODO REMOVE
    logger << trainedModel->label[0] << std::endl; ///TODO REMOVE
    logger << trainedModel->label[1] << std::endl; ///TODO REMOVE

    
    ASSERT_THROW((trainedModel->label[0] == ESVM_POSITIVE_CLASS && model->label[1] == ESVM_NEGATIVE_CLASS) ||
                 (trainedModel->label[1] == ESVM_POSITIVE_CLASS && model->label[0] == ESVM_NEGATIVE_CLASS),
                 "ESVM model labels must be set to expected distinct positive and negative class values");
    
    logger << "DEBUG -- esvm 1" << std::endl; ///TODO REMOVE

    if (trainedModel->free_sv == 0)     // trained from samples
    {
        logger << "DEBUG -- esvm free_sv 0" << std::endl; ///TODO REMOVE

        ASSERT_THROW(trainedModel->param.C > 0, "ESVM model cost must be greater than zero");
        int nWeights = trainedModel->param.nr_weight;
        ASSERT_THROW(nWeights == 0 || nWeights == 2, "ESVM model must have either two weights (positive, negative) or none");
        if (nWeights == 2)
        {
            ASSERT_THROW(trainedModel->param.weight[0] > 0, "ESVM model positive class weight must be greater than zero");
            ASSERT_THROW(trainedModel->param.weight[1] > 0, "ESVM model negative class weight must be greater than zero");
            ASSERT_THROW(trainedModel->param.weight_label[0] == trainedModel->label[0], "ESVM model weight label [0] must match label [0]");
            ASSERT_THROW(trainedModel->param.weight_label[1] == trainedModel->label[1], "ESVM model weight label [1] must match label [1]");
        }
    }
    else if (trainedModel->free_sv == 1)     // loaded from pre-trained file
    {
        logger << "DEBUG -- esvm free_sv 1" << std::endl; ///TODO REMOVE

        ASSERT_THROW(trainedModel->rho != nullptr, "ESVM model constant for decision function must be specified");
        logger << "DEBUG -- esvm free_sv 1.1" << std::endl; ///TODO REMOVE
        ASSERT_THROW(model->sv_coef != nullptr, "ESVM model coefficients container for decision functions must be specified");
        ASSERT_THROW(model->sv_coef[0] != nullptr, "ESVM model specific coefficients for unique decision function must be specified");
        logger << "DEBUG -- esvm free_sv 1.2" << std::endl; ///TODO REMOVE
        ASSERT_THROW(model->SV != nullptr, "ESVM model support vector container must be specified");
        ASSERT_THROW(model->SV[0] != nullptr, "ESVM model specific support vectors must be specified");
        logger << "DEBUG -- esvm free_sv 1.3" << std::endl; ///TODO REMOVE
    }
    else
        throw std::runtime_error("Unsupported model 'free_sv' mode");

    logger << "DEBUG -- esvm 2" << std::endl; ///TODO REMOVE

    model = trainedModel;
    targetID = id;
}

ESVM::~ESVM()
{
    if (isModelTrained())
        svm_free_and_destroy_model(&model);
}

/*
    Loads an ESVM model file form the specified model format
*/
bool ESVM::loadModelFile(std::string modelFilePath, FileFormat format, std::string id)
{
    if (!isModelTrained())
        svm_free_and_destroy_model(&model);    
    
    targetID = (id == "") ? modelFilePath : id;

    if (format == LIBSVM)
        model = svm_load_model(modelFilePath.c_str());    
    else if (format == BINARY)
        loadModelFile_binary(modelFilePath);
    else
        throw std::runtime_error("Unsupported file format");

    return isModelTrained();
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

    try
    {
        // check for header
        std::string headerStr = ESVM_MODEL_BIN_FILE_HEADER;
        const char *headerChar = headerStr.c_str();
        int headerLength = headerStr.size();
        char *headerCheck = new char[headerLength + 1]; // +1 for the terminating '\0'
        modelFile.read(headerCheck, headerLength);
        headerCheck[headerLength] = '\0';               // avoids comparing different strings because '\0' is not found
        ASSERT_THROW(std::string(headerChar) == std::string(headerCheck), "Expected BINARY model file header was not found");

        // set assumed parameters and prepare containers
        model = new svm_model;
        svm_parameter param;
        param.svm_type = C_SVC;
        param.kernel_type = LINEAR;
        model->param = param;
        model->nr_class = 2;
        model->sv_coef = new double*[model->nr_class - 1];
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
        modelFile.read(reinterpret_cast<char*>(&model->nSV[0]), model->nr_class * sizeof(model->nSV[0]));   // positive/negative SV (any order)
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

        model->free_sv = 1;     // flag model obtained from pre-trained file instead of trained from samples
        modelFile.close();
    }
    catch (std::exception& ex)
    {
        if (modelFile.is_open())
            modelFile.close();
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
        return svm_save_model(modelFilePath.c_str(), model) == 0;   // 0 if success, -1 otherwise
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
        (char)      | len(header)           | 'ESVM_MODEL_BIN_FILE_HEADER'
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
        int nFeatures = getFeatureVector(model->SV[0]).size();
        ASSERT_THROW(model->l > 0, "Cannot save a model that doesn't contain any support vector");
        ASSERT_THROW(nFeatures > 0, "Cannot save a model with support vectors not containing any feature");

        // write header and counts for later reading
        std::string headerStr = ESVM_MODEL_BIN_FILE_HEADER;
        const char *headerChar = headerStr.c_str();
        modelFile.write(headerChar, headerStr.size());
        modelFile.write(reinterpret_cast<const char*>(&model->rho[0]), sizeof(model->rho[0]));
        modelFile.write(reinterpret_cast<const char*>(&model->label[0]), model->nr_class * sizeof(model->label[0]));
        modelFile.write(reinterpret_cast<const char*>(&model->nSV[0]), model->nr_class * sizeof(model->nSV[0]));
        modelFile.write(reinterpret_cast<const char*>(&nFeatures), sizeof(nFeatures));

        // write support vectors and decision function coefficients    
        modelFile.write(reinterpret_cast<const char*>(&model->sv_coef[0][0]), model->l * sizeof(model->sv_coef[0][0]));
        for (int sv = 0; sv < model->l; sv++)
            modelFile.write(reinterpret_cast<const char*>(&model->SV[sv][0]), nFeatures * sizeof(model->SV[sv][0]));
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
    
    // check for header
    std::string headerStr = ESVM_SAMPLES_BIN_FILE_HEADER;
    const char *headerChar = headerStr.c_str();    
    int headerLength = headerStr.size();    
    char *headerCheck = new char[headerLength + 1]; // +1 for the terminating '\0'
    samplesFile.read(headerCheck, headerLength);
    headerCheck[headerLength] = '\0';               // avoids comparing different strings because '\0' is not found
    ASSERT_THROW(std::string(headerChar) == std::string(headerCheck), "Expected BINARY samples data file header was not found");
    
    // check for samples and feature counts
    int nSamples = 0;
    int nFeatures = 0;
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

/*
    Reads feature vectors and corresponding target output class from a LIBSVM formatted data sample file
*/
void ESVM::readSampleDataFile_libsvm(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors, std::vector<int>& targetOutputs)
{
    std::ifstream trainingFile(filePath);
    ASSERT_THROW(trainingFile, "Could not open specified ESVM sample data file: '" + filePath + "'");

    std::vector<FeatureVector> samples;
    std::vector<int> targets;
    int nFeatures = 0;
    static std::string delimiter = ":";
    static int offDelim = delimiter.length();

    // loop each line
    while (trainingFile)
    {
        std::string line;
        if (!getline(trainingFile, line)) break;

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
                int target = 0;
                std::istringstream(spart) >> target;
                ASSERT_THROW(target == ESVM_POSITIVE_CLASS || target == ESVM_NEGATIVE_CLASS,
                             "Invalid class label specified for ESVM training from file");
                targets.push_back(target);
                firstPart = false;
            }
            else
            {
                // Reading features
                size_t offset = spart.find(delimiter);
                ASSERT_THROW(offset != std::string::npos, "Failed to find feature 'index:value' delimiter");
                std::istringstream(spart.substr(0, offset)) >> index;
                std::istringstream(spart.erase(0, offset + offDelim)) >> value;

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
    ASSERT_THROW(trainingFile.eof(), "Reading ESVM training file finished without reaching EOF");

    sampleFeatureVectors = samples;
    targetOutputs = targets;
}

/*
    Writes feature vectors and corresponding target output class to a data sample file
*/
void ESVM::writeSampleDataFile(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors,
                               std::vector<int>& targetOutputs, FileFormat format)
{
    ASSERT_THROW(sampleFeatureVectors.size() > 0, "Number of samples must be greater than zero");
    ASSERT_THROW(sampleFeatureVectors.size() == targetOutputs.size(), "Number of samples must match number of corresponding target outputs");

    int nSamples = sampleFeatureVectors.size();
    int nFeatures = sampleFeatureVectors[0].size();
    for (int s = 0; s < nSamples; s++)
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
        (char)      | len(header)           | 'ESVM_SAMPLES_BIN_FILE_HEADER'
        (int)       | 1                     | nSamples
        (int)       | 1                     | nFeatures
        (int)       | nSamples              | Targets
        (double)    | nSamples * nFeatures  | Samples Features
    */

    // check opened file
    std::ofstream samplesFile(filePath, std::ios::out | std::ios::binary);
    ASSERT_THROW(samplesFile.is_open(), "Failed to open the specified samples data BINARY file: '" + filePath + "'");
    
    // get sample and feature counts (already checked valid dimensions from calling function)
    int nSamples = sampleFeatureVectors.size();
    int nFeatures = sampleFeatureVectors[0].size();

    // write header and counts for later reading
    std::string headerStr = ESVM_SAMPLES_BIN_FILE_HEADER;
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
    throw std::runtime_error("Not implemented");
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
        model = svm_train(&prob, &param);
    }
    catch (std::exception& ex)
    {
        logger << "Exception occurred during ESVM training: " << ex.what() << std::endl;
        throw ex;
    }

    /// ################################################ DEBUG
    logger << "ESVM trained with parameters:" << std::endl
           << "   targetID:    " << targetID << std::endl
           << "   C:           " << param.C << std::endl
           << "   eps:         " << param.eps << std::endl
           << "   probability: " << param.probability << std::endl
           << "   shrinking:   " << param.shrinking << std::endl
           << "   W mode:      " << ESVM_WEIGHTS_MODE << std::endl
           << "   nr W:        " << param.nr_weight << std::endl
           #if ESVM_WEIGHTS_MODE
           << "   Wp:          " << param.weight[0] << std::endl
           << "   Wn:          " << param.weight[1] << std::endl
           << "   Wp lbl:      " << param.weight_label[0] << std::endl
           << "   Wn lbl:      " << param.weight_label[1] << std::endl
           #endif/*ESVM_WEIGHTS_MODE*/
           ;    // end line no matter if weights are used or not
    if (param.probability)
    {
        logger << "   probA:       " << model->probA[0] << " | dummy check: " << model->probA[1] << std::endl
               << "   probB:       " << model->probB[0] << " | dummy check: " << model->probB[1] << std::endl;
    }
    /// ################################################ DEBUG
}

bool ESVM::isModelTrained()
{
                                /// TODO - check multiple parameters or not?
    return (model != nullptr);  /// && model->SV != nullptr && model->sv_coef != nullptr && model->
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
    int nClass = model->nr_class;
    double* decisionValues = new double[nClass*(nClass - 1) / 2];
    svm_predict_values(model, getFeatureNodes(probeSample), decisionValues);
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
    Converts an array of LIBSVM 'svm_nodes' to a feature vector
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
