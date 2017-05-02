#include "esvm.h"
#include "esvmOptions.h"
#include "generic.h"

#include <sys/stat.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>

#include "boost/filesystem.hpp"
namespace bfs = boost::filesystem;

/*
    Initializes and trains an ESVM using list of positive and negative feature vectors
*/
ESVM::ESVM(std::vector<FeatureVector> positives, std::vector<FeatureVector> negatives, std::string id)
    : targetID(id), esvmModel(nullptr)
{
    ASSERT_THROW(positives.size() > 0 && negatives.size() > 0, "Exemplar-SVM cannot train without both positive and negative feature vectors");
        
    int posSamples = (int)positives.size();
    int negSamples = (int)negatives.size();

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
    : targetID(id), esvmModel(nullptr)
{
    int Np = (int)std::count(targetOutputs.begin(), targetOutputs.end(), ESVM_POSITIVE_CLASS);
    int Nn = (int)std::count(targetOutputs.begin(), targetOutputs.end(), ESVM_NEGATIVE_CLASS);

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
    : targetID(id), esvmModel(nullptr)
{
    // get samples
    std::vector<FeatureVector> samples;
    std::vector<int> targets;
    readSampleDataFile(trainingSamplesFilePath, samples, targets);

    int Np = (int)std::count(targets.begin(), targets.end(), ESVM_POSITIVE_CLASS);
    int Nn = (int)std::count(targets.begin(), targets.end(), ESVM_NEGATIVE_CLASS);
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
    : targetID(id), esvmModel(nullptr)
{
    checkModelParameters_assert(trainedModel);
    resetModel(trainedModel);
    targetID = id;
}

// Default constructor
ESVM::ESVM()
    : targetID(""), esvmModel(nullptr)
{}

// Copy constructor
ESVM::ESVM(const ESVM& esvm)
{
    ///TODO REMOVE
    ///logstream logger(LOGGER_FILE);
    ///logger << "COPY_CTOR!" << std::endl;    

    targetID = esvm.targetID;
    esvmModel = deepCopyModel(esvm.esvmModel);

    ///logger << "COPY_CTOR - AFTER OPERATIONS: " << std::endl;
    ///logModelParameters(true);    
}

// Move constructor
/*
ESVM::ESVM(ESVM&& esvm)
{
    ///TODO REMOVE
    logstream logger(LOGGER_FILE);
    logger << "MOVE_CTOR!" << std::endl;

    targetID = esvm.targetID;
    esvmModel = esvm.esvmModel; // copy parameters and memory references (move)

    // remove moved model references
    if (esvm.isModelSet())
    {
        esvm.esvmModel->param.weight = nullptr;
        esvm.esvmModel->param.weight_label = nullptr;
        esvm.esvmModel->label = nullptr;
        esvm.esvmModel->probA = nullptr;
        esvm.esvmModel->probB = nullptr;
        esvm.esvmModel->rho = nullptr;
        esvm.esvmModel->nSV = nullptr;
        esvm.esvmModel->SV = nullptr;
        esvm.esvmModel->sv_coef = nullptr;
        esvm.esvmModel->sv_indices = nullptr;
        esvm.esvmModel = nullptr;
    }
}*/

// Copy assigment
ESVM& ESVM::operator=(const ESVM& esvm)
{
    ///TODO REMOVE
    ///logstream logger(LOGGER_FILE);
    ///logger << "EQUAL_CTOR!" << std::endl;

    targetID = esvm.targetID;
    esvmModel = deepCopyModel(esvm.esvmModel);
    return *this;

    //ESVM e(esvm);
    //logger << "EQUAL_CTOR - AFTER COPY - BEFORE RETURN (e): " << std::endl;
    //e.targetID = "E";
    //esvm.targetID = "ESVM";
    //e.logModelParameters(true);
    //logger << "EQUAL_CTOR - AFTER COPY - BEFORE RETURN (esvm): " << std::endl;
    //esvm.logModelParameters(true);
    //return ESVM(esvm);
}

// Builds an 'empty' model ensuring all 'null' references
svm_model* ESVM::makeEmptyModel()
{
    svm_model* model = new svm_model;
    model->free_sv = 0;
    model->l = 0;
    model->nr_class = 0;
    model->param.probability = ESVM_USE_PREDICT_PROBABILITY;
    model->label = nullptr;
    model->nSV = nullptr;
    model->probA = nullptr;
    model->probB = nullptr;
    model->rho = nullptr;
    model->SV = nullptr;
    model->sv_coef = nullptr;
    model->sv_indices = nullptr;
    return model;
}

// Deepcopy of all 'svm_model' subparts
svm_model* ESVM::deepCopyModel(svm_model* model)
{
    if (!model) return nullptr;
    
    ///logstream logger(LOGGER_FILE);///TODO REMOVE
    ///logger << "DEEPCOPY!" << std::endl;///TODO REMOVE

    // deep copy of memory
    svm_model* newModel = makeEmptyModel();
    newModel->free_sv = model->free_sv;
    newModel->l = model->l;
    newModel->nr_class = model->nr_class;
    newModel->param = model->param;    

    newModel->label = new int[newModel->nr_class];
    for (int c = 0; c < newModel->nr_class; ++c)
        newModel->label[c] = model->label[c];

    if (!model->free_sv)
    {
        ///logger << "free_sv == 0" << std::endl;///TODO REMOVE

        if (!model->param.weight)
            newModel->param.weight = nullptr;
        else {
            newModel->param.weight = new double[newModel->param.nr_weight];
            newModel->param.weight_label = new int[newModel->param.nr_weight];
            for (int w = 0; w < newModel->param.nr_weight; ++w) {
                newModel->param.weight[w] = model->param.weight[w];
                newModel->param.weight_label[w] = model->param.weight_label[w];
            }
        }
    }
    else
    {
        ///logger << "Free_sv != 0 (1)" << std::endl;///TODO REMOVE
        
        newModel->nSV = new int[newModel->nr_class];
        for (int c = 0; c < newModel->nr_class; ++c)
            newModel->nSV[c] = model->nSV[c];

        ///logger << "Free_sv != 0 (2)" << std::endl;///TODO REMOVE

        newModel->sv_coef = new double*[model->nr_class - 1];
        for (int c_1 = 0; c_1 < newModel->nr_class - 1; ++c_1) {
            newModel->sv_coef[c_1] = new double[newModel->l];
            for (int cn = 0; cn < newModel->l; ++cn)
                newModel->sv_coef[c_1][cn] = model->sv_coef[c_1][cn];
        }

        ///logger << "Free_sv != 0 (3)" << std::endl;///TODO REMOVE

        int nFeatures = 0;
        while (model->SV[0][nFeatures++].index != -1); // count 'svm_nodes'

        ///logger << "Free_sv != 0 (4) " << std::to_string(nFeatures) << " " << std::to_string(newModel->l) << std::endl;///TODO REMOVE
        
        newModel->sv_indices = (model->sv_indices) ? new int[newModel->l] : nullptr;
        newModel->SV = new svm_node*[newModel->l];
        for (int sv = 0; sv < newModel->l; ++sv) {
            ///logger << "Free_sv != 0 (4.1)" << std::endl;///TODO REMOVE
            if (model->sv_indices)
                newModel->sv_indices[sv] = model->sv_indices[sv];
            ///logger << "Free_sv != 0 (4.11) " << std::to_string(sv) << std::endl;///TODO REMOVE
            newModel->SV[sv] = new svm_node[nFeatures];
            ///logger << "Free_sv != 0 (4.2)" << std::endl;///TODO REMOVE
            for (int f = 0; f < nFeatures; ++f)
                newModel->SV[sv][f] = model->SV[sv][f];
            ///logger << "Free_sv != 0 (4.3)" << std::endl;///TODO REMOVE
        }

        //////logger << "Free_sv != 0 (5)" << std::endl;///TODO REMOVE

        int nClassPairWise = newModel->nr_class*(newModel->nr_class - 1) / 2;
        newModel->rho = new double[nClassPairWise];
        for (int cPW = 0; cPW < nClassPairWise; ++cPW)
            newModel->rho[cPW] = model->rho[cPW];

        ///logger << "Free_sv != 0 (6)" << std::endl;///TODO REMOVE

        if (ESVM_USE_PREDICT_PROBABILITY && newModel->param.probability && model->probA && model->probB) {
            newModel->param.probability = 1;
            newModel->probA = new double[nClassPairWise];
            newModel->probB = new double[nClassPairWise];
            for (int p = 0; p < nClassPairWise; ++p) {
                newModel->probA[p] = model->probA[p];
                newModel->probB[p] = model->probB[p];
            }
        }
        else {
            newModel->param.probability = 0;
            newModel->probA = nullptr;
            newModel->probB = nullptr;
        }
    }

    ///logger << "DEEPCOPY != null? " << (newModel != nullptr) << std::endl;///TODO REMOVE

    return newModel;
}

// Deallocates all model contained memory block and the model itself
void ESVM::destroyModel(svm_model** model)
{
    if (model != nullptr && *model != nullptr)
    {
        svm_model* pModel = *model;
        delete[] pModel->label;
        pModel->label = nullptr;

        bool freeParam = pModel->free_sv == FreeModelState::PARAM || pModel->free_sv == FreeModelState::MULTI;
        bool freeModel = pModel->free_sv == FreeModelState::MODEL || pModel->free_sv == FreeModelState::MULTI;
        
        if (freeParam) {
            delete[] pModel->param.weight;
            pModel->param.weight = nullptr;
            delete[] pModel->param.weight_label;
            pModel->param.weight_label = nullptr;
        }

        if (freeModel) {
            delete[] pModel->rho;
            pModel->rho = nullptr;
            delete[] pModel->nSV;
            pModel->nSV = nullptr;
            delete[] pModel->sv_indices;
            pModel->sv_indices = nullptr;
            if (pModel->sv_coef)
                for (int c = 0; c < pModel->nr_class - 1; ++c)
                    delete[] pModel->sv_coef[c];
            delete[] pModel->sv_coef;
            pModel->sv_coef = nullptr;
            if (pModel->SV)
                for (int sv = 0; sv < pModel->l; ++sv)
                    delete[] pModel->SV[sv];
            delete[] pModel->SV;
            pModel->SV = nullptr;
        }

        if (ESVM_USE_PREDICT_PROBABILITY && pModel->param.probability) {
            delete[] pModel->probA;
            delete[] pModel->probB;
        }
        pModel->probA = nullptr;
        pModel->probB = nullptr;

        delete[] pModel;
        *model = nullptr;
    }
}

// Deallocation of unused support vectors and training parameters, updates parameters accordingly for new model state
//      free only sample vectors not used as support vectors and update model state
//      cannot directly free inner x[] 'svm_node' as they are shared with 'model->SV'
void ESVM::removeTrainedModelUnusedData(svm_model* model, svm_problem* problem)
{
    ASSERT_THROW(model != nullptr, "Missing model reference to remove unused sample vectors and training paramters");
    ASSERT_THROW(model->sv_indices != nullptr, "Missing model 'sv_indices' reference to indicate which unused sample vectors to remove");
    ASSERT_THROW(model->free_sv == FreeModelState::PARAM, "Improper 'free_sv' mode to allow deallocation of unused sample vectors and parameters");
    ASSERT_THROW(problem != nullptr, "Missing problem reference to remove unused sample vectors");
    ASSERT_THROW(problem->x != nullptr, "Missing problem contained sample references to remove unused sample vectors");

    // remove unused sample vectors
    int i = 0;
    for (int s = 0; s < problem->l; ++s) {
        if (model->sv_indices[i] - 1 == s)  // indices are one-based
            i++;
        else
            delete[] problem->x[s];
    }

    // remove unused training paramters
    if (model->param.probability) {
        delete[] model->param.weight;
        delete[] model->param.weight_label;
    }
    model->param.weight = nullptr;
    model->param.weight_label = nullptr;

    // destroy problem contained data
    delete[] problem->x;
    delete[] problem->y;
    problem->x = nullptr;
    problem->y = nullptr;

    // update mode
    model->free_sv = FreeModelState::MODEL;
}

// Deallocation of model subparts
void ESVM::resetModel(svm_model* model, bool copy)
{
    if (isModelSet())
        destroyModel(&esvmModel);
    esvmModel = copy ? deepCopyModel(model) : model;    // set requested model or 'null'
}

// Destructor
ESVM::~ESVM()
{
    try {
        ///logstream logger(LOGGER_FILE);  ///TODO REMOVE
        ///logger << "DTOR BEFORE RESET" << std::endl;///TODO REMOVE
        ///logModelParameters(true);
        resetModel();
        ///logger << "DTOR AFTER RESET" << std::endl;///TODO REMOVE
        ///logModelParameters(true);
    }
    catch (...) {}
}

void ESVM::logModelParameters(bool displaySV) const
{
    logModelParameters(esvmModel, targetID, displaySV);
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

    logger << "ESVM model parameters:" << std::endl
           << "   targetID:      " << id << std::endl
           << "   free sv:       " << model->free_sv << std::endl
           << "   svm type:      " << svm_type_name(model) << std::endl
           << "   kernel type:   " << svm_kernel_name(model) << std::endl
           << "   nr class:      " << model->nr_class << std::endl
           << "   pos label:     " << model->label[0] << std::endl
           << "   neg label:     " << model->label[1] << std::endl
           << "   C:             " << (model->free_sv == FreeModelState::MODEL ? "n/a" : std::to_string(model->param.C)) << std::endl
           << "   eps:           " << (model->free_sv == FreeModelState::MODEL ? "n/a" : std::to_string(model->param.eps)) << std::endl
           << "   shrinking:     " << (model->free_sv == FreeModelState::MODEL ? "n/a" : std::to_string(model->param.shrinking)) << std::endl
           << "   probability:   " << (model->free_sv == FreeModelState::MODEL ? "n/a" : std::to_string(model->param.probability)) << std::endl;
    if (model->param.probability) {
    logger << "   probA:         " << (model->probA == nullptr ? "'null'" : std::to_string(model->probA[0])) << std::endl
           << "   probB:         " << (model->probB == nullptr ? "'null'" : std::to_string(model->probB[0])) << std::endl; 
    } // end if
    logger << "   W mode:        " << ESVM_WEIGHTS_MODE << std::endl;
    if (model->free_sv != FreeModelState::MODEL) {  // when using a pre-trained model, 'rho' is used instead of weights (they are not set)           
    #if ESVM_WEIGHTS_MODE
    logger << "   nr W:          " << model->param.nr_weight << std::endl
           << "   W pos:         " << model->param.weight[0] << std::endl
           << "   W neg:         " << model->param.weight[1] << std::endl
           << "   W pos label:   " << model->param.weight_label[0] << std::endl
           << "   W neg label:   " << model->param.weight_label[1] << std::endl;
    #endif/*ESVM_WEIGHTS_MODE*/
    } // end if
    if (model->free_sv != FreeModelState::PARAM) {  
    logger << "   rho:           " << (model->rho ? std::to_string(model->rho[0]) : "'null'") << std::endl
           << "   pos SV:        " << model->nSV[0] << std::endl
           << "   neg SV:        " << model->nSV[1] << std::endl
           << "   total SV:      " << model->l << std::endl
           << "   SV coef:       " << featuresToVectorString(svCoefVector) << std::endl
           << "   SV:            " << (displaySV ? model->SV ? "" : "'null'" : "'displaySV=false'") << std::endl;
    } // end if
    if (displaySV && model->SV) {
    for (int sv = 0; sv < model->l; ++sv) {
    logger << "      " << featuresToVectorString(getFeatureVector(model->SV[sv])) << std::endl;
    } // end for 
    } // end if
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

    bool checkParam = model->free_sv == FreeModelState::PARAM || model->free_sv == FreeModelState::MULTI;
    bool checkModel = model->free_sv == FreeModelState::MODEL || model->free_sv == FreeModelState::MULTI;

    if (!(checkParam || checkModel))
        THROW("Unsupported model 'free_sv' mode");

    if (checkParam)
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
    if (checkModel) 
    {
        ASSERT_THROW(model->rho != nullptr, "ESVM model constant for decision function must be specified");
        ASSERT_THROW(model->sv_coef != nullptr, "ESVM model coefficients container for decision functions must be specified");
        ASSERT_THROW(model->sv_coef[0] != nullptr, "ESVM model specific coefficients for unique decision function must be specified");
        ASSERT_THROW(model->SV != nullptr, "ESVM model support vector container must be specified");
        for (int sv = 0; sv < model->l; ++sv)
            ASSERT_THROW(model->SV[sv] != nullptr, "ESVM model specific support vectors must be specified");
    }
}

/*
    Verifies if the specified header can be found at the start of a binary file
*/
bool ESVM::checkBinaryHeader(std::ifstream& binaryFileStream, std::string header)
{
    if (!binaryFileStream.is_open()) return false;
    size_t headerLength = header.size();
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
        loadModelFile_libsvm(modelFilePath);
    else if (format == BINARY)
        loadModelFile_binary(modelFilePath);
    else
        THROW("Unsupported file format");

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
        // load pre-trained model, returns NULL if failed, otherwise adress of trained model with 'free_sv' = 1
        svm_model *model = svm_load_model(filePath.c_str());
        if (model && model->free_sv)
        {
            resetModel(model);
            return;
        }
    }
    resetModel();
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
        svm_parameter param;        
        param.svm_type = C_SVC;
        param.kernel_type = LINEAR;
        model = makeEmptyModel();
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
        model->probA = nullptr;
        model->probB = nullptr;

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
bool ESVM::saveModelFile(std::string modelFilePath, FileFormat format) const
{
    ASSERT_THROW(isModelSet(), "Cannot save an unset model");

    if (format == LIBSVM)
        return svm_save_model(modelFilePath.c_str(), esvmModel) == 0;     // 0 if success, -1 otherwise
    else if (format == BINARY)
    {        
        saveModelFile_binary(modelFilePath);
        struct stat buffer;
        return stat(modelFilePath.c_str(), &buffer) != 1;   // quickly checks if a file exists
    }
    else
        THROW("Unsupported file format");
}

/*
    Writes the ESVM model to a BINARY model file
*/
void ESVM::saveModelFile_binary(std::string filePath) const
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
        int nFeatures = (int)getFeatureVector(esvmModel->SV[0]).size(); // warning: format 'int' required, not 'size_t' for matching binary dimension
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
        THROW("Unsupported file format");
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
        static size_t offDelim = delimiter.length();

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
                    THROW("Undefined parser mode");
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
                    #else
                    THROW("Undefined parser mode");
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
        THROW("Unsupported file format");
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
    int nSamples = (int)sampleFeatureVectors.size();     // warning: format 'int' required, not 'size_t' for matching binary dimension
    int nFeatures = (int)sampleFeatureVectors[0].size(); // warning: format 'int' required, not 'size_t' for matching binary dimension

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
    prob.l = (int)samples.size();   // number of training data        
    
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
    svm_model* trainedModel = nullptr;
    try
    {
        const char* msg = svm_check_parameter(&prob, &param);
        ASSERT_THROW(msg == nullptr, "Failure message from 'svm_check_parameter': " + std::string(msg) + "\n");
    }
    catch(std::exception& ex)
    {
        logger << "Exception occurred during parameter check: [" << ex.what() << "]" << std::endl;
        throw ex;
    }

    logger << "ESVM training..." << std::endl;
    try
    {
        trainedModel = svm_train(&prob, &param);
    }
    catch (std::exception& ex)
    {
        logger << "Exception occurred during ESVM training: [" << ex.what() << "]" << std::endl;
        throw ex;
    }

    removeTrainedModelUnusedData(trainedModel, &prob);
    resetModel(trainedModel, false);

    /// ################################################ DEBUG
    logModelParameters(esvmModel, targetID, false);
    /// ################################################ DEBUG
}

bool ESVM::isModelSet() const
{
    return (esvmModel != nullptr);
}

bool ESVM::isModelTrained() const
{
    ///TODO REMOVE
    /*logstream logger(LOGGER_FILE);
    logger << "MODEL TRAINED?!" << std::endl;
    logModelParameters(true);
    */

    /// TODO - check multiple parameters or not?   && model->SV != nullptr && model->sv_coef != nullptr && model->
    return (isModelSet() && esvmModel->free_sv);
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
double ESVM::predict(FeatureVector probeSample) const
{
    ASSERT_THROW(isModelTrained(), "Cannot predict with untrained ESVM model");

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
    svm_predict_values(esvmModel, getFeatureNodes(probeSample), decisionValues);
    return decisionValues[0];
}

/*
    Predicts the classification values for the specified list of feature vector samples using the trained ESVM model.
*/
std::vector<double> ESVM::predict(std::vector<FeatureVector> probeSamples) const
{
    size_t nPredictions = probeSamples.size();
    std::vector<double> outputs(nPredictions);
    for (size_t p = 0; p < nPredictions; p++)
        outputs[p] = predict(probeSamples[p]);
    return outputs;
}

/*
    Predicts all classification values for each of the feature vector samples within the file using the trained ESVM model.
    The file must be saved in the LIBSVM sample data format.
    Ground truth class read from the file are returned using 'probeGroundTruths' if specified. 
*/
std::vector<double> ESVM::predict(std::string probeSamplesFilePath, std::vector<int>* probeGroundTruths) const
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
    return getFeatureNodes(&features[0], (int)features.size());
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

std::string svm_type_name(svm_model *model)
{
    if (model == nullptr) return "'null'";
    return svm_type_name(model->param.svm_type);
}

std::string svm_type_name(int type)
{
    switch (type)
    {
        case C_SVC:         return "C_SVC";
        case NU_SVC:        return "NU_SVC";
        case ONE_CLASS:     return "ONE_CLASS";
        case EPSILON_SVR:   return "EPSILON_SVR";
        case NU_SVR:        return "NU_SVR";
        default:            return "UNDEFINED (" + std::to_string(type) + ")";
    }
}

std::string svm_kernel_name(svm_model *model)
{
    if (model == nullptr) return "'null'";
    return svm_kernel_name(model->param.kernel_type);
}

std::string svm_kernel_name(int type)
{
    switch (type)
    {
        case LINEAR:        return "LINEAR";
        case POLY:          return "POLY";
        case RBF:           return "RBF";
        case SIGMOID:       return "SIGMOID";
        case PRECOMPUTED:   return "PRECOMPUTED";
        default:            return "UNDEFINED (" + std::to_string(type) + ")";
    }
}
