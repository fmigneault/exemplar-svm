#include "esvm.h"
#include "esvmOptions.h"
#include "esvmUtils.h"

#include "datafile.h"
#include "testing.h"

#include "boost/filesystem.hpp"
namespace bfs = boost::filesystem;

/*
    Initializes and trains an ESVM using list of positive and negative feature vectors
*/
ESVM::ESVM(vector<FeatureVector> positives, vector<FeatureVector> negatives, string id)
    : targetID(id), esvmModel(nullptr)
{
    ASSERT_THROW(positives.size() > 0 && negatives.size() > 0, "Exemplar-SVM cannot train without both positive and negative feature vectors");
        
    int posSamples = (int)positives.size();
    int negSamples = (int)negatives.size();

    vector<int> targets(posSamples + negSamples, ESVM_NEGATIVE_CLASS);
    for (int s = 0; s < posSamples; ++s)
        targets[s] = ESVM_POSITIVE_CLASS;

    vector<FeatureVector> samples;
    samples.insert(samples.end(), positives.begin(), positives.end());
    samples.insert(samples.end(), negatives.begin(), negatives.end());

    // train with penalty weights according to specified mode
    // greater penalty attributed to incorrectly classifying a positive vs the many negatives 
    vector<double> weights = calcClassWeightsFromMode(posSamples, negSamples);
    trainModel(samples, targets, weights);    
}

/*
    Initializes and trains an ESVM using a combined list of positive and negative feature vectors with corresponding labels    
*/
ESVM::ESVM(vector<FeatureVector> samples, vector<int> targetOutputs, string id)
    : targetID(id), esvmModel(nullptr)
{
    int Np = (int)count(targetOutputs.begin(), targetOutputs.end(), ESVM_POSITIVE_CLASS);
    int Nn = (int)count(targetOutputs.begin(), targetOutputs.end(), ESVM_NEGATIVE_CLASS);

    // train with penalty weights according to specified mode
    // greater penalty attributed to incorrectly classifying a positive vs the many negatives 
    vector<double> weights = calcClassWeightsFromMode(Np, Nn);
    trainModel(samples, targetOutputs, weights);
}

/*
    Initializes and trains an ESVM using a pre-generated file of feature vectors and correponding labels
    The file must be saved in the LIBSVM sample data format
*/
ESVM::ESVM(string trainingSamplesFilePath, string id)
    : targetID(id), esvmModel(nullptr)
{
    // get samples
    vector<FeatureVector> samples;
    vector<int> targets;
    readSampleDataFile(trainingSamplesFilePath, samples, targets);

    int Np = (int)count(targets.begin(), targets.end(), ESVM_POSITIVE_CLASS);
    int Nn = (int)count(targets.begin(), targets.end(), ESVM_NEGATIVE_CLASS);
    targetID = id;

    // train using loaded samples
    vector<double> weights = calcClassWeightsFromMode(Np, Nn);
    trainModel(samples, targets, weights);    
}

/* 
    Initializes and trains an ESVM using a pre-loaded and pre-trained SVM model
    Model can be saved with 'saveModelFile' method in LIBSVM format and retrieved with 'svm_load_model'
*/
ESVM::ESVM(svmModel* trainedModel, string id)
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
    targetID = esvm.targetID;
    esvmModel = deepCopyModel(esvm.esvmModel);
}

// Move constructor
ESVM::ESVM(ESVM&& esvm)
{
    this->swap(*this, esvm);
    
    /*
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
    */
}

// Copy assigment
ESVM& ESVM::operator=(ESVM esvm)
{
    ///TODO REMOVE
    ///logstream logger(LOGGER_FILE);
    ///logger << "EQUAL_CTOR!" << endl;

    //// check for self-assignment
    //if (&esvm == this)
    //    return *this;
    //
    //targetID = esvm.targetID;
    //esvmModel = deepCopyModel(esvm.esvmModel);
    //return *this;

    swap(*this, esvm);
    return *this;
}

// Builds an 'empty' model ensuring all 'null' references
svmModel* ESVM::makeEmptyModel()
{
    svmModel* model = Malloc(svmModel, 1);
    
    model->nr_class = 0;
    model->label = nullptr;

    #if ESVM_USE_LIBSVM

    model->free_sv = 0;
    model->l = 0;    
    model->param.probability = ESVM_USE_PREDICT_PROBABILITY;    
    model->nSV = nullptr;
    model->probA = nullptr;
    model->probB = nullptr;
    model->rho = nullptr;
    model->SV = nullptr;
    model->sv_coef = nullptr;
    model->sv_indices = nullptr;

    #elif ESVM_USE_LIBLINEAR

    model->bias = 0;
    model->nr_feature = 0;
    model->w = nullptr;
    model->param.C = 0;
    model->param.eps = 0;
    model->param.nr_weight = 0;
    model->param.weight = nullptr;
    model->param.weight_label = nullptr;
    model->param.init_sol = nullptr;

    #endif

    return model;
}

// Deepcopy of all 'svmModel' subparts
svmModel* ESVM::deepCopyModel(svmModel* model)
{
    if (!model) return nullptr;
    
    // deep copy of memory
    svmModel* newModel = makeEmptyModel();
    newModel->param = model->param;
    newModel->nr_class = model->nr_class;
    newModel->label = Malloc(int, newModel->nr_class);
    for (int c = 0; c < newModel->nr_class; ++c)
        newModel->label[c] = model->label[c];
    
    if (getFreeSV(model) == FreeModelState::PARAM) {
        if (!model->param.weight)
            newModel->param.weight = nullptr;
        else {
            newModel->param.weight = Malloc(double, newModel->param.nr_weight);
            newModel->param.weight_label = Malloc(int, newModel->param.nr_weight);
            for (int w = 0; w < newModel->param.nr_weight; ++w) {
                newModel->param.weight[w] = model->param.weight[w];
                newModel->param.weight_label[w] = model->param.weight_label[w];
            }
        }
    }

    #if ESVM_USE_LIBSVM
    
    newModel->free_sv = model->free_sv;
    newModel->l = model->l;

    if (getFreeSV(model) != FreeModelState::PARAM) {
        newModel->nSV = Malloc(int, newModel->nr_class);
        for (int c = 0; c < newModel->nr_class; ++c)
            newModel->nSV[c] = model->nSV[c];

        newModel->sv_coef = Malloc(double*, model->nr_class - 1);
        for (int c_1 = 0; c_1 < newModel->nr_class - 1; ++c_1) {
            newModel->sv_coef[c_1] = Malloc(double, newModel->l);
            for (int cn = 0; cn < newModel->l; ++cn)
                newModel->sv_coef[c_1][cn] = model->sv_coef[c_1][cn];
        }

        int nFeatures = 0;                              // contains +1 for (-1,?)
        while (model->SV[0][nFeatures++].index != -1);  // count 'svm_nodes'

        newModel->sv_indices = (model->sv_indices) ? Malloc(int, newModel->l) : nullptr;
        newModel->SV = Malloc(svm_node*, newModel->l);
        for (int sv = 0; sv < newModel->l; ++sv) {
            if (model->sv_indices)
                newModel->sv_indices[sv] = model->sv_indices[sv];
            newModel->SV[sv] = Malloc(svm_node, nFeatures);
            for (int f = 0; f < nFeatures; ++f)
                newModel->SV[sv][f] = model->SV[sv][f];
        }

        int nClassPairWise = newModel->nr_class*(newModel->nr_class - 1) / 2;
        newModel->rho = Malloc(double, nClassPairWise);
        for (int cPW = 0; cPW < nClassPairWise; ++cPW)
            newModel->rho[cPW] = model->rho[cPW];

        if (ESVM_USE_PREDICT_PROBABILITY && newModel->param.probability && model->probA && model->probB) {
            newModel->param.probability = 1;
            newModel->probA = Malloc(double, nClassPairWise);
            newModel->probB = Malloc(double, nClassPairWise);
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

    #elif ESVM_USE_LIBLINEAR

    newModel->bias = model->bias;
    newModel->nr_feature = model->nr_feature;

    if (getFreeSV(model) != FreeModelState::PARAM) {
        int nw = (model->nr_feature + model->bias) * model->nr_class;
        newModel->w = Malloc(double, nw);
        std::memcpy(newModel->w, model->w, nw * sizeof(double));
    }

    #endif

    return newModel;
}

// Deallocates all model contained memory block and the model itself
void ESVM::destroyModel(svmModel** model)
{
    if (model != nullptr && *model != nullptr)
    {
        svmModel* pModel = *model;
        FreeNull(pModel->label);

        bool freeParam = getFreeSV(pModel) == FreeModelState::PARAM || getFreeSV(pModel) == FreeModelState::MULTI;
        bool freeModel = getFreeSV(pModel) == FreeModelState::MODEL || getFreeSV(pModel) == FreeModelState::MULTI;
        
        if (freeParam) {
            FreeNull(pModel->param.weight);
            FreeNull(pModel->param.weight_label);
        }

        #if ESVM_USE_LIBSVM

        if (freeModel) {
            FreeNull(pModel->rho);
            FreeNull(pModel->nSV);
            FreeNull(pModel->sv_indices);
            if (pModel->sv_coef)
                for (int c = 0; c < pModel->nr_class - 1; ++c)
                    free(pModel->sv_coef[c]);
            FreeNull(pModel->sv_coef);
            if (pModel->SV)
                for (int sv = 0; sv < pModel->l; ++sv)
                    free(pModel->SV[sv]);
            FreeNull(pModel->SV);
        }

        if (ESVM_USE_PREDICT_PROBABILITY && pModel->param.probability) {
            free(pModel->probA);
            free(pModel->probB);
        }
        pModel->probA = nullptr;
        pModel->probB = nullptr;

        #elif ESVM_USE_LIBLINEAR

        if (freeParam)
            FreeNull(pModel->param.init_sol);
        if (freeModel)
            FreeNull(pModel->w);

        #endif

        FreeNull(pModel);
        *model = nullptr;
    }
}

// Deallocation of unused support vectors and training parameters, updates parameters accordingly for new model state
//      free only sample vectors not used as support vectors and update model state
//      cannot directly free inner 'svm_node' 2d-array 'problem->x[]' as they are shared with 'model->SV'
void ESVM::removeTrainedModelUnusedData(svmModel* model, svmProblem* problem)
{
    #if ESVM_USE_LIBSVM

    ASSERT_THROW(model != nullptr, "Missing model reference to remove unused sample vectors and training paramters");
    ASSERT_THROW(model->sv_indices != nullptr, "Missing model 'sv_indices' reference to indicate which unused sample vectors to remove");
    ASSERT_THROW(model->free_sv == FreeModelState::PARAM, "Improper 'free_sv' mode to allow deallocation of unused sample vectors and parameters");
    ASSERT_THROW(problem != nullptr, "Missing problem reference to remove unused sample vectors");
    ASSERT_THROW(problem->x != nullptr, "Missing problem contained sample references to remove unused sample vectors");

    // remove unused sample vectors
    int iSV = 0;
    for (int sv = 0; sv < problem->l; ++sv) {
        if (iSV < model->l &&                   // avoid accessing pass the lass 'model->SV' when all visited
            model->sv_indices[iSV] - 1 == sv)   // indices in 'sv_indices' are one-based
            iSV++;                              // increment SV indices until all found
        else
            free(problem->x[sv]);
    }

    // remove unused training paramters
    if (model->param.probability) {
        free(model->param.weight);
        free(model->param.weight_label);
    }
    model->param.weight = nullptr;
    model->param.weight_label = nullptr;

    // destroy problem contained data
    FreeNull(problem->x);
    FreeNull(problem->y);

    // update mode
    model->free_sv = FreeModelState::MODEL;

    #endif/*ESVM_USE_LIBSVM*/
}

// Deallocation of model subparts
void ESVM::resetModel(svmModel* model, bool copy)
{
    if (isModelSet())
        destroyModel(&esvmModel);
    esvmModel = copy ? deepCopyModel(model) : model;    // set requested model or 'null'
}

// Free SV status according to employed base SVM library
FreeModelState ESVM::getFreeSV(svmModel* model)
{
    #if ESVM_USE_LIBSVM
    return model->free_sv;
    #elif ESVM_USE_LIBLINEAR
    return (model->w != nullptr ? FreeModelState::PARAM : FreeModelState::MODEL);
    #endif
}

// Destructor
ESVM::~ESVM()
{
    try { resetModel(); }
    catch (...) {}
}

void ESVM::logModelParameters(bool displaySV) const
{
    logModelParameters(esvmModel, targetID, displaySV);
}

void ESVM::logModelParameters(svmModel *model, string id, bool displaySV)
{
    logstream logger(LOGGER_FILE);
    if (!model)
    {
        logger << "ESVM model is 'null'" << endl;
        return;
    }

    logger << "ESVM model parameters:" << endl
           << "   base SVM:      " << ESVM_BASE << endl
           << "   targetID:      " << id << endl
           << "   svm type:      " << svm_type_name(model) << endl
           << "   kernel type:   " << svm_kernel_name(model) << endl
           << "   C:             " << (getFreeSV(model) == FreeModelState::MODEL ? "n/a (pre-trained)" : to_string(model->param.C)) << endl
           << "   eps:           " << (getFreeSV(model) == FreeModelState::MODEL ? "n/a (pre-trained)" : to_string(model->param.eps)) << endl
           << "   nr class:      " << model->nr_class << endl
           << "   pos label:     " << model->label[0] << endl
           << "   neg label:     " << model->label[1] << endl
           << "   W mode:        " << ESVM_WEIGHTS_MODE << endl;
    if (getFreeSV(model) != FreeModelState::MODEL) {  // when using a pre-trained model, 'rho' is used instead of weights (they are not set)           
    #if ESVM_WEIGHTS_MODE
    logger << "   nr W:          " << model->param.nr_weight << endl
           << "   W pos:         " << model->param.weight[0] << endl
           << "   W neg:         " << model->param.weight[1] << endl
           << "   W pos label:   " << model->param.weight_label[0] << endl
           << "   W neg label:   " << model->param.weight_label[1] << endl;
    #endif/*ESVM_WEIGHTS_MODE*/
    } // end if

    #if ESVM_USE_LIBSVM

    vector<double> svCoefVector;
    if (model->sv_coef)
        svCoefVector = vector<double>(model->sv_coef[0], model->sv_coef[0] + model->l);
    logger << "   free sv:       " << model->free_sv << endl
           << "   shrinking:     " << (model->free_sv == FreeModelState::MODEL ? "n/a" : to_string(model->param.shrinking)) << endl
           << "   probability:   " << (model->free_sv == FreeModelState::MODEL ? "n/a" : to_string(model->param.probability)) << endl;
    if (model->param.probability) {
    logger << "   probA:         " << (model->probA == nullptr ? "'null'" : to_string(model->probA[0])) << endl
           << "   probB:         " << (model->probB == nullptr ? "'null'" : to_string(model->probB[0])) << endl; 
    } // end if
    if (model->free_sv != FreeModelState::PARAM) {  
    logger << "   rho:           " << (model->rho ? to_string(model->rho[0]) : "'null'") << endl
           << "   pos SV:        " << model->nSV[0] << endl
           << "   neg SV:        " << model->nSV[1] << endl
           << "   total SV:      " << model->l << endl
           << "   SV coef:       " << featuresToVectorString(svCoefVector) << endl
           << "   SV:            " << (displaySV ? model->SV ? "" : "'null'" : "'displaySV=false'") << endl;
    } // end if
    if (displaySV && model->SV) {
    for (int sv = 0; sv < model->l; ++sv) {
    logger << "      " << featuresToVectorString(getFeatureVector(model->SV[sv])) << endl;
    } // end for 
    } // end if

    #elif ESVM_USE_LIBLINEAR
    
    logger << "   bias:          " << model->bias << endl
           << "   nr features:   " << model->nr_feature << endl
           << "   W:             " << (displaySV ? model->w != nullptr ? "" : "'null'" : "'displaySV=false'") << endl;
    if (displaySV && model->w != nullptr) {
        int nw = (model->nr_feature + model->bias) * model->nr_class;
        FeatureVector fv(nw);
        fv.assign(model->w, model->w + nw);
        logger << "      " << featuresToVectorString(fv) << endl;
    }

    #endif
}

/*
    Verifies that the specified 'svmModel' parameters are adequately set to be employed by the ESVM class
*/
bool ESVM::checkModelParameters(svmModel* model)
{ 
    try { checkModelParameters_assert(model); }
    catch (...) { return false; }
    return true;
}

/*
    Verifies that the specified 'svmModel' parameters are adequately set to be employed by the ESVM class
    This is the version employed internally by the class, other method (no '_assert') is publicly available for quick validation test
*/
void ESVM::checkModelParameters_assert(svmModel* model)
{
    ASSERT_THROW(model != nullptr, "No SVM model reference specified");
    ASSERT_THROW(model->nr_class == 2, "ESVM model must have two classes (positives, negatives)");    
    ASSERT_THROW(model->label != nullptr, "ESVM model label container should be specified");
    ASSERT_THROW((model->label[0] == ESVM_POSITIVE_CLASS && model->label[1] == ESVM_NEGATIVE_CLASS) ||
                 (model->label[1] == ESVM_POSITIVE_CLASS && model->label[0] == ESVM_NEGATIVE_CLASS),
                 "ESVM model labels must be set to expected distinct positive and negative class values");
    
    # if ESVM_USE_LIBSVM

    ASSERT_THROW(model->param.svm_type == C_SVC, "ESVM model must be a C-SVM classifier");
    ASSERT_THROW(model->param.kernel_type == LINEAR, "ESVM model must have a LINEAR kernel");
    ASSERT_THROW(model->l > 1, "ESVN model number of samples must be greater than one (at least 1 positive and 1 negative)");

    #if ESVM_USE_PREDICT_PROBABILITY
    ASSERT_THROW(model->param.probability == 1, "Probability option disabled when it should be set to '1'");
    ASSERT_THROW(model->probA != nullptr, "Reference of probability estimate parameter 'probA' not specified for ESVM using probability prediction");
    ASSERT_THROW(model->probB != nullptr, "Reference of probability estimate parameter 'probB' not specified for ESVM using probability prediction");
    #else
    ASSERT_THROW(model->param.probability == 0, "Probability option enabled when it should be set to '0'");
    ASSERT_THROW(model->probA == nullptr, "Reference of probability estimate parameter 'probA' not null for ESVM not using probability prediction");
    ASSERT_THROW(model->probB == nullptr, "Reference of probability estimate parameter 'probB' not null for ESVM not using probability prediction");
    #endif/*ESVM_USE_PREDICT_PROBABILITY*/

    #endif/*ESVM_USE_LIBSVM*/

    bool checkParam = getFreeSV(model) == FreeModelState::PARAM || getFreeSV(model) == FreeModelState::MULTI;
    bool checkModel = getFreeSV(model) == FreeModelState::MODEL || getFreeSV(model) == FreeModelState::MULTI;

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
        #if ESVM_USE_LIBSVM

        ASSERT_THROW(model->rho != nullptr, "ESVM model constant for decision function must be specified");
        ASSERT_THROW(model->sv_coef != nullptr, "ESVM model coefficients container for decision functions must be specified");
        ASSERT_THROW(model->sv_coef[0] != nullptr, "ESVM model specific coefficients for unique decision function must be specified");
        ASSERT_THROW(model->nSV != nullptr, "ESVM model number of support vectors container must be specified");
        ASSERT_THROW(model->nSV[0] > 0, "Number of positive ESVM support vector must be greater than zero");
        ASSERT_THROW(model->nSV[1] > 0, "Number of negative ESVM support vector must be greater than zero");
        ASSERT_THROW(model->SV != nullptr, "ESVM model support vector container must be specified");
        for (int sv = 0; sv < model->l; ++sv)
            ASSERT_THROW(model->SV[sv] != nullptr, "ESVM model specific support vectors must be specified");

        #elif ESVM_USE_LIBLINEAR

        ASSERT_THROW(model->param.solver_type == L2R_L2LOSS_SVC, "ESVM model must have a L2R_L2LOSS_SVC solver");
        ASSERT_THROW(model->bias == 0, "ESVM model bias must be equal to 0");
        ASSERT_THROW(model->nr_feature > 0, "ESVM model must have a positive feature count");
        ASSERT_THROW(model->w != nullptr, "ESVM model weights must be specified");

        #endif
    }
}

/*
    Reads feature vectors and corresponding target output class from the specified formatted data sample file
*/
void ESVM::readSampleDataFile(string filePath, vector<FeatureVector>& sampleFeatureVectors, vector<int>& targetOutputs, FileFormat format)
{
    DataFile::readSampleDataFile(filePath, sampleFeatureVectors, targetOutputs, format, format == LIBSVM ? "" : ESVM_BINARY_HEADER_SAMPLES);
    for (size_t t = 0; t < targetOutputs.size(); ++t)
        ASSERT_THROW(targetOutputs[t] == ESVM_POSITIVE_CLASS || targetOutputs[t] == ESVM_NEGATIVE_CLASS, 
                     "Invalid class label specified in file for ESVM");
}

/*
    Reads feature vectors and corresponding target output class from the specified formatted data sample file
*/
void ESVM::readSampleDataFile(string filePath, vector<FeatureVector>& sampleFeatureVectors, FileFormat format)
{
    vector<int> dummyOutputTargets;
    ESVM::readSampleDataFile(filePath, sampleFeatureVectors, dummyOutputTargets, format);
}

/*
    Writes feature vectors and corresponding target output class to a data sample file
*/
void ESVM::writeSampleDataFile(string filePath, vector<FeatureVector>& sampleFeatureVectors, vector<int>& targetOutputs, FileFormat format)
{
    for (size_t t = 0; t < targetOutputs.size(); ++t)
        ASSERT_THROW(targetOutputs[t] == ESVM_POSITIVE_CLASS || targetOutputs[t] == ESVM_NEGATIVE_CLASS,
                     "Target output value must correspond to either positive or negative class");
    DataFile::writeSampleDataFile(filePath, sampleFeatureVectors, targetOutputs, format, format == LIBSVM ? "" : ESVM_BINARY_HEADER_SAMPLES);
}

/*
    Loads an ESVM model file form the specified model format
*/
bool ESVM::loadModelFile(string modelFilePath, FileFormat format, string id)
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
void ESVM::loadModelFile_libsvm(string filePath)
{
    ifstream modelFile(filePath, ios::in | ios::binary);
    bool isBinary = DataFile::checkBinaryHeader(modelFile, ESVM_BINARY_HEADER_MODEL_LIBSVM) ||
                    DataFile::checkBinaryHeader(modelFile, ESVM_BINARY_HEADER_MODEL_LIBLINEAR);
    if (modelFile.is_open()) 
        modelFile.close();
    if (!isBinary)
    {
        // load pre-trained model, returns NULL if failed, otherwise adress of trained model with 'free_sv' = 1
        svmModel *model = svmLoadModel(filePath.c_str());
        if (model && getFreeSV(model) != FreeModelState::PARAM)
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
void ESVM::loadModelFile_binary(string filePath)
{
    // check for opened file
    ifstream modelFile(filePath, ios::in | ios::binary);
    ASSERT_THROW(modelFile.is_open(), "Failed to open the specified model BINARY file: '" + filePath + "'");

    svmModel* model;
    try
    {
        #if ESVM_USE_LIBSVM

        // check for header
        ASSERT_THROW(DataFile::checkBinaryHeader(modelFile, ESVM_BINARY_HEADER_MODEL_LIBSVM), "Expected BINARY file header was not found");

        // set assumed parameters and prepare containers
        svmParam param;
        param.svm_type = C_SVC;
        param.kernel_type = LINEAR;
        model = makeEmptyModel();
        model->param = param;
        model->nr_class = 2;        
        model->rho = Malloc(double, 1);             // 1 decision function parameter
        model->sv_coef = Malloc(double*, 1);        // 1 x N sv coefficients for 1 decision function
        model->label = Malloc(int, model->nr_class);
        model->nSV = Malloc(int, model->nr_class);

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
        vector<FeatureVector> sampleSV(model->l);
        model->sv_coef[0] = Malloc(double, model->l);
        modelFile.read(reinterpret_cast<char*>(&model->sv_coef[0][0]), model->l * sizeof(model->sv_coef[0][0]));
        model->SV = Malloc(svmFeature*, model->l);
        for (int sv = 0; sv < model->l; ++sv)
        {
            sampleSV[sv] = FeatureVector(nFeatures);
            modelFile.read(reinterpret_cast<char*>(&sampleSV[sv][0]), nFeatures * sizeof(&sampleSV[sv][0]));
            model->SV[sv] = getFeatureNodes(sampleSV[sv]);
            ASSERT_THROW(modelFile.good(), "Invalid file stream status when reading model");
        }

        model->param.probability = ESVM_USE_PREDICT_PROBABILITY;
        model->probA = nullptr;
        model->probB = nullptr;

        model->free_sv = FreeModelState::MODEL; // flag model obtained from pre-trained file instead of trained from samples
        modelFile.close();
        checkModelParameters_assert(model);
        resetModel(model);

        #endif/*ESVM_USE_LIBSVM*/
    }
    catch (exception& ex)
    {
        if (modelFile.is_open())
            modelFile.close();
        if (model)
            svmFreeModel(model);
        throw ex;
    }    
}

/*
    Saves the ESVM model file in the specified model format
*/
bool ESVM::saveModelFile(string modelFilePath, FileFormat format) const
{
    ASSERT_THROW(isModelSet(), "Cannot save an unset model");

    if (format == LIBSVM)
        return svmSaveModel(modelFilePath.c_str(), esvmModel) == 0;     // 0 if success, -1 otherwise
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
void ESVM::saveModelFile_binary(string filePath) const
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

    ofstream modelFile(filePath, ios::out | ios::binary);
    ASSERT_THROW(modelFile.is_open(), "Failed to open the specified model BINARY file: '" + filePath + "'");

    try
    {
        #if ESVM_USE_LIBSVM

        // get support vector and feature counts (expected valid dimensions from properly trained model)
        int nFeatures = (int)getFeatureVector(esvmModel->SV[0]).size(); // warning: format 'int' required, not 'size_t' for matching binary dimension
        ASSERT_THROW(esvmModel->l > 0, "Cannot save a model that doesn't contain any support vector");
        ASSERT_THROW(nFeatures > 0, "Cannot save a model with support vectors not containing any feature");

        // write header and counts for later reading
        string headerStr = ESVM_BINARY_HEADER_MODEL;
        const char *headerChar = headerStr.c_str();
        modelFile.write(headerChar, headerStr.size());
        modelFile.write(reinterpret_cast<const char*>(&esvmModel->rho[0]), sizeof(esvmModel->rho[0]));
        modelFile.write(reinterpret_cast<const char*>(&esvmModel->label[0]), esvmModel->nr_class * sizeof(esvmModel->label[0]));
        modelFile.write(reinterpret_cast<const char*>(&esvmModel->nSV[0]), esvmModel->nr_class * sizeof(esvmModel->nSV[0]));
        modelFile.write(reinterpret_cast<const char*>(&nFeatures), sizeof(nFeatures));

        // write support vectors and decision function coefficients    
        modelFile.write(reinterpret_cast<const char*>(&esvmModel->sv_coef[0][0]), esvmModel->l * sizeof(esvmModel->sv_coef[0][0]));
        for (int sv = 0; sv < esvmModel->l; ++sv)
            modelFile.write(reinterpret_cast<const char*>(&esvmModel->SV[sv][0]), nFeatures * sizeof(esvmModel->SV[sv][0]));
        modelFile.close();

        #endif/*ESVM_USE_LIBSVM*/
    }
    catch(exception& ex)
    {
        if (modelFile.is_open())
            modelFile.close();
        throw ex;
    }
}

/*
    Trains the ESVM using the sample feature vectors and their corresponding target outpus
*/
void ESVM::trainModel(vector<FeatureVector> samples, vector<int> targetOutputs, vector<double> classWeights)
{    
    ASSERT_THROW(samples.size() > 1, "Number of samples must be greater than one (at least 1 positive and 1 negative)");
    ASSERT_THROW(samples.size() == targetOutputs.size(), "Number of samples must match number of corresponding target outputs");
    ASSERT_THROW(classWeights.size() == 2, "Exemplar-SVM expects two weigths (positive, negative)");

    logstream logger(LOGGER_FILE);

    svmProblem prob;
    prob.l = (int)samples.size();   // number of training data        
    
    // convert and assign training vectors and corresponding target values for classification 
    prob.y = Malloc(double, prob.l);
    prob.x = Malloc(svmFeature*, prob.l);
    /// ############################################# #pragma omp parallel for
    for (int s = 0; s < prob.l; ++s)
    {
        prob.y[s] = targetOutputs[s];
        prob.x[s] = getFeatureNodes(samples[s]);
    }

    // set training parameters    
    svmParam param;
    param.C = 1;                // cost constraint violation used for w*C
    param.eps = 0.001;          // stopping optimization criterion

    #if ESVM_USE_LIBSVM
    
    param.svm_type = C_SVC;     // cost classifier SVM
    param.kernel_type = LINEAR; // linear kernel    
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
    
    #elif ESVM_USE_LIBLINEAR

    param.solver_type = L2R_L2LOSS_SVC;

    #endif/*ESVM_USE_LIBSVM*/

    param.p = 0.1;              // epsilon in epsilon-insensitive loss function of support vector regression (SVR) types in LIBSVM/LIBLINEAR

    #if ESVM_WEIGHTS_MODE == 0
    param.nr_weight = 0;
    param.weight = nullptr;
    param.weight_label = nullptr;
    #else/*ESVM_WEIGHTS_MODE != 0*/
    param.nr_weight = 2;                    // number of weights
    param.weight = Malloc(double, 2);       // class weights (positive, negative)
    param.weight[0] = classWeights[0];
    param.weight[1] = classWeights[1];
    param.weight_label = Malloc(int, 2);    // class labels
    param.weight_label[0] = ESVM_POSITIVE_CLASS;
    param.weight_label[1] = ESVM_NEGATIVE_CLASS;        
    #endif/*ESVM_WEIGHTS_MODE*/

    // validate parameters and train models
    svmModel* trainedModel = nullptr;
    try
    {
        const char* msg = svmCheckParam(&prob, &param);
        ASSERT_THROW(msg == nullptr, "Failure message from 'svm_check_parameter': " + string(msg) + "\n");
    }
    catch(exception& ex)
    {
        logger << "Exception occurred during parameter check: [" << ex.what() << "]" << endl;
        throw ex;
    }

    logger << "ESVM training..." << endl;
    try
    {
        trainedModel = svmTrain(&prob, &param);
    }
    catch (exception& ex)
    {
        logger << "Exception occurred during ESVM training: [" << ex.what() << "]" << endl;
        throw ex;
    }

    removeTrainedModelUnusedData(trainedModel, &prob);
    resetModel(trainedModel, false);

    #if ESVM_DISPLAY_TRAIN_PARAMS
    logModelParameters(esvmModel, targetID, ESVM_DISPLAY_TRAIN_PARAMS == 2);
    #endif/*ESVM_DISPLAY_TRAIN_PARAMS*/
}

bool ESVM::isModelSet() const
{
    return (esvmModel != nullptr);
}

bool ESVM::isModelTrained() const
{
    return (isModelSet() && getFreeSV(esvmModel) != FreeModelState::PARAM);
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
vector<double> ESVM::calcClassWeightsFromMode(int positivesCount, int negativesCount)
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
    logger << "ESVM weight initialization" << endl
           << "   N:  " << N  << endl
           << "   Np: " << Np << endl
           << "   Nn: " << Nn << endl
           << "   Wp: " << Wp << endl
           << "   Wn: " << Wn << endl;
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
        double* probEstimates = Malloc(double, model->nr_class);
        svmPredictProbability(model, getFeatureVector(probeSample), probEstimates);
        return probEstimates[0];
    }
    #endif/*ESVM_USE_PREDICT_PROBABILITY*/

    // Obtain decision values directly (instead of predicted label/probability from 'svm_predict'/'svm_predict_probability')
    // Since the number of decision values of each class combination is calculated with [ nr_class*(nr_class-1)/2 ],
    // and that we have only 2 classes, we have only one decision value (positive vs. negative)    
    double* decisionValues = new double[esvmModel->nr_class * (esvmModel->nr_class - 1) / 2]; 
    svmPredictValues(esvmModel, getFeatureNodes(probeSample), decisionValues);
    double decision = decisionValues[0];
    delete[] decisionValues;
    return decision;
}

/*
    Predicts the classification values for the specified list of feature vector samples using the trained ESVM model.
*/
vector<double> ESVM::predict(vector<FeatureVector> probeSamples) const
{
    size_t nPredictions = probeSamples.size();
    vector<double> outputs(nPredictions);
    for (size_t p = 0; p < nPredictions; ++p)
        outputs[p] = this->predict(probeSamples[p]);
    return outputs;
}

/*
    Predicts all classification values for each of the feature vector samples within the file using the trained ESVM model.
    The file must be saved in the LIBSVM sample data format.
    Ground truth class read from the file are returned using 'probeGroundTruths' if specified. 
*/
vector<double> ESVM::predict(string probeSamplesFilePath, vector<int>* probeGroundTruths) const
{
    vector<int> classGroundTruths;
    vector<FeatureVector> samples;
    readSampleDataFile(probeSamplesFilePath, samples, classGroundTruths);
    if (probeGroundTruths != nullptr)
        *probeGroundTruths = classGroundTruths;
    return predict(samples);
}

/*
    Converts an array of LIBSVM 'svm_node' / LIBLINEAR 'feature_node' to a feature vector
    Assumes that the last feature node is (-1,?), but it is not inclued in the feature vector
*/
FeatureVector ESVM::getFeatureVector(svmFeature* features)
{
    FeatureVector fv;
    int f = 0;
    while (features[f].index != -1)    
        fv.push_back(features[f++].value);
    return fv;
}

/*
    Converts a feature vector to an array of LIBSVM 'svm_node' / LIBLINEAR 'feature_node'
*/
svmFeature* ESVM::getFeatureNodes(FeatureVector features)
{
    return getFeatureNodes(&features[0], (int)features.size());
}

/*
    Converts an array of 'double' features to an array of LIBSVM 'svm_node' / LIBLINEAR 'feature_node'
*/
svmFeature* ESVM::getFeatureNodes(double* features, int featureCount)
{
    svmFeature* fv = Malloc(svmFeature, featureCount + 1);
    /// ############################################# #pragma omp parallel for
    for (int f = 0; f < featureCount; ++f)
    {
        fv[f].index = f + 1;        // indexes should be one based
        fv[f].value = features[f];
    }
    fv[featureCount].index = -1;    // Additional feature value must be (-1,?) to end the vector (see LIBSVM README)
    return fv;
}
