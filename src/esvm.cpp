#include "esvm.h"
#include "esvmOptions.h"
#include "generic.h"
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
    trainEnsembleModel(samples, targets, weights);    
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
    trainEnsembleModel(samples, targets, weights);    
}

/* 
    Initializes and trains an ESVM using a pre-loaded and pre-trained SVM model
    Model can be saved with 'saveModelFile' method
*/
ESVM::ESVM(svm_model* trainedModel, std::string id)
{        
    static std::string posStr = std::to_string(ESVM_POSITIVE_CLASS);
    static std::string negStr = std::to_string(ESVM_NEGATIVE_CLASS);
    
    ASSERT_THROW(trainedModel != nullptr, "No SVM model reference specified to intialize ESVM");
    ASSERT_THROW(trainedModel->param.svm_type == C_SVC, "ESVM model must be a C-SVM classifier");
    ASSERT_THROW(trainedModel->param.kernel_type == LINEAR, "ESVM model must have a LINEAR kernel");
    ASSERT_THROW(trainedModel->param.C > 0, "ESVM model cost must be greater than zero");
    ASSERT_THROW(trainedModel->nr_class == 2, "ESVM model must have two classes (positives, negatives)");
    ASSERT_THROW(trainedModel->label[0] == ESVM_POSITIVE_CLASS, "ESVM model positive class label [0] must be equal to " + posStr);
    ASSERT_THROW(trainedModel->label[1] == ESVM_NEGATIVE_CLASS, "ESVM model negative class label [1] must be equal to " + negStr);
    ASSERT_THROW(trainedModel->l > 1, "Number of samples must be greater than one (at least 1 positive and 1 negative)");
    ASSERT_THROW(trainedModel->nSV[0] > 0, "Number of positive ESVM support vector must be greater than zero");
    ASSERT_THROW(trainedModel->nSV[1] > 0, "Number of negative ESVM support vector must be greater than zero");

    int nWeights = trainedModel->param.nr_weight;
    ASSERT_THROW(nWeights == 0 || nWeights == 2, "ESVM model must have either two weights (positive, negative) or none");
    if (nWeights == 2)
    {
        ASSERT_THROW(trainedModel->param.weight[0] > 0, "ESVM model positive class weight must be greater than zero");
        ASSERT_THROW(trainedModel->param.weight[1] > 0, "ESVM model negative class weight must be greater than zero");
        ASSERT_THROW(trainedModel->param.weight_label[0] == ESVM_POSITIVE_CLASS, "ESVM model positive weight label [0] must be equal to " + posStr);
        ASSERT_THROW(trainedModel->param.weight_label[1] == ESVM_NEGATIVE_CLASS, "ESVM model negative weight label [1] must be equal to " + negStr);
    }

    ensembleModel = trainedModel;
    targetID = id;
}

/*
    Saves the ESVM model file in the LIBSVM model format
*/
bool ESVM::saveModelFile(std::string modelFilePath)
{
    return svm_save_model(modelFilePath.c_str(), ensembleModel) == 0;   // 0 if success, -1 otherwise
}

/*
    Reads feature vectors and corresponding target output class from a LIBSVM formatted data sample file
*/
void ESVM::readSampleDataFile(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors, std::vector<int>& targetOutputs)
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
    Reads feature vectors from a LIBSVM formatted data sample file
*/
void ESVM::readSampleDataFile(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors)
{
    std::vector<int> dummyOutputTargets;
    readSampleDataFile(filePath, sampleFeatureVectors, dummyOutputTargets);
}

/*
    Trains the ESVM using the sample feature vectors and their corresponding target outpus
*/
void ESVM::trainEnsembleModel(std::vector< FeatureVector > samples, std::vector<int> targetOutputs, std::vector<double> classWeights)
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
        prob.x[s] = getFeatureVector(samples[s]);
    }
        
    /// ################################################ DEBUG
    /// logger << "'trainEnsembleModel' samples converted to 'svm_node'" << std::endl;
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
    #else/*ESVM_WEIGHTS_MODE*/
    param.nr_weight = 2;                                                            // number of weights
    param.weight = new double[2] { classWeights[0], classWeights[1] };              // class weights (positive, negative)
    param.weight_label = new int[2] { ESVM_POSITIVE_CLASS, ESVM_NEGATIVE_CLASS };   // class labels    
    #endif/*USE_WEIGHTS*/

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
        ensembleModel = svm_train(&prob, &param);
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
        logger << "   probA:       " << ensembleModel->probA[0] << " | dummy check: " << ensembleModel->probA[1] << std::endl
               << "   probB:       " << ensembleModel->probB[0] << " | dummy check: " << ensembleModel->probB[1] << std::endl;  
    }
    /// ################################################ DEBUG
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
    ASSERT_THROW(ensembleModel != nullptr, "Ensemble model of Exemplar-SVM is not initialized");

    #if ESVM_USE_PREDICT_PROBABILITY
    if (ensembleModel->param.probability)
    {
        /*
        double* probEstimates = (double *)malloc(ensembleModel->nr_class * sizeof(double)); // = new double[ensembleModel->nr_class];
        double p = svm_predict_probability(ensembleModel, getFeatureVector(probeSample), probEstimates);
        */
        double* probEstimates = new double[ensembleModel->nr_class];
        svm_predict_probability(ensembleModel, getFeatureVector(probeSample), probEstimates);

        /// ################################################ DEBUG
        logstream logger(LOGGER_FILE);
        logger << "ESVM predict" << std::endl;
        for (int s = 0; s < ensembleModel->nr_class; s++)
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
    int nClass = ensembleModel->nr_class;
    double* decisionValues = new double[nClass*(nClass - 1) / 2];
    svm_predict_values(ensembleModel, getFeatureVector(probeSample), decisionValues);
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
    Converts a feature vector to an array of LIBSVM 'svm_node'
*/
svm_node* ESVM::getFeatureVector(FeatureVector features)
{
    return getFeatureVector(&features[0], features.size());
}

/*
    Converts an array of 'double' features to an array of LIBSVM 'svm_node'
*/
svm_node* ESVM::getFeatureVector(double* features, int featureCount)
{
    svm_node* fv = new svm_node[featureCount+1];
    /// ############################################# #pragma omp parallel for
    for (int f = 0; f < featureCount; f++)
    {
        fv[f].index = f + 1;        // indexes should be one based
        fv[f].value = features[f];
    }
    fv[featureCount].index = -1;    // Additional feature value must be (-1,?) to end the vector (see LIBSVM README)
    return fv;
}
