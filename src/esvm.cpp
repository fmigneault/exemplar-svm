#include "esvm.h"
#include "helperFunctions.h"
#include <fstream>
#include <sstream>

/*
    Initializes and trains an ESVM using list of positive and negative feature vectors
*/
ESVM::ESVM(std::vector< FeatureVector > positives, std::vector< FeatureVector > negatives, std::string id)
{
    ASSERT_MSG(positives.size() > 0 && negatives.size() > 0, "Exemplar-SVM cannot train without both positive and negative feature vectors");
        
    int posSamples = positives.size();
    int negSamples = negatives.size();

    std::vector<int> targets(posSamples + negSamples, ESVM_NEGATIVE_CLASS);
    for (int s = 0; s < posSamples; s++)
        targets[s] = ESVM_POSITIVE_CLASS;

    std::vector< FeatureVector > samples;
    samples.insert(samples.end(), positives.begin(), positives.end());
    samples.insert(samples.end(), negatives.begin(), negatives.end());

    // train with penalty weights according to specified mode
    // greater penalty attributed to incorrectly classifying a positive vs the many negatives 
    std::vector<double> weights = calcClassWeightsFromMode(posSamples, negSamples);
    trainEnsembleModel(samples, targets, weights);
    targetID = id;
}

/*
    Initializes and trains an ESVM using a pre-generated file of feature vectors and correponding labels
    The file must be saved in the LIBSVM sample data format
*/
ESVM::ESVM(std::string trainingFilePath, std::string id)
{
    // get samples
    std::vector< FeatureVector > samples;
    std::vector< int > targets;
    readSampleDataFile(trainingFilePath, samples, targets);

    int Np = std::count(targets.begin(), targets.end(), ESVM_POSITIVE_CLASS);
    int Nn = std::count(targets.begin(), targets.end(), ESVM_NEGATIVE_CLASS);

    // train using loaded samples
    std::vector<double> weights = calcClassWeightsFromMode(Np, Nn);
    trainEnsembleModel(samples, targets, weights);
    targetID = id;
}

/* 
    Initializes and trains an ESVM using a pre-loaded and pre-trained SVM model
    Model can be saved with 'saveModelFile' method
*/
ESVM::ESVM(svm_model* trainedModel, std::string id)
{        
    std::string posStr = std::to_string(ESVM_POSITIVE_CLASS);
    std::string negStr = std::to_string(ESVM_NEGATIVE_CLASS);
    
    ASSERT_LOG(trainedModel != nullptr, "No SVM model reference specified to intialize ESVM");
    ASSERT_LOG(trainedModel->param.svm_type == C_SVC, "ESVM model must be a C-SVM classifier");
    ASSERT_LOG(trainedModel->param.kernel_type == LINEAR, "ESVM model must have a LINEAR kernel");
    ASSERT_LOG(trainedModel->param.C > 0, "ESVM model cost must be greater than zero");
    ASSERT_LOG(trainedModel->nr_class == 2, "ESVM model must have two classes (positives, negatives)");
    ASSERT_LOG(trainedModel->label[0] == ESVM_POSITIVE_CLASS, "ESVM model positive class label [0] must be equal to " + posStr);
    ASSERT_LOG(trainedModel->label[1] == ESVM_NEGATIVE_CLASS, "ESVM model negative class label [1] must be equal to " + negStr);
    ASSERT_LOG(trainedModel->l > 1, "Number of samples must be greater than one (at least 1 positive and 1 negative)");    
    ASSERT_LOG(trainedModel->nSV[0] > 0, "Number of positive ESVM support vector must be greater than zero");
    ASSERT_LOG(trainedModel->nSV[1] > 0, "Number of negative ESVM support vector must be greater than zero");

    int nWeights = trainedModel->param.nr_weight;
    ASSERT_LOG(nWeights == 0 || nWeights == 2, "ESVM model must have either two weights (positive, negative) or none");
    if (nWeights == 2)
    {
        ASSERT_LOG(trainedModel->param.weight[0] > 0, "ESVM model positive class weight must be greater than zero");
        ASSERT_LOG(trainedModel->param.weight[1] > 0, "ESVM model negative class weight must be greater than zero");
        ASSERT_LOG(trainedModel->param.weight_label[0] == ESVM_POSITIVE_CLASS, "ESVM model positive weight label [0] must be equal to " + posStr);
        ASSERT_LOG(trainedModel->param.weight_label[1] == ESVM_NEGATIVE_CLASS, "ESVM model negative weight label [1] must be equal to " + negStr);
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
void ESVM::readSampleDataFile(std::string filePath, std::vector< FeatureVector >& sampleFeatureVectors, std::vector<int>& targetOutputs)
{
    std::ifstream trainingFile(filePath);
    ASSERT_LOG(trainingFile, "Could not open specified ESVM sample data file:\n    '" + filePath + "'");

    std::vector< FeatureVector > samples;
    std::vector< int > targets;
    int nFeatures = 0;
    std::string delimiter = ":";
    int offDelim = delimiter.length();

    // loop each line
    while (trainingFile)
    {
        std::string line;
        if (!getline(trainingFile, line)) break;

        bool firstPart = true;
        std::istringstream ss(line);
        std::string spart;
        int prev = 0;
        int index = 0;
        double value = 0;
        FeatureVector features;

        // loop each part delimited by a space
        while (ss)
        {
            getline(ss, spart, ' ');
            if (firstPart)
            {
                // Reading label
                int target = 0;
                std::istringstream(spart) >> target;
                ASSERT_LOG(target == ESVM_POSITIVE_CLASS || target == ESVM_NEGATIVE_CLASS,
                    "Invalid class label specified for ESVM training from file");
                targets.push_back(target);
                firstPart = false;
            }
            else
            {
                // Reading features
                size_t offset = spart.find(delimiter);
                ASSERT_LOG(offset != std::string::npos, "Couldn't feature index:value delimiter");
                std::istringstream(spart.substr(0, offset)) >> index;
                std::istringstream(spart.erase(0, offset + offDelim)) >> value;
                ASSERT_LOG(index - prev > 0, "Feature indexes must be in ascending order");

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
            ASSERT_LOG(nFeatures == features.size(), "Loaded feature vectors must have a consistent dimension");
        samples.push_back(features);
    }
    ASSERT_LOG(trainingFile.eof(), "Reading ESVM training file finished without reaching EOF");

    sampleFeatureVectors = samples;
    targetOutputs = targets;
}

/*
    Trains the ESVM using the sample feature vectors and their corresponding target outpus
*/
void ESVM::trainEnsembleModel(std::vector< FeatureVector > samples, std::vector<int> targetOutputs, std::vector<double> classWeights)
{    
    ASSERT_LOG(samples.size() > 1, "Number of samples must be greater than one (at least 1 positive and 1 negative)");
    ASSERT_LOG(samples.size() == targetOutputs.size(), "Number of samples must match number of corresponding target outputs");    
    ASSERT_LOG(classWeights.size() == 2, "Exemplar-SVM expects two weigths (positive, negative)");

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

    logstream logger(LOGGER_FILE);
    /// ################################################ DEBUG
    /*
    log << "ESVM training samples | outputs" << std::endl;
    for (int s = 0; s < samples.size(); s++)
        log << "      " << featuresToVectorString(samples[s]) << " | " << prob.y[s] << std::endl;
    */
    /// ################################################ DEBUG

    // set training parameters    
    svm_parameter param;
    param.C = 10;               // cost constraint violation used for w*C
    param.svm_type = C_SVC;
    param.kernel_type = LINEAR;
    /// NOT USED BY C-SVM ####  param.p = 0.1;              // sensitiveness of loss of support vector regression
    param.eps = 0.001;          // stopping optimization criterion
    param.probability = 0;      // possibility to use probability outputs instead of (+1,-1) classes (adds extra training time)
    param.shrinking = 0;        // use problem shrinking heuristics
    param.cache_size = 100;     // size in MB

    #if ESVM_WEIGHTS_MODE == 0
    param.nr_weight = 0;
    param.weight = nullptr;
    param.weight_label = nullptr;
    #else/*ESVM_WEIGHTS_MODE*/
    param.nr_weight = 2;        // number of weights
    param.weight = new double[2] { classWeights[0], classWeights[1] };              // class weights (positive, negative)
    param.weight_label = new int[2] { ESVM_POSITIVE_CLASS, ESVM_NEGATIVE_CLASS };   // class labels    
    #endif/*USE_WEIGHTS*/
    
    // validate parameters and train models    
    try
    {
        const char* msg = svm_check_parameter(&prob, &param);
        if (msg)
            throw new std::exception(msg);
    }
    catch(std::exception& ex)
    {
        logger << "Exception during parameter check: " << ex.what() << std::endl;
    }

    logger << "ESVM training..." << std::endl;
    try
    {
        ensembleModel = svm_train(&prob, &param);
    }
    catch (std::exception& ex)
    {
        logger << "Exception during ESVM training: " << ex.what() << std::endl;
    }

    /// ################################################ DEBUG
    logger << "ESVM trained with parameters:" << std::endl
        << "   C:           " << param.C << std::endl
        << "   eps:         " << param.eps << std::endl
        << "   probability: " << param.probability << std::endl
        << "   shrinking:   " << param.shrinking << std::endl
        << "   W mode:      " << ESVM_WEIGHTS_MODE << std::endl
        #if ESVM_WEIGHTS_MODE        
        << "   Wp:          " << param.weight[0] << std::endl
        << "   Wn:          " << param.weight[1] << std::endl
        << "   Wp lbl:      " << param.weight_label[0] << std::endl
        << "   Wn lbl:      " << param.weight_label[1] << std::endl;
        #endif/*ESVM_WEIGHTS_MODE*/
        << "   nr W:        " << param.nr_weight << std::endl;        
    if (param.probability)
    {
        logger << "   probA:      " << ensembleModel->probA[0] << " | dummy check: " << ensembleModel->probA[1] << std::endl;
        logger << "   probB:      " << ensembleModel->probB[0] << " | dummy check: " << ensembleModel->probB[1] << std::endl;  
    }

    /// ensembleModel->probA[0] = -4.8;
    /// ensembleModel->probB[0] = 1.20;

    /// ################################################ DEBUG
}

/*
    Calculates positive and negative class weights (Wp, Wn) according to the specified weighting mode.

    Modes:
        0: (Wp = 0, Wn = 0)         unused
        1: (Wp = 1, Wn = 0.01)      enforced values
        2: (Wp = N/Np, Wn = N/Nn)   ratio of sample counts
        3: (Wp = 1, Wn = Np/Nn)     ratio of sample counts normalized for positives (Np/Nn = [N/Nn]/[N/Np])
*/
std::vector<double> ESVM::calcClassWeightsFromMode(int positivesCount, int negativesCount)
{
    int Np = positivesCount, Nn = negativesCount, N = Nn + Np;
    ASSERT_LOG(Np > 0, "Number of positives must be greater than zero");
    ASSERT_LOG(Nn > 0, "Number of negatives must be greater than zero");
    ASSERT_LOG(Nn > Np, "Number of negatives should be (much) greater than postives for ESVM");

    #if ESVM_WEIGHTS_MODE == 0
    double Wp = 0;
    double Wn = 0;
    #elif ESVM_WEIGHTS_MODE == 1
    double Wp = 1;
    double Wn = 0.01;
    #elif ESVM_WEIGHTS_MODE == 2
    double Wp = (double)N / (double)Np;
    double Wn = (double)N / (double)Np;
    #elif ESVM_WEIGHTS_MODE == 3
    double Wp = 1;
    double Wn = (double)Np / (double)Np;
    #endif/*ESVM_WEIGHTS_MODE*/

    /// ################################################ DEBUG
    logstream log(LOGGER_FILE);
    log << "ESVM weight initialization" << std::endl
        << "   N:  " << N  << std::endl
        << "   Np: " << Np << std::endl
        << "   Nn: " << Nn << std::endl
        << "   Wp: " << Wp << std::endl
        << "   Wn: " << Wn << std::endl;
    /// ################################################ DEBUG

    return { Wp, Wn };
}

/*
    Predicts the classification value for the specified feature vector sample using the trained ESVM model
*/
double ESVM::predict(FeatureVector sample)
{    
    if (ensembleModel == nullptr)
        throw new std::exception("Ensemble model of Exemplar-SVM is not initialized");

    #if ESVM_USE_PREDICT_PROBABILITY
    if (ensembleModel->param.probability)
    {
        /*
        double* probEstimates = (double *)malloc(ensembleModel->nr_class * sizeof(double)); // = new double[ensembleModel->nr_class];
        double p = svm_predict_probability(ensembleModel, getFeatureVector(sample), probEstimates);
        */
        double* probEstimates = new double[ensembleModel->nr_class];
        svm_predict_probability(ensembleModel, getFeatureVector(sample), probEstimates);

        /// ################################################ DEBUG
        logstream log(LOGGER_FILE);
        log << "ESVM predict" << std::endl;
        for (int s = 0; s < ensembleModel->nr_class; s++)
        {
            log << "   probEstimates " << s << ": " << probEstimates[s] << std::endl;
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
    svm_predict_values(ensembleModel, getFeatureVector(sample), decisionValues);
    return decisionValues[0];
}

/*
    Predicts all classification values for each of the feature vector samples within the file using the trained ESVM model
    The file must be saved in the LIBSVM sample data format
*/
std::vector<double> ESVM::predict(std::string filename, std::vector<int>& classGroundTruths)
{
    classGroundTruths = std::vector<int>();
    return std::vector<double>();
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
