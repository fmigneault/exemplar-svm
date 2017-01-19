#include "esvm.h"

/// ################################################ DEBUG
#include "helperFunctions.h"
/// ################################################ DEBUG

ESVM::ESVM(std::vector< FeatureVector > positives, std::vector< FeatureVector > negatives, std::string id = "")
{
    if (positives.size() <= 0 && negatives.size() <= 0)
        throw new std::exception("Exemplar-SVM cannot initialize without positive and negative feature vectors");
    
    targetID = id;
    int posSamples = positives.size();
    int negSamples = negatives.size();    
    int allSamples = posSamples + negSamples;    
    int posOutput = +1;
    int negOutput = -1;

    std::vector<int> outputs(posSamples + negSamples, negOutput);
    for (int s = 0; s < posSamples; s++)
        outputs[s] = posOutput;

    std::vector< FeatureVector > samples;
    samples.insert(samples.end(), positives.begin(), positives.end());
    samples.insert(samples.end(), negatives.begin(), negatives.end());
    
    // train with penalty weights
    // greater penalty attributed to incorrectly classifying a positive vs the many negatives
    #if SVM_WEIGHTS_MODE == 2
    double posWeight = (double)allSamples / (double)posSamples;
    double negWeight = (double)allSamples / (double)negSamples;
    double maxWeight = std::max(posWeight, negWeight);
    posWeight /= maxWeight;
    negWeight /= maxWeight;
    #elif SVM_WEIGHTS_MODE == 1
    double posWeight = 1;
    double negWeight = 0.01;
    #elif SVM_WEIGHTS_MODE == 0
    double posWeight = 0, negWeight = 0;
    #endif/*SVM_WEIGHTS_MODE*/
    /// ################################################ OVERWRITE FOR NOW

    /// ################################################ DEBUG
    logstream log(LOGGER_FILE);
    log << "ESVM initialization" << std::endl
        << "   posSamples: " << posSamples << std::endl
        << "   negSamples: " << negSamples << std::endl
        << "   posWeight:  " << posWeight << std::endl
        << "   negWeight:  " << negWeight << std::endl
        << "   posOutput:  " << posOutput << std::endl
        << "   negOutput:  " << negOutput << std::endl;
    /*    << "   samples | outputs:  " << std::endl;
    for (int s = 0; s < allSamples; s++)
        log << featuresToVectorString(samples[s]) << " | " << outputs[s] << std::endl;
    */
    /// ################################################ DEBUG

    trainEnsembleModel(samples, outputs, posOutput, negOutput, posWeight, negWeight);
}

ESVM::ESVM(std::string filename, std::string id)
{
    
}

void ESVM::trainEnsembleModel(std::vector< FeatureVector > samples, std::vector<int> outputs, 
                              int positiveOutput, int negativeOutput, double positiveWeight, double negativeWeight)
{    
    logstream logger(LOGGER_FILE);

    svm_problem prob;    
    prob.l = samples.size();    // number of training data        
    
    // convert and assign training vectors and corresponding target values for classification 
    prob.y = new double[prob.l];
    prob.x = new svm_node*[prob.l];
    /// ############################################# #pragma omp parallel for
    for (int s = 0; s < prob.l; s++)
    {
        prob.y[s] = outputs[s];
        prob.x[s] = getFeatureVector(samples[s]);
    }

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

    #if SVM_WEIGHTS_MODE == 0
    param.nr_weight = 0;
    param.weight = nullptr;
    param.weight_label = nullptr;
    #else/*SVM_WEIGHTS_MODE*/
    param.nr_weight = 2;        // number of weights
    param.weight = new double[2] { positiveWeight, negativeWeight };    // class weights (positive, negative)
    param.weight_label = new int[2] { positiveOutput, negativeOutput }; // class labels    
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
        << "   W mode:      " << SVM_WEIGHTS_MODE << std::endl        
        #if SVM_WEIGHTS_MODE        
        << "   Wp:          " << param.weight[0] << std::endl
        << "   Wn:          " << param.weight[1] << std::endl
        << "   Wp lbl:      " << param.weight_label[0] << std::endl
        << "   Wn lbl:      " << param.weight_label[1] << std::endl;
        #endif/*SVM_WEIGHTS_MODE*/
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

double ESVM::predict(FeatureVector sample)
{    
    if (ensembleModel == nullptr)
        throw new std::exception("Ensemble model of Exemplar-SVM is not initialized");

    #if ENABLE_PREDICT_PROBABILITY
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
    #endif/*ENABLE_PREDICT_PROBABILITY*/
    
    // Obtain decision values directly (instead of predicted label/probability from 'svm_predict'/'svm_predict_probability')
    // Since the number of decision values of each class combination is calculated with [ nr_class*(nr_class-1)/2 ],
    // and that we have only 2 classes, we have only one decision value (positive vs. negative)
    int nClass = ensembleModel->nr_class;
    double* decisionValues = new double[nClass*(nClass - 1) / 2];
    svm_predict_values(ensembleModel, getFeatureVector(sample), decisionValues);
    return decisionValues[0];
}

std::vector<double> ESVM::predict(std::string filename, std::vector<int>& classGroundTruths)
{
    classGroundTruths = std::vector<int>();
    return std::vector<double>();
}

svm_node* ESVM::getFeatureVector(std::vector<double> features)
{
    return getFeatureVector(&features[0], features.size());
}

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
