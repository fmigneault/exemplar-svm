#include "esvm.h"
#include <iterator>

ESVM::ESVM(std::vector< FeatureVector > positives, std::vector< FeatureVector > negatives, std::string id)
{
    if (positives.size() <= 0 && negatives.size() <= 0)
        throw new std::exception("Exemplar-SVM cannot initialize without positive and negative feature vectors");
    
    targetID = id;
    int posSamples = positives.size();
    int negSamples = negatives.size();    
        
    std::vector<int> outputs(posSamples + negSamples, 2);   // negatives as class 2
    for (int s = 0; s < posSamples; s++)
        outputs[s] = 1;                                     // positives as class 1

    std::vector< FeatureVector > samples;
    samples.insert(samples.end(), positives.begin(), positives.end());
    samples.insert(samples.end(), negatives.begin(), negatives.end());
    
    trainEnsembleModel(samples, outputs, posSamples, negSamples);
}

void ESVM::trainEnsembleModel(std::vector< FeatureVector > samples, std::vector<int> outputs, double positiveWeight, double negativeWeight)
{    
    svm_problem prob;    
    prob.l = samples.size();    // number of training data        
    
    // convert and assign target values for classification 
    prob.y = &std::vector<double>(outputs.begin(), outputs.end())[0];
    
    // convert and assign training vectors
    prob.x = new svm_node*[prob.l];
    int nPos = 0;
    int nNeg = 0;
    #pragma omp parallel for
    for (int s = 0; s < prob.l; s++)
        prob.x[s] = getFeatureVector(samples[s]);

    // set training parameters    
    svm_parameter param;
    param.probability = 0;      // probability outputs instead of (+1,-1) classes
    param.C = 1;                // cost constraint violation used for w*C
    param.p = 0.1;              // sensitiveness of loss of support vector regression
    param.eps = 0.00001;        // stopping criterion
    param.nr_weight = 2;        // number of weights
    param.weight = new double[2] { positiveWeight / prob.l, negativeWeight / prob.l };      // class weights (positive, negative)
    param.weight_label = new int[2] { 1, 2 };                                               // class labels
    param.kernel_type = LINEAR;
    param.cache_size = 10000;
    
    // validate parameters and train models
    const char* msg = svm_check_parameter(&prob, &param);
    if (msg == NULL)
        throw new std::exception(msg);
    ensembleModel = svm_train(&prob, &param);
}

double ESVM::predict(std::vector<double> sample)
{    
    if (ensembleModel == nullptr)
        throw new std::exception("Ensemble model of Exemplar-SVM is not initialized");

    if (ensembleModel->param.probability)
    {
        double* probEstimates = new double[ensembleModel->nr_class];
        return svm_predict_probability(ensembleModel, getFeatureVector(sample), probEstimates);
    }
    
    return svm_predict(ensembleModel, getFeatureVector(sample));
}

svm_node* ESVM::getFeatureVector(std::vector<double> features)
{
    return getFeatureVector(&features[0], features.size());
}

svm_node* ESVM::getFeatureVector(double* features, int featureCount)
{
    svm_node* fv = new svm_node[featureCount+1];
    #pragma omp parallel for
    for (int f = 0; f < featureCount; f++)
    {
        fv[f].index = f;
        fv[f].value = features[f];
    }
    fv[featureCount].index = -1;    // Additional feature value must be -1 to end the vector (see LIBSVM README)
    return fv;
}
