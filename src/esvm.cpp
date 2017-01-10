#include "esvm.h"
#include <iterator>

ESVM::ESVM(std::vector< std::vector<double> > positives, std::vector< std::vector<double> > negatives)
{
    if (positives.size() <= 0 && negatives.size() <= 0)
        throw new std::exception("Exemplar-SVM cannot initialize without positive and negative feature vectors");

    int posSamples = positives.size();
    int negSamples = negatives.size();
    int allSamples = posSamples + negSamples;
        
    std::vector<double> outputs(allSamples, -1);
    for (int p = 0; p < posSamples; p++)
        outputs[p] = 1;

    std::vector< std::vector<double> > samples(allSamples);
    samples.insert(samples.end(), positives.begin(), positives.end());
    samples.insert(samples.end(), negatives.begin(), negatives.end());
        
    trainEnsembleModel(samples, outputs);    
}

void ESVM::trainEnsembleModel(std::vector< std::vector<double> > samples, std::vector<double> outputs)
{    
    svm_problem* prob;
    prob->l = samples.size();       // number of training data        
    prob->y = &outputs[0];          // target values for classification    
    
    // convert training vectors 
    svm_node** arrSamples = new svm_node*[prob->l];
    #pragma omp parallel for
    for (int s = 0; s < prob->l; s++)
        arrSamples[s] = getFeatureVector(samples[s]);
    prob->x = arrSamples;

    svm_parameter* param;
    param->probability = 1;
    param->C = 1;                   // cost constraint violation used for w*C
    param->p = 0.5;                 // sensitiveness of loss of support vector regression
    param->eps = 0.00001;           // stopping criterion
    param->nr_weight = 2;           // number of weights
    param->weight = new double[] { 100, 1 };    // class weights
    param->weight_label = new int[] { 1, -1 };  // class labels
    param->kernel_type = LINEAR;
    
    const char* msg = svm_check_parameter(prob, param);
    if (msg == NULL)
        throw new std::exception(msg);
    ensembleModel = svm_train(prob, param);
}

double ESVM::predict(std::vector<double> sample)
{    
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
    svm_node* fv = new svm_node[featureCount];
    #pragma omp parallel for
    for (int f = 0; f < featureCount; f++)
    {
        fv[f].index = f;
        fv[f].value = features[f];
    }
    return fv;
}
