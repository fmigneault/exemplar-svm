#include "esvm.h"
#include <iterator>

/// ################################################ DEBUG
#include "helperFunctions.h"
/// ################################################ DEBUG

ESVM::ESVM(std::vector< FeatureVector > positives, std::vector< FeatureVector > negatives, std::string id)
{
    if (positives.size() <= 0 && negatives.size() <= 0)
        throw new std::exception("Exemplar-SVM cannot initialize without positive and negative feature vectors");
    
    targetID = id;
    int posSamples = positives.size();
    int negSamples = negatives.size();    
    int allSamples = posSamples + negSamples;    
    int posOutput = 1;
    int negOutput = 0;

    std::vector<int> outputs(posSamples + negSamples, negOutput);
    for (int s = 0; s < posSamples; s++)
        outputs[s] = posOutput;

    std::vector< FeatureVector > samples;
    samples.insert(samples.end(), positives.begin(), positives.end());
    samples.insert(samples.end(), negatives.begin(), negatives.end());
    
    // train with penalty weights
    // greater penalty attributed to incorrectly classifying a positive vs the many negatives
    double negWeight = (double)allSamples / (double)negSamples;
    double posWeight = (double)allSamples / (double)posSamples;

    /// ################################################ DEBUG
    logstream log(LOGGER_FILE);
    log << "ESVM initialization" << std::endl;
    log << "   posSamples: " << posSamples << std::endl;
    log << "   negSamples: " << negSamples << std::endl;
    log << "   posWeight:  " << posWeight << std::endl;
    log << "   negWeight:  " << negWeight << std::endl;
    log << "   posOutput:  " << posOutput << std::endl;
    log << "   negOutput:  " << negOutput << std::endl;
    log << "   samples | outputs:  " << std::endl;
    for (int s = 0; s < allSamples; s++)
    {
        std::string ss = "{";
        for (int f = 0; f < samples[s].size(); f++)
        {
            if (f!=0) ss += ",";
            ss += std::to_string(samples[s][f]);            
        }
        log << "      " << s << ": " << ss << "} | " << outputs[s] << std::endl;
    }
    /// ################################################ DEBUG

    trainEnsembleModel(samples, outputs, posOutput, negOutput, posWeight, negWeight);
}

void ESVM::trainEnsembleModel(std::vector< FeatureVector > samples, std::vector<int> outputs, 
                              int positiveOutput, int negativeOutput, double positiveWeight, double negativeWeight)
{    
    svm_problem prob;    
    prob.l = samples.size();    // number of training data        
    
    // convert and assign training vectors and corresponding target values for classification 
    prob.y = new double[prob.l];
    prob.x = new svm_node*[prob.l];
    #pragma omp parallel for
    for (int s = 0; s < prob.l; s++)
    {
        prob.y[s] = outputs[s];
        prob.x[s] = getFeatureVector(samples[s]);
    }

    /// ################################################ DEBUG
    /*logstream log(LOGGER_FILE);
    log << "ESVM training" << std::endl;
    for (int s = 0; s < samples.size(); s++)
    {
        std::string ss = "{";
        for (int f = 0; f < samples[s].size()+1; f++)
        {
            if (f != 0) ss += ",";
            ss += "(";
            ss += std::to_string(prob.x[s][f].index);
            ss += ": ";
            ss += std::to_string(prob.x[s][f].value);
            ss += ")";
        }
        log << "      " << s << ": " << ss << "} | " << prob.y[s] << std::endl;
    }*/
    /// ################################################ DEBUG

    // set training parameters    
    svm_parameter param;
    param.probability = 1;      // probability outputs instead of (+1,-1) classes
    param.C = 1;                // cost constraint violation used for w*C
    param.p = 0.1;              // sensitiveness of loss of support vector regression
    param.eps = 0.00001;        // stopping criterion
    param.nr_weight = 2;        // number of weights
    param.weight = new double[2] { positiveWeight, negativeWeight };    // class weights (positive, negative)
    param.weight_label = new int[2] { positiveOutput, negativeOutput }; // class labels
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
        double p = svm_predict_probability(ensembleModel, getFeatureVector(sample), probEstimates);

        /// ################################################ DEBUG
        /*logstream log(LOGGER_FILE);
        log << "ESVM predict" << std::endl;
        for (int s = 0; s < ensembleModel->nr_class; s++)
        {
            log << "probEstimates " << s << ": " << probEstimates[s] << std::endl;
        }*/
        /// ################################################ DEBUG

        return p;
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
        fv[f].index = f + 1;        // indexes should be one based
        fv[f].value = features[f];
    }
    fv[featureCount].index = -1;    // Additional feature value must be (-1,?) to end the vector (see LIBSVM README)
    return fv;
}
