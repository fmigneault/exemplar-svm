#include "ensembleEsvm.h"
#include "esvmOptions.h"
#include "generic.h"

#include <fstream>
#include <sstream>

/*
    Initializes an ESVM
*/
EnsembleESVM::EnsembleESVM()
{ 
    cv::Size patchCounts(3, 3);    
    size_t nPatches = patchCounts.area();
    for(int i = 0; i < patchCounts.area(); i++)
        ensembleEsvm.push_back(ESVM()); 
}

/*
    Predicts the classification value for the specified feature vector sample using the trained ESVM model.
*/
std::vector<double> EnsembleESVM::predict(const cv::Mat roi)
{
    FeatureVector feats;
    std::vector<double> predictions;
    for(int i = 0; i < 9; i++)
        predictions.push_back(ensembleEsvm[i].predict(feats));

    return predictions;
}
