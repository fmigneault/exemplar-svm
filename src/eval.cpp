/* --------------------
Evaluation operations
---------------------*/
#include "eval.h"
#include "generic.h"
#include <algorithm>

// Counts the number of FP/FN/TP/TN according to specified scores, corresponding target class outputs and threshold value
// Target values are assumed to be greater than zero for positives and is negative class otherwise
void countThresholdScores(std::vector<double> scores, std::vector<int> targets, double threshold, int* FP, int* FN, int* TP, int* TN)
{
    ASSERT_LOG(FP != nullptr && FN != nullptr && TP != nullptr && TN != nullptr, "FP/FN/TP/TN have to be specified");
    ASSERT_LOG(scores.size() == targets.size(), "Number of scores and target classes must match");

    *FP = 0; *FN = 0; *TP = 0; *TN = 0;
    int nScore = scores.size();
    for (int i = 0; i < nScore; i++)
    {
        if      (scores[i] >= threshold && targets[i] > 0)  *TP++;
        else if (scores[i] >= threshold && targets[i] <= 0) *FP++;
        else if (scores[i] <  threshold && targets[i] > 0)  *FN++;
        else if (scores[i] <  threshold && targets[i] <= 0) *TN++;
    }
}

// Positive Predictive Value (Precision)
double calcPPV(int TP, int FP) { return (double)TP / (double)(TP + FP); }

// True Positive Rate (Recall - Sensitivity)
double calcTPR(int TP, int FN) { return (double)TP / (double)(TP + FN); }

// True Negative Rate (Specificity)
double calcTNR(int TN, int FP) { return (double)TN / (double)(TN + FP); }
double calcSPC(int TN, int FP) { return calcTNR(TN, FP); }

// False Positive Rate 
double calcFPR(int FP, int TN) { return (double)FP / (double)(FP + TN); }

// Weighted accuracy
double calcACC(int Np, int Nn, double TPR, double FPR) { return Np * TPR + Nn * (1 - FPR); }

// Area or partial area under ROC curve (AUC or pAUC)
double calcAUC(std::vector<double> TPR, std::vector<double> FPR, double pFPR)
{
    ASSERT_LOG(TPR.size() == FPR.size(), "Number of TPR and FPR values must match");
    ASSERT_LOG(pFPR > 0 && pFPR <= 1, "Partial FPR value must be in ]0,1] interval");

    // find sorted value indexes in ascending order
    int nPoints = TPR.size();
    std::vector<int> sortedIndexes = std::vector<int>(nPoints);
    std::generate(sortedIndexes.begin(), sortedIndexes.end(), SequenceGen(0));      // Indexes { 0, 1, ..., N-1 }
    std::sort(sortedIndexes.begin(), sortedIndexes.end(), Compare<double>(FPR));    // Indexes sorted according to FPR
    
    double pAUC = 0; 
    bool goNext = true;
    for (std::vector<int>::iterator idx = sortedIndexes.begin(); idx != sortedIndexes.end() - 1 && goNext; ++idx)
    {
        double currFPR = FPR[*idx];
        double currTPR = TPR[*idx];
        double nextFPR = FPR[*(idx + 1)];
        double nextTPR = TPR[*(idx + 1)];

        // find intermediate point if partial threshold is reached (linear interpolation)
        if (nextFPR > pFPR)
        {
            nextTPR = currTPR + (pFPR - currFPR) * (nextTPR - currTPR) / (nextFPR - currFPR);
            nextFPR = pFPR;
            goNext = false;
        }

        // accumulate partial trapezoidal area
        pAUC += (nextTPR + currTPR) * (nextFPR - currFPR) / 2;        
    }
    return pAUC;
}
