/* --------------------
Evaluation operations
---------------------*/
#include "eval.h"
#include "generic.h"
#include <algorithm>

// Counts the number of TP/TN/FP/FN according to specified scores, corresponding target class outputs and threshold value
// Target values are assumed to be greater than zero for positives and is negative class otherwise
void countConfusionMatrix(std::vector<double> scores, std::vector<int> targets, double threshold, int* TP, int* TN, int* FP, int* FN)
{
    ASSERT_LOG(FP != nullptr && FN != nullptr && TP != nullptr && TN != nullptr, "FP/FN/TP/TN have to be specified");
    ASSERT_LOG(scores.size() == targets.size(), "Number of scores and target classes must match");

    *FP = 0; *FN = 0; *TP = 0; *TN = 0;
    int nScore = scores.size();
    for (int i = 0; i < nScore; i++)
    {
        if      (scores[i] >= threshold && targets[i] > 0)  (*TP)++;
        else if (scores[i] >= threshold && targets[i] <= 0) (*FP)++;
        else if (scores[i] <  threshold && targets[i] > 0)  (*FN)++;
        else if (scores[i] <  threshold && targets[i] <= 0) (*TN)++;
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

// Accuracy
double calcACC(int Np, int Nn, double TPR, double FPR) { return (double)(Np * TPR + Nn * (1 - FPR)) / (double)(Np + Nn); }
double calcACC(int TP, int TN, int FP, int FN) { return (double)(TP + TN) / (double)(TP + TN + FP + FN); }

// Area or partial area under ROC curve (AUC or pAUC)
// Partial area is obtained from zero FPR up to pFPR
// The function assumes that (TPR,FPR) pairs are sorted in ascending order of FPR (ie: FPR from [0..p] or [0..1])
double calcAUC(std::vector<double> TPR, std::vector<double> FPR, double pFPR)
{
    ASSERT_LOG(TPR.size() == FPR.size(), "Number of TPR and FPR values must match");
    ASSERT_LOG(pFPR > 0 && pFPR <= 1, "Partial FPR value must be in ]0,1] interval");

    // find sorted value indexes in ascending order
    int nPoints = TPR.size() - 1;
    double pAUC = 0; 
    bool goNext = true;
    for (int n = 0; n < nPoints && goNext; n++)
    {
        double currFPR = FPR[n];
        double currTPR = TPR[n];
        double nextFPR = FPR[n + 1];
        double nextTPR = TPR[n + 1];

        // find intermediate point if partial threshold is reached (linear interpolation)
        if (nextFPR > pFPR)
        {
            nextTPR = currTPR + (pFPR - currFPR) * (nextTPR - currTPR) / (nextFPR - currFPR);
            nextFPR = pFPR;
            goNext = false;
        }

        // accumulate partial trapezoidal area
        pAUC += (nextTPR + currTPR) * std::abs(nextFPR - currFPR) / 2;        
    }
    return pAUC;
}
