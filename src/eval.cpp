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
    size_t nScore = scores.size();
    for (size_t i = 0; i < nScore; i++)
    {
        if      (scores[i] >= threshold && targets[i] > 0)  (*TP)++;
        else if (scores[i] >= threshold && targets[i] <= 0) (*FP)++;
        else if (scores[i] <  threshold && targets[i] > 0)  (*FN)++;
        else if (scores[i] <  threshold && targets[i] <= 0) (*TN)++;
    }
}

void countConfusionMatrix(std::vector<double> scores, std::vector<int> targets, double threshold, ConfusionMatrix* cm)
{
    ASSERT_LOG(cm != nullptr, "Confusion matrix has to be specified");
    countConfusionMatrix(scores, targets, threshold, &(cm->TP), &(cm->TN), &(cm->FP), &(cm->FN));
}

// Positive Predictive Value (Precision)
double calcPPV(int TP, int FP) { return (double)TP / (double)(TP + FP); }
double calcPPV(ConfusionMatrix cm) { return calcPPV(cm.TP, cm.FP); }

// True Positive Rate (Recall - Sensitivity)
double calcTPR(int TP, int FN) { return (double)TP / (double)(TP + FN); }
double calcTPR(ConfusionMatrix cm) { return calcTPR(cm.TP, cm.FN); }

// True Negative Rate (Specificity)
double calcTNR(int TN, int FP) { return (double)TN / (double)(TN + FP); }
double calcSPC(int TN, int FP) { return calcTNR(TN, FP); }
double calcTNR(ConfusionMatrix cm) { return calcTNR(cm.TN, cm.FP); }
double calcSPC(ConfusionMatrix cm) { return calcSPC(cm.TN, cm.FP); }

// False Positive Rate 
double calcFPR(int FP, int TN) { return (double)FP / (double)(FP + TN); }
double calcFPR(ConfusionMatrix cm) { return calcFPR(cm.FP, cm.TN); }

// Accuracy
double calcACC(int Np, int Nn, double FPR, double TPR) { return (double)(Np * TPR + Nn * (1 - FPR)) / (double)(Np + Nn); }
double calcACC(int TP, int TN, int FP, int FN) { return (double)(TP + TN) / (double)(TP + TN + FP + FN); }
double calcACC(ConfusionMatrix cm) { return calcACC(cm.TP, cm.TN, cm.FP, cm.FN); }

// Area or partial area under ROC curve (AUC or pAUC)
// Partial area is obtained from zero FPR up to pFPR
// The function assumes that (FPR,TPR) pairs are sorted in ascending order of FPR (ie: FPR from [0..p] or [0..1])
double calcAUC(std::vector<double> FPR, std::vector<double> TPR, double pFPR)
{
    ASSERT_LOG(TPR.size() == FPR.size(), "Number of TPR and FPR values must match");
    ASSERT_LOG(pFPR > 0 && pFPR <= 1, "Partial FPR value must be in ]0,1] interval");

    // find sorted value indexes in ascending order
    size_t nPoints = TPR.size() - 1;
    double pAUC = 0; 
    bool goNext = true;
    for (size_t n = 0; n < nPoints && goNext; n++)
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

double calcAUC(std::vector<ConfusionMatrix> cm, double pFPR)
{
    size_t nPoints = cm.size();
    std::vector<double> FPR(nPoints), TPR(nPoints);
    for (size_t i = 0; i < nPoints; i++)
    {
        FPR[i] = calcFPR(cm[i]);
        TPR[i] = calcTPR(cm[i]);
    }
    return calcAUC(FPR, TPR, pFPR);
}

// Area under Precision-Recall curve (AUPR)
// The function assumes that (TPR,PPV) pairs are sorted in ascending order of TPR (ie: TPR from [0..1])
double calcAUPR(std::vector<double> TPR, std::vector<double> PPV)
{
    /* filter NaN values
           these are possible and valid when the threshold is greater than the maximum obtained score, which makes PPV = 0/0
           because all are evaluated as negatives (FP or TP ) by definition and [PPV = TP/(TP+FP)] is based only on positives 

           since TPR is expected in ascending order, the corresponding PPV NaN values should be at the start of the vector
    */
    // find last NaN position
    size_t nPoints = PPV.size();
    size_t nanPos = 0;
    for (size_t i = 0; i < nPoints; i++)
    {
        if (!std::isnan(PPV[i])) break;
        nanPos++;
    }
    // remove the amount of NaN found
    TPR = std::vector<double>(TPR.begin() + nanPos, TPR.end());
    PPV = std::vector<double>(PPV.begin() + nanPos, PPV.end());

    return calcAUC(TPR, PPV);   // employ the AUC functionality for similar calculation procedure
}

double calcAUPR(std::vector<ConfusionMatrix> cm)
{
    size_t nPoints = cm.size();
    std::vector<double> PPV(nPoints), TPR(nPoints);
    for (size_t i = 0; i < nPoints; i++)
    {
        PPV[i] = calcPPV(cm[i]);
        TPR[i] = calcTPR(cm[i]);
    }
    return calcAUPR(TPR, PPV);
}
