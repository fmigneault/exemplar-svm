/* --------------------
Evaluation operations
---------------------*/
#ifndef EVAL_H
#define EVAL_H

#include <vector>

class ConfusionMatrix
{
public:
    int TP = 0; int TN = 0;
    int FP = 0; int FN = 0;
    inline ConfusionMatrix() {};
    inline ConfusionMatrix(int TP, int TN, int FP, int FN) { this->TP = TP; this->TN = TN; this->FP = FP; this->FN = FN; }
};

// Counts the number of TP/TN/FP/FN according to specified scores, corresponding target class outputs and threshold value
// Target values are assumed to be greater than zero for positives and is negative class otherwise
void countConfusionMatrix(std::vector<double> scores, std::vector<int> targets, double threshold, int* TP, int* TN, int* FP, int* FN);
void countConfusionMatrix(std::vector<double> scores, std::vector<int> targets, double threshold, ConfusionMatrix* cm);

// Positive Predictive Value (Precision)
double calcPPV(int TP, int FP);
double calcPPV(ConfusionMatrix cm);

// True Positive Rate (Recall - Sensitivity)
double calcTPR(int TP, int FN);
double calcTPR(ConfusionMatrix cm);

// True Negative Rate (Specificity)
double calcTNR(int TN, int FP);
double calcSPC(int TN, int FP);
double calcTNR(ConfusionMatrix cm);
double calcSPC(ConfusionMatrix cm);

// False Positive Rate 
double calcFPR(int FP, int TN);
double calcFPR(ConfusionMatrix cm);

// Accuracy
double calcACC(int Np, int Nn, double FPR, double TPR);
double calcACC(int TP, int TN, int FP, int FN);
double calcACC(ConfusionMatrix cm);

// Area or partial area under ROC curve (AUC or pAUC)
// Partial area is obtained from zero FPR up to pFPR (defaults to 1.0 for AUC)
// The result is divided by the corresponding maximum possible area (full or partial) to obtain a percentage in [0,1] range
// The function assumes that (FPR,TPR) pairs are sorted in ascending order of FPR (ie: FPR from [0..pFPR] or [0..1])
double calcAUC(std::vector<double> FPR, std::vector<double> TPR, double pFPR = 1.0);
double calcAUC(std::vector<ConfusionMatrix> cm, double pFPR = 1.0);

// Area under Precision-Recall curve (AUPR)
// The function assumes that (TPR,PPV) pairs are sorted in ascending order of TPR (ie: TPR from [0..1])
double calcAUPR(std::vector<double> TPR, std::vector<double> PPV);
double calcAUPR(std::vector<ConfusionMatrix> cm);

#endif/*EVAL_H*/