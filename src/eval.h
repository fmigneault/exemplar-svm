/* --------------------
Evaluation operations
---------------------*/
#ifndef EVAL_H
#define EVAL_H

#include <vector>

// Counts the number of TP/TN/FP/FN according to specified scores, corresponding target class outputs and threshold value
// Target values are assumed to be greater than zero for positives and is negative class otherwise
void countConfusionMatrix(std::vector<double> scores, std::vector<int> targets, double threshold, int* TP, int* TN, int* FP, int* FN);

// Positive Predictive Value (Precision)
double calcPPV(int TP, int FP);

// True Positive Rate (Recall - Sensitivity)
double calcTPR(int TP, int FN);

// True Negative Rate (Specificity)
double calcTNR(int TN, int FP);
double calcSPC(int TN, int FP);

// False Positive Rate 
double calcFPR(int FP, int TN);

// Accuracy
double calcACC(int Np, int Nn, double TPR, double FPR);
double calcACC(int TP, int TN, int FP, int FN);

// Area or partial area under ROC curve (AUC or pAUC)
// Partial area is obtained from zero FPR up to pFPR
// The function assumes that (TPR,FPR) pairs are sorted in ascending order of FPR (ie: FPR from [0..p] or [0..1])
double calcAUC(std::vector<double> TPR, std::vector<double> FPR, double pFPR = 1.0);

#endif/*EVAL_H*/