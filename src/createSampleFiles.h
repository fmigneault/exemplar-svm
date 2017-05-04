#ifndef CREATE_NEGATIVES_H
#define CREATE_NEGATIVES_H

#include <string>
#include <stdlib.h>
#include "esvmTests.h"
#include "esvm.h"
#include "norm.h"
#include <stdio.h>
#include "opencv2/opencv.hpp"
#include "mvector.hpp"
#include "generic.h"
#include "imgUtils.h"
#include "feHOG.h"
#include <iostream>
#include <fstream>
#include "esvmOptions.h"

/* Processes */
int proc_generateConvertedImageTypes();
int proc_createNegativesSampleFiles();
int proc_createProbesSampleFiles(std::string positivesImageDirPath, std::string negativesImageDirPath);

/* Utilities */
xstd::mvector<2, cv::Mat> loadAndProcessImages(std::string dirPath, std::string imageExtension);

#endif/*CREATE_NEGATIVES_H*/
