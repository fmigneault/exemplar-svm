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
int create_negatives();
int create_probes(std::string positives, std::string negatives);

#endif/*CREATE_NEGATIVES_H*/
