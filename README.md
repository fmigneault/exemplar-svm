# Exemplar-SVM

---

## Description

Implementation of C++ Exemplar-SVM using either [LIBSVM](https://github.com/cjlin1/libsvm) or [LIBLINEAR](https://github.com/cjlin1/liblinear) library.

### Additional details

Alternative libsvm implementations can be employed to include some variations or improvements: 

* Parallel multiprocessing with libsvm-openmp forks [fmigneault/libsvm](https://github.com/fmigneault/libsvm/tree/v322-openmp-win64-bins) or [TeamLIVIA/libsvm](https://bitbucket.org/TeamLIVIA/libsvm/branch/v322-openmp-win64-bins). 
* Planned integration of SVM incremental learning (via [liblinear-incdec](https://www.csie.ntu.edu.tw/~cjlin/papers/ws/) extension)  
* Planned integration of SVM multicore training (via [liblinear-multicore](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/multicore-liblinear/) extension | fork [TeamLIVIA/LIBLINEAR-multicore](https://bitbucket.org/TeamLIVIA/liblinear-multicore))

---

## Data

Original data is generated for testing purposes using [ChokePoint](http://arma.sourceforge.net/chokepoint/) dataset for video-based face recognition in the single sample per person problem.

If testing operations, methods and functions are to be run to validate operational modes, code or features, please download files from [ExemplarSVM-LIBSVM-Data](https://drive.google.com/drive/folders/0Bw9khIGD6JbbRzFfVDJ3cFNTM3c?usp=sharing) and place them under the root *ExemplarSVM-LIBSVM* repository directory.

If using only the `ESVM` and `ensembleESVM` classes, above data is not mandatory, but you will need to generate your own instead for other classification tasks.
