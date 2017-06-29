# Exemplar-SVM

---

## Description

Implementation of C++ Exemplar-SVM using [LIBSVM](https://github.com/cjlin1/libsvm) library.

### Additional details

* Alternatively, multiprocessing can be employed using [libsvm-openmp](https://github.com/KenjiKyo/libsvm/tree/v322-openmp-win64-bins) fork.

---

## Data

Original data is generated for testing purposes using [ChokePoint](http://arma.sourceforge.net/chokepoint/) dataset for video-based face recognition in the single sample per person problem.

If testing operations, methods and functions are to be run to validate operational modes, code or features, please download files from [ExemplarSVM-LIBSVM-Data](https://drive.google.com/drive/folders/0Bw9khIGD6JbbRzFfVDJ3cFNTM3c?usp=sharing) and place them under the root *ExemplarSVM-LIBSVM* repository directory.

If using only the `ESVM` and `ensembleESVM` classes, above data is not mandatory, but you will need to generate your own instead for other classification tasks.