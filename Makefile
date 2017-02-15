TARGET	= FAST-DT.exe

CC = g++
CFLAGS = -c -Wall -O3 -fopenmp -g -std=c++11
LDFLAGS = 

# local headers folder
LOCAL_DIR := "./src" 
LIB_SVM := "./LIBSVM"
HOG_FEAT := "./FeatureExtractorHOG/src"
HOG_FEAT := "./FeatureExtractorHOG/src"
LBP_FEAT1 := "./FeatureExtractorLBP/src"
LBP_FEAT2 := "./FeatureExtractorLBP"
M_VEC := "./mvector"

INCLUDE_DIRS =	-I$(LIB_SVM) \
				-I$(OPENCV_DIR)/include/ -I$(OPENCV_DIR)/include/opencv -I$(OPENCV_DIR)/include/opencv2 \
				-I$(LOCAL_DIR) -I$(HOG_FEAT) -I$(LBP_FEAT1) -I$(LBP_FEAT2) -I$(M_VEC)

# OpenCV dynamic libraries installation folder
LIB_DIRS = -L$(OPENCV_DIR)/lib

LIBS =	-lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_video -lgomp -lboost_filesystem -lboost_system -lopencv_videoio -lopencv_imgcodecs -lfftw3

CPP_SRCS += \
./LIBSVM/svm.cpp \
./FeatureExtractorHOG/src/feHOG.cpp \
./FeatureExtractorLBP/src/feLBP.cpp \
./FeatureExtractorLBP/LBP/LBP.cpp \
./src/esvm.cpp \
./src/esvmTests.cpp \
./src/eval.cpp \
./src/imgUtils.cpp \
./src/logger.cpp \
./src/norm.cpp \
./src/main.cpp 


CPP_OBJECTS = $(CPP_SRCS:%.cpp=%.o)
CPP_DEPS = $(CPP_OBJECTS:.o=.d)

.PHONY: all clean

all: $(TARGET)

-include $(CPP_DEPS)
	
$(TARGET): $(CPP_OBJECTS)
	$(CC) $(LDFLAGS) $(LIB_DIRS) -o $@ $(CPP_OBJECTS) $(C_OBJECTS) $(LIBS)

%.o : %.cpp
	$(CC) $(CFLAGS) -MM -MF $(patsubst %.o,%.d,$@) $(INCLUDE_DIRS) $(INCLUDES) $<
	$(CC) $(CFLAGS) $(INCLUDE_DIRS) $(INCLUDES) $< -o $@

clean:
	rm -f $(CPP_OBJECTS)
	rm -f $(CPP_DEPS)
	rm -f $(TARGET)

