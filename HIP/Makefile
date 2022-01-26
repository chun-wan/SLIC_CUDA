
SRC = $(wildcard *.cpp)
OBJ = $(SRC:.cpp=.o)
EXE = SLIC_HIP

ROCM_PATH ?= /opt/rocm

OPENCV_INC = /usr/local/include/opencv4/
HIPCC      = ${ROCM_PATH}/bin/hipcc
HIP_INC  = -I${ROCM_PATH}/include 
HIP_INC += -I${OPENCV_INC}


OPENCV_LIB = -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio
HIP_LIB  = -L${ROCM_PATH}/lib
HIP_LIB  += ${OPENCV_LIB}

CXXFLAGS       = -O3 -std=c++11
AMDGPU_TARGET  = $(shell ${ROCM_PATH}/bin/rocminfo | grep -m 1 -o -P 'gfx.{0,4}')
CXXFLAGS      += --amdgpu-target=${AMDGPU_TARGET}


INC = $(HIP_INC)
LIB = $(HIP_LIB) $(ROC_LIBS)

rocm: $(OBJ)
	$(HIPCC) $(CXXFLAGS) $(INC) $(OBJ) $(LIB) -o $(EXE)

.cpp.o:
	$(HIPCC) $(CXXFLAGS) $(INC) -c $< -o $@


clean:
	rm -rf $(OBJ) $(EXE) core.*
