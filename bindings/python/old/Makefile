# Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Flags for directing the runtime makefile what to include
DEBUG           ?= 0		# Include debugging symbols
MAX_DIM         ?= 5		# Maximum number of dimensions
OUTPUT_LEVEL    ?= LEVEL_DEBUG	# Compile time logging level
USE_CUDA        ?= 1		# Include CUDA support (requires CUDA)
USE_GASNET      ?= 0		# Include GASNet support (requires GASNet)
USE_HDF         ?= 0		# Include HDF5 support (requires HDF5)
ALT_MAPPERS     ?= 0		# Include alternative mappers (not recommended)
USE_HIP         ?= 0		# Include HIP support (requires HIP)
HIP_TARGET      ?= ROCM
GPU_ARCH        ?= 60 61 62 70 72 75 80 86
USE_GPU_REDUCTIONS ?= 0
FF_USE_NCCL  ?= 0
# FF_PYTHON_USE_INDEX_LOADER = 1

ifeq ($(strip $(DARWIN)),1)
  PYTHON_EXT := dylib
else
  PYTHON_EXT := so
endif

USE_PYTHON := 1
ifndef PYTHON_LIB
  ifndef PYTHON_ROOT
    PYTHON_EXE := $(shell which python3 | head -1)
    ifeq ($(PYTHON_EXE),)
      $(error cannot find python - set PYTHON_ROOT if not in PATH)
    endif
    PYTHON_VERSION_MAJOR := $(shell $(PYTHON_EXE) -c 'import sys; print(sys.version_info.major)')
    PYTHON_VERSION_MINOR := $(shell $(PYTHON_EXE) -c 'import sys; print(sys.version_info.minor)')
    PYTHON_ROOT := $(dir $(PYTHON_EXE))
  endif

  # Try searching for common locations of the Python shared library.
  ifneq ($(strip $(PYTHON_ROOT)),)
    PYTHON_LIB := $(wildcard $(PYTHON_ROOT)/libpython$(PYTHON_VERSION_MAJOR).$(PYTHON_VERSION_MINOR)*.$(PYTHON_EXT))
    ifeq ($(strip $(PYTHON_LIB)),)
      PYTHON_LIB := $(wildcard $(abspath $(PYTHON_ROOT)/../lib/libpython$(PYTHON_VERSION_MAJOR).$(PYTHON_VERSION_MINOR)*.$(PYTHON_EXT)))
      ifeq ($(strip $(PYTHON_LIB)),)
        PYTHON_LIB := $(wildcard $(abspath $(PYTHON_ROOT)/../lib64/libpython$(PYTHON_VERSION_MAJOR).$(PYTHON_VERSION_MINOR)*.$(PYTHON_EXT)))
        ifeq ($(strip $(PYTHON_LIB)),)
          $(warning cannot find libpython$(PYTHON_VERSION_MAJOR).$(PYTHON_VERSION_MINOR)*.$(PYTHON_EXT) - falling back to using LD_LIBRARY_PATH)
          PYTHON_LIB :=
        endif
      endif
    endif
  endif
endif

$(info    VAR is $(PYTHON_EXE))

# Put the binary file name here
OUTFILE		  ?= flexflow_python

GEN_SRC		  ?= main.cc flexflow_c.cc flexflow_dataloader.cc # .cc files
GEN_GPU_SRC	  ?= flexflow_dataloader.cu # .cu files
ifeq ($(strip $(HIP_TARGET)),CUDA)
GEN_HIP_SRC	  ?= flexflow_dataloader.cu # .cu files
else
GEN_HIP_SRC	  ?= flexflow_dataloader.cpp # .cpp files
endif

FF_NATIVE_LIB = flexflow/core/libflexflow_native_python.$(PYTHON_EXT)
FF_NATIVE_LIB_NAME = libflexflow_native_python.$(PYTHON_EXT)
INC_FLAGS     += -I${FF_HOME}
CC_FLAGS      += -fPIC
NVCC_FLAGS    += -Xcompiler -fPIC
ifeq ($(strip $(USE_HIP)),1)
ifeq ($(strip $(HIP_TARGET)),CUDA)
HIPCC_FLAGS    += -Xcompiler -fPIC
else
HIPCC_FLAGS    += -fPIC
endif
endif

CC_FLAGS	  ?= -DBINDINGS_AUGMENT_PYTHONPATH -DFF_USE_PYTHON 

#INC_FLAGS       += -DDISABLE_LEGION_CUDA_HIJACK 
  
ifeq ($(strip $(FF_ENABLE_DEBUG)), 1)
CC_FLAGS   += -DFF_DEBUG
NVCC_FLAGS += -DFF_DEBUG
HIPCC_FLAGS += -DFF_DEBUG
endif

ifeq ($(strip $(FF_PYTHON_USE_INDEX_LOADER)), 1)
CC_FLAGS   += -DFF_PYTHON_USE_INDEX_LOADER
NVCC_FLAGS += -DFF_PYTHON_USE_INDEX_LOADER
HIPCC_FLAGS += -DFF_PYTHON_USE_INDEX_LOADER
endif

ifeq ($(shell uname), Darwin)
	LD_FLAGS += -Wl,-force_load,liblegion.a
else
	LD_FLAGS += -Wl,--whole-archive -llegion -Wl,--no-whole-archive
endif

FF_USE_PYTHON = $(USE_PYTHON)

FF_HOME ?= ../
LG_RT_DIR ?= $(FF_HOME)/deps/legion/runtime

include $(FF_HOME)/FlexFlow.mk

NO_BUILD_ALL=1
.PHONY: all
all: $(OUTFILE) flexflow/core/legion_cffi_header.py flexflow/core/flexflow_cffi_header.py $(FF_NATIVE_LIB)

DEFINE_HEADERS_DIR ?= ./

flexflow/core/legion_cffi_header.py: legion_cffi_header.py.in legion_cffi_build.py $(LG_RT_DIR)/legion.h
	$(PYTHON_EXE) legion_cffi_build.py --runtime-dir $(LG_RT_DIR) --defines-dir $(DEFINE_HEADERS_DIR) --output-dir flexflow/core
	
flexflow/core/flexflow_cffi_header.py: flexflow_cffi_header.py.in flexflow_cffi_build.py flexflow_c.h
	$(PYTHON_EXE) flexflow_cffi_build.py --ffhome-dir $(FF_HOME) --libname $(FF_NATIVE_LIB_NAME) --output-dir flexflow/core

# build the .so for native python
# hack: we do not need main.cc anymore
NATIVE_SHARED_OBJS := $(filter-out main.cc.o,$(APP_OBJS))
$(FF_NATIVE_LIB): $(NATIVE_SHARED_OBJS) $(SLIB_LEGION) $(SLIB_REALM)
	$(CXX) $(CC_FLAGS) $(INC_FLAGS) -shared -lstdc++ -L$(FF_HOME)/python $(NATIVE_SHARED_OBJS) $(LD_FLAGS) $(LEGION_LIBS) $(LEGION_LD_FLAGS) -o $@

clean::
	$(RM) -f $(FF_NATIVE_LIB) legion_defines.h realm_defines.h flexflow/core/legion_cffi_header.py flexflow/core/flexflow_cffi_header.py *.pyc
	$(RM) -rf build dist *.egg-info
