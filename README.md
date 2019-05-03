# Hornet #

This repository provides the Hornet data structure and algorithms on sparse graphs and matrices.

## Getting Started ##

The document is organized as follows:

* [Requirements](#requirements)
* [Quick start](#quick-start)
* [Supported graph formats](#supported-graph-formats)
* [Code Documentation](#code-documentation)
* [Notes](#notes)
* [Reporting bugs and contributing](#reporting-bugs-and-contributing)
* [Publications](#publications)
* [Hornet Developers](#hornet-developers)
* [License](#license)

### Requirements ###

* [Nvidia Modern GPU](https://developer.nvidia.com/cuda-gpus) (compute capability &ge; 6.0): Pascal and Volta architectures.
* [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) 9 or greater.
* GCC or [Clang](https://clang.llvm.org) host compiler with support for C++14.
  Note, the compiler must be compatible with the related CUDA toolkit version.
  For more information see [CUDA Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
* [CMake](https://cmake.org) v3.8 or greater.
* 64-bit Operating System (Ubuntu 16.04 or above suggested).

### Quick start ###

The following basic steps are required to build and execute Hornet:
```bash
git clone --recursive https://github.com/hornet-gt/hornet
export CUDACXX=<path_to_CUDA_nvcc_compiler>
cd hornet/build
cmake ..
make -j
```

To build HornetsNest (algorithms directory) and Maximum Clique Algorithm:
```bash
cd hornetsnest/build
cmake ..
make -j
./corenum path-to-graph
```


By default, the CUDA compiler `nvcc` uses `gcc/g++` found in the current
execution search path as host compiler
(`cc --version` to get the default compiler on the actual system).
To force a different host compiler for compiling C++ files (`*.cpp`)
you need to set the following environment variables:
 ```bash
CC=<path_to_host_C_compiler>
CXX=<path_to_host_C++_compiler>
```

Note: host `.cpp` compiler and host side `.cu` compiler may be different.
The host side compiler must be compatible with the current CUDA Toolkit
version installed on the system
(see [CUDA Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)).

The syntax and the input parameters of Hornet are explained in detail in
 `docs/Syntax.txt`. They can also be found by typing `./HornetTest --help`.

