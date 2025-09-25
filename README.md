# MyCUDAPackage.jl

This repository is a template for a Julia package that ships with custom CUDA source code. The CUDA code is compiled into a shared library via a build script when the package is installed.

The package provides a single function, add_vectors!, which uses a custom CUDA kernel to add two vectors on the GPU.

### Prerequisites

Before installing, ensure the following are available on your system:

- An NVIDIA GPU with a compatible driver.
- A C++ compiler toolchain (g++).

### Installation

Open the Julia REPL, press ] to enter Pkg mode, and run:

```Julia
(v1.11) pkg> add https://github.com/oscarvanderheide/MyCUDAPackage.git
```

### Verification

To test the package and verify the custom kernel, run (while still in Pkg mode): 

```Julia
(v1.11) pkg> test MyCUDAPackage
```