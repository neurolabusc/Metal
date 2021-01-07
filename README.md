## About

Apple's [Metal](https://developer.apple.com/metal/) framework is useful for both graphics as well as general computation. Metal compute shaders can in theory compete with [NVidia CUDA](https://developer.nvidia.com/CUDA-zone). While Metal does not support double-precision floating point data, this data type has poor performance on NVidia's commodity graphics cards and is not required for many scientific tasks. For example, the neuroimaging tools Eddy, ProbtrackX and BedpostX all use CUDA's single-precision data types. 

##### Minimal Example

This program updates [Matthew Hamilton's example](https://gist.github.com/mhamilt/a5c2bbb02684e5db362712c9be7a02ca). The `add.metal` kernel simply returns the sum of an array. You should first compile the kernel using xcrun and then include the compiled library into the executable:

```
xcrun -sdk macosx metal -c add.metal -o add.air
xcrun -sdk macosx metallib add.air -o default.metallib
clang -framework Foundation -framework metal main.m -o hello  -Wl,-sectcreate,addseg,addsect,default.metallib
./hello
```

- Graphical [Metal demos](https://github.com/neurolabusc/Metal-Demos).

##### Matrix Multiplication Example

Matrix multiplication is a core function for machine learning, spatial processing and statistics. [Once upon a time](http://machinethink.net/blog/mps-matrix-multiplication/), the  Metal Performance Shaders [MPSMatrixMultiplication()](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication) function offered the fastest alternative for macOS computers. However, users report severe limitations, for example matrix rows/columns must be evenly divisible by 8, no support for double-precision, and limited matrix sizes. However, the new Apple Silicon computers include a dedicated Apple Matrix (AMX) co-processor that does not share these limitations. Beyond being more flexible, the provided Swift script illustrates that AMX outperforms Metal. The test also demonstrates that on the M1 computers, MPSMatrixMultiplication fails when any row or column exceeds a size of 65535, whereas the AMX [cblas_sgemm](https://developer.apple.com/documentation/accelerate/1513264-cblas_sgemm?language=objc) can handle any size. Therefore, while in theory the unified memory of these new computers aids mixing and matching Metal and CPU computations, usage of the MPSMatrixMultiplication function should be discouraged with this hardware.

```
swiftc -framework Foundation -framework metal mmul.swift -o mmul; 
./mmul
BLAS Elapsed time min:  1.7490387  mean:  1.7490387
METAL Elapsed time min:  8.759975  mean:  8.759975
A values: [0.39646477...0.4103283]
B values: [0.7618172...0.48532528]
BLAS results: [31.582188...35.474823]
Metal results: [31.582188...35.474823]
Largest error: 0.0
```

 - [Example noting limitations: rows/columns must be divisible by 8, care with managed mode](https://developer.apple.com/forums/thread/105534).
 - [Description of Metal Matrix multiplication](http://machinethink.net/blog/mps-matrix-multiplication/) with [Code at Github](https://github.com/hollance/MPS-Matrix-Multiplication). This suggests MTLBuffer can hold at most 256MB of data. Mentions that code must be [aligned to page size](https://stackoverflow.com/questions/27365905/pointer-memory-alignment-to-16k-in-swift-for-metal-buffer-creation/27373987). Suggests StorageModeShared may be a good choice for unified memory.
 



