

https://github.com/KhronosGroup/OpenCL-CLHPP
```
 git clone --recursive https://github.com/KhronosGroup/OpenCL-CLHPP
 git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader
 git clone https://github.com/KhronosGroup/OpenCL-Headers

 cmake -D CMAKE_INSTALL_PREFIX=./OpenCL-Headers/install -S ./OpenCL-Headers -B ./OpenCL-Headers/build 
 cmake --build ./OpenCL-Headers/build --target install

 cmake -D CMAKE_PREFIX_PATH=/absolute/path/to/OpenCL-Headers/install -D CMAKE_INSTALL_PREFIX=./OpenCL-ICD-Loader/install -S ./OpenCL-ICD-Loader -B ./OpenCL-ICD-Loader/build 
 cmake --build ./OpenCL-ICD-Loader/build --target install

 cmake -D CMAKE_PREFIX_PATH="/absolute/path/to/OpenCL-Headers/install;/absolute/path/to/OpenCL-ICD-Loader/install" -D CMAKE_INSTALL_PREFIX=./OpenCL-CLHPP/install -S ./OpenCL-CLHPP -B ./OpenCL-CLHPP/build 
 cmake --build ./OpenCL-CLHPP/build --target install

```


https://github.com/artyom-beilis/pytorch_dlprim
```
 git clone --recurse-submodules https://github.com/artyom-beilis/pytorch_dlprim.git

 cd pytorch_dlprim
 mkdir build
 cd build
 cmake -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DOCL_PATH="/Users/*/git/git/OpenCL-CLHPP/include/" -DCMAKE_PREFIX_PATH="/usr/local/anaconda3/lib/python3.11/site-packages/torch/share/cmake/Torch;/usr/local/anaconda3/lib/python3.11/site-packages/torch/share/cmake/ATen" -DCMAKE_CXX_STANDARD=17 ..
 make
 make install

 python mnist.py --device ocl:0
```
