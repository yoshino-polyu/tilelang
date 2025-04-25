cd ~/tilelang
mkdir build
cp 3rdparty/tvm/cmake/config.cmake build
cd build
echo "set(USE_LLVM /home/v-xle/clang+llvm-17.0.4-x86_64-linux-gnu-ubuntu-22.04/bin/llvm-config)" >> config.cmake
echo "set(USE_CUDA /usr/local/cuda)" >> config.cmake
echo "set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)" >> config.cmake
echo "set(zstd_INCLUDE_DIR /usr/include)" >> config.cmake
echo "set(zstd_LIBRARY /usr/lib/x86_64-linux-gnu/libzstd.so)" >> config.cmake
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

cmake ..
make -j 8
export PYTHONPATH=/home/v-xle/tilelang/:$PYTHONPATH
