#!/bin/bash
current=`pwd`
dir=/root/driver
logs=/dockerx

cd $dir/MLOpen/build_hip
export MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0
make clean
rm -rf *
CXX=/opt/rocm/hcc/bin/hcc cmake -DMIOPEN_TEST_ALL=ON -DMIOPEN_BACKEND=HIP -DMIOPEN_MAKE_BOOST_PUBLIC=ON -DMIOPEN_TEST_FLAGS="--disable-verification-cache" -DBoost_USE_STATIC_LIBS=Off -DCMAKE_PREFIX_PATH="/opt/rocm/hcc;/opt/rocm/hip" -DCMAKE_CXX_FLAGS="-isystem /usr/include/x86_64-linux-gnu/" .. | tee -a hipmlopen_build.log
make -j$(nproc) | tee -a hipmlopen_build.log
make check -j$(nproc) 2>&1| tee $logs/mlopen-ut.log
unset MIOPEN_CONV_PRECISE_ROCBLAS_TIMING


cd $dir/MLOpen/build_ocl
make clean
rm -rf *
cmake -DMIOPEN_TEST_ALL=ON -DMIOPEN_BACKEND=OpenCL -DMIOPEN_MAKE_BOOST_PUBLIC=ON -DBoost_USE_STATIC_LIBS=Off -DMIOPEN_TEST_FLAGS="--disable-verification-cache" -DOPENCL_INCLUDE_DIRS=/opt/rocm/opencl/include/ -DOPENCL_LIBRARIES=/opt/rocm/opencl/lib/x86_64/libamdocl64.so .. | tee -a mlopenocl_build.log
make -j$(nproc) | tee -a hipmlopen_build.log
make check -j$(nproc) 2>&1| tee $logs/mlopen-ut.log
