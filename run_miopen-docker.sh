#!/bin/bash
current=`pwd`
dir=/root/driver
logs=/dockerx

cd $dir/MLOpen/build_hip
export MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0
make clean
rm -rf *
CXX=/opt/rocm/hcc/bin/hcc cmake -DMIOPEN_TEST_ALL=ON -DMIOPEN_BACKEND=HIP -DMIOPEN_MAKE_BOOST_PUBLIC=ON -DMIOPEN_TEST_FLAGS="--disable-verification-cache" -DBoost_USE_STATIC_LIBS=Off -DCMAKE_PREFIX_PATH="/opt/rocm/hcc;/opt/rocm/hip" -DCMAKE_CXX_FLAGS="-isystem /usr/include/x86_64-linux-gnu/" .. | tee -a hipmlopen_build.log
make -j$(nproc) | tee -a mlopenhip_build.log
make check -j$(nproc) 2>&1| tee $logs/mlopenhip-ut.log
unset MIOPEN_CONV_PRECISE_ROCBLAS_TIMING
make MIOpenDriver
./bin/MIOpenDriver conv -W 341 -H 79 -c 32 -n 4 -k 32 -y 5 -x 10 -p 0 -q 0 -u 2 -v 2 -t 1 -V 0 -i 1 2>&1 | tee -a MIOpenDriver_run.log
./bin/MIOpenDriver conv -n 1 -c 512 -H 256 -W 256 -k 6 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 2>&1 | tee -a MIOpenDriver_run.log
./bin/MIOpenDriver conv -n 1 -c 64 -H 130 -W 130 -k 3 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 2>&1 | tee -a MIOpenDriver_run.log


cd $dir/MLOpen/build_opencl
make clean
rm -rf *
cmake -DMIOPEN_TEST_ALL=ON -DMIOPEN_BACKEND=OpenCL -DMIOPEN_MAKE_BOOST_PUBLIC=ON -DBoost_USE_STATIC_LIBS=Off -DMIOPEN_TEST_FLAGS="--disable-verification-cache" -DOPENCL_INCLUDE_DIRS=/opt/rocm/opencl/include/ -DOPENCL_LIBRARIES=/opt/rocm/opencl/lib/x86_64/libamdocl64.so .. | tee -a mlopenocl_build.log
make -j$(nproc) | tee -a mlopenocl_build.log
make check -j$(nproc) 2>&1| tee $logs/mlopenocl-ut.log

make MIOpenDriver
./bin/MIOpenDriver conv -W 341 -H 79 -c 32 -n 4 -k 32 -y 5 -x 10 -p 0 -q 0 -u 2 -v 2 -t 1 -V 0 -i 1 2>&1 | tee -a MIOpenDriver_run.log
./bin/MIOpenDriver conv -n 1 -c 512 -H 256 -W 256 -k 6 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 2>&1 | tee -a MIOpenDriver_run.log
./bin/MIOpenDriver conv -n 1 -c 64 -H 130 -W 130 -k 3 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 2>&1 | tee -a MIOpenDriver_run.log

