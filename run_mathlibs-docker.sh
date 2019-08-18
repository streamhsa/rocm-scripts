#!/bin/bash
current=`pwd`
dir=/root/driver
logs=/dockerx

cd $dir/rocRAND/build
ctest --output-on-failure 2>&1 | tee $logs/rocrand-ut.log
./benchmark/benchmark_rocrand_kernel --engine all --dis all 2>&1 | tee $logs/bm_rocrand_kernal.log
./benchmark/benchmark_rocrand_generate --engine all --dis all 2>&1 | tee $logs/bm_rocrand_generate.log

cd $dir/rocPRIM/
#rm -rf build && ./install -ic 
cd build
ctest --output-on-failure 2>&1 | tee $logs/rocprim.log

cd $dir/rocThrust/
#rm -rf build && ./install -ic 
cd build
ctest --output-on-failure 2>&1 | tee $logs/rocthurst.log

cd $dir/hipCUB/
#rm -rf build && ./install -ic
cd build
ctest --output-on-failure 2>&1 | tee $logs/hipcub.log

cd $dir
./rocFFT/build/release/clients/staging/rocfft-test 2>&1 | tee $logs/rocfft.log
./rocALUTION/build/release/clients/staging/rocalution-test 2>&1 | tee $logs/rocalution.log
./hipBLAS/build/release/clients/staging/hipblas-test 2>&1 | tee $logs/hipblas.log
./rocSPARSE/build/release/clients/staging/rocsparse-test 2>&1 | tee $logs/rocsparse-test.log
./hipSPARSE/build/release/clients/staging/hipsparse-test 2>&1 | tee $logs/hipsparse-test.log
./rocBLAS/build/release/clients/staging/rocblas-test --gtest_filter=-*known_bug* 2>&1 | tee $logs/rocblas.log



