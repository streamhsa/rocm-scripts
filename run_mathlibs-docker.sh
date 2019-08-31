#!/bin/bash
current=`pwd`
dir=/root/driver
logs=/dockerx

echo "==============================rocrand=============================="
cd $dir/rocRAND/build
ctest --output-on-failure 2>&1 | tee $logs/rocrand-ut.log
./benchmark/benchmark_rocrand_kernel --engine all --dis all 2>&1 | tee $logs/bm_rocrand_kernal.log
./benchmark/benchmark_rocrand_generate --engine all --dis all 2>&1 | tee $logs/bm_rocrand_generate.log

echo "==============================rocprim============================="
cd $dir/rocPRIM/
rm -rf build && mkdir build
cd build
CXX=/opt/rocm/bin/hcc cmake -DBUILD_TEST=ON -DDISABLE_WERROR=ON -DBUILD_BENCHMARK=OFF ../.
make -j$(nproc)
ctest --output-on-failure 2>&1 | tee $logs/rocprim.log

echo "==============================rocthrust=============================="
cd $dir/rocThrust/
rm -rf build && mkdir build
cd build
CXX=/opt/rocm/bin/hcc cmake -DBUILD_TEST=ON ../.
make -j$(nproc)
ctest --output-on-failure 2>&1 | tee $logs/rocthurst.log

echo "==============================hipcub=============================="
cd $dir/hipCUB/
rm -rf build && mkdir build
cd build
CXX=/opt/rocm/bin/hcc cmake -DBUILD_TEST=ON ../.
make -j$(nproc)
ctest --output-on-failure 2>&1 | tee $logs/hipcub.log



cd $dir
echo "==============================rocfft============================="
./rocFFT/build/release/clients/staging/rocfft-test 2>&1 | tee $logs/rocfft.log
echo "==============================rocalution============================="
./rocALUTION/build/release/clients/staging/rocalution-test 2>&1 | tee $logs/rocalution.log
echo "=============================hipblas============================="
./hipBLAS/build/release/clients/staging/hipblas-test 2>&1 | tee $logs/hipblas.log
echo "==============================rocsparse============================="
./rocSPARSE/build/release/clients/staging/rocsparse-test 2>&1 | tee $logs/rocsparse-test.log
echo "==============================hipsparse=============================="
./hipSPARSE/build/release/clients/staging/hipsparse-test 2>&1 | tee $logs/hipsparse-test.log
echo "==============================rocblas============================="
./rocBLAS/build/release/clients/staging/rocblas-test --gtest_filter=-*known_bug* 2>&1 | tee $logs/rocblas.log



