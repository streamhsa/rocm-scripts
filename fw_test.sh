#!/bin/bash
current=`pwd`
dir=/root
mkdir -p /root/logs
logs=/root/logs

rm -rf /root/logs/*
echo "=======================Firmware Smoke Test Started on ROCm stack==================="
echo "======================running clinfo==============================" 2>&1 | tee $logs/clinfo.log
/opt/rocm/opencl/bin/x86_64/clinfo 2>&1 | tee -a $logs/clinfo.log
echo "======================running rocminfo=============================" 2>&1 | tee $logs/rocminfo.log
/opt/rocm/bin/rocminfo 2>&1 | tee -a $logs/rocminfo.log

echo "======================running rocsmi=============================" 2>&1 | tee $logs/rocmsmi.log
/opt/rocm/bin/rocm_smi.py 2>&1 | tee -a $logs/rocmsmi.log

echo "======================running hipinfo=============================" 2>&1 | tee $logs/hipinfo.log
cd /opt/rocm/hip/samples/1_Utils/hipInfo
make
./hipInfo 2>&1 | tee -a $logs/hipinfo.log

echo "======================running rocm-bandwidth validation=================" 2>&1 | tee $logs/rocm-bw.log
/opt/rocm/bin/rocm_bandwidth_test 

cd $dir/driver/rocblas/build/release/clients/staging

echo "===================== Running rocBLAS : GEMM - Single Precision =================================" 2>&1 | tee $logs/rocblas_gemm_single.log

./rocblas-bench -f gemm -r s -m 2047 -n 2047 -k 2047 --lda 2047 --ldb 2047 --ldc 2047 --transposeB T 2>&1 | tee -a $logs/rocblas_gemm_single.log
./rocblas-bench -f gemm -r s -m 2047 -n 2047 -k 2047 --lda 2047 --ldb 2047 --ldc 2047 --transposeB N 2>&1 | tee -a $logs/rocblas_gemm_single.log
./rocblas-bench -f gemm -r s -m 2048 -n 2048 -k 2048 --lda 2048 --ldb 2048 --ldc 2048 --transposeB T 2>&1 | tee -a $logs/rocblas_gemm_single.log
./rocblas-bench -f gemm -r s -m 2048 -n 2048 -k 2048 --lda 2048 --ldb 2048 --ldc 2048 --transposeB N 2>&1 | tee -a $logs/rocblas_gemm_single.log
./rocblas-bench -f gemm -r s -m 4095 -n 4095 -k 4095 --lda 4095 --ldb 4095 --ldc 4095 --transposeB T 2>&1 | tee -a $logs/rocblas_gemm_single.log
./rocblas-bench -f gemm -r s -m 4095 -n 4095 -k 4095 --lda 4095 --ldb 4095 --ldc 4095 --transposeB N 2>&1 | tee -a $logs/rocblas_gemm_single.log
./rocblas-bench -f gemm -r s -m 4096 -n 4096 -k 4096 --lda 4096 --ldb 4096 --ldc 4096 --transposeB T 2>&1 | tee -a $logs/rocblas_gemm_single.log
./rocblas-bench -f gemm -r s -m 4096 -n 4096 -k 4096 --lda 4096 --ldb 4096 --ldc 4096 --transposeB N 2>&1 | tee -a $logs/rocblas_gemm_single.log
./rocblas-bench -f gemm -r s -m 5503 -n 5503 -k 5503 --lda 5503 --ldb 5503 --ldc 5503 --transposeB T 2>&1 | tee -a $logs/rocblas_gemm_single.log
./rocblas-bench -f gemm -r s -m 5503 -n 5503 -k 5503 --lda 5503 --ldb 5503 --ldc 5503 --transposeB N 2>&1 | tee -a $logs/rocblas_gemm_single.log
./rocblas-bench -f gemm -r s -m 5504 -n 5504 -k 5504 --lda 5504 --ldb 5504 --ldc 5504 --transposeB T 2>&1 | tee -a $logs/rocblas_gemm_single.log
./rocblas-bench -f gemm -r s -m 5504 -n 5504 -k 5504 --lda 5504 --ldb 5504 --ldc 5504 --transposeB N 2>&1 | tee -a $logs/rocblas_gemm_single.log
./rocblas-bench -f gemm -r s -m 5760 -n 5760 -k 5760 --lda 5760 --ldb 5760 --ldc 5760 --transposeB T 2>&1 | tee -a $logs/rocblas_gemm_single.log
./rocblas-bench -f gemm -r s -m 5760 -n 5760 -k 5760 --lda 5760 --ldb 5760 --ldc 5760 --transposeB N 2>&1 | tee -a $logs/rocblas_gemm_single.log
./rocblas-bench -f gemm -r s -m 6143 -n 6143 -k 6143 --lda 6143 --ldb 6143 --ldc 6143 --transposeB T 2>&1 | tee -a $logs/rocblas_gemm_single.log
./rocblas-bench -f gemm -r s -m 6143 -n 6143 -k 6143 --lda 6143 --ldb 6143 --ldc 6143 --transposeB N 2>&1 | tee -a $logs/rocblas_gemm_single.log
./rocblas-bench -f gemm -r s -m 6144 -n 6144 -k 6144 --lda 6144 --ldb 6144 --ldc 6144 --transposeB T 2>&1 | tee -a $logs/rocblas_gemm_single.log
./rocblas-bench -f gemm -r s -m 6144 -n 6144 -k 6144 --lda 6144 --ldb 6144 --ldc 6144 --transposeB N 2>&1 | tee -a $logs/rocblas_gemm_single.log
./rocblas-bench -f gemm -r s -m 7744 -n 7744 -k 7744 --lda 7744 --ldb 7744 --ldc 7744 --transposeB T 2>&1 | tee -a $logs/rocblas_gemm_single.log
./rocblas-bench -f gemm -r s -m 7744 -n 7744 -k 7744 --lda 7744 --ldb 7744 --ldc 7744 --transposeB N 2>&1 | tee -a $logs/rocblas_gemm_single.log
./rocblas-bench -f gemm -r s -m 8191 -n 8191 -k 8191 --lda 8191 --ldb 8191 --ldc 8191 --transposeB T 2>&1 | tee -a $logs/rocblas_gemm_single.log
./rocblas-bench -f gemm -r s -m 8191 -n 8191 -k 8191 --lda 8191 --ldb 8191 --ldc 8191 --transposeB N 2>&1 | tee -a $logs/rocblas_gemm_single.log
./rocblas-bench -f gemm -r s -m 8192 -n 8192 -k 8192 --lda 8192 --ldb 8192 --ldc 8192 --transposeB T 2>&1 | tee -a $logs/rocblas_gemm_single.log
./rocblas-bench -f gemm -r s -m 8192 -n 8192 -k 8192 --lda 8192 --ldb 8192 --ldc 8192 --transposeB N 2>&1 | tee -a $logs/rocblas_gemm_single.log

echo "===================== Running rocBLAS : GEMM - Double Precision =================================" 2>&1 | tee $logs/rocblas_gemm_double.log

./rocblas-bench -f gemm -r d -m 2047 -n 2047 -k 2047 --lda 2047 --ldb 2047 --ldc 2047 --transposeB T 2>&1 | tee -a $logs/rocblas_gemm_double.log
./rocblas-bench -f gemm -r d -m 2047 -n 2047 -k 2047 --lda 2047 --ldb 2047 --ldc 2047 --transposeB N 2>&1 | tee -a $logs/rocblas_gemm_double.log
./rocblas-bench -f gemm -r d -m 2048 -n 2048 -k 2048 --lda 2048 --ldb 2048 --ldc 2048 --transposeB T 2>&1 | tee -a $logs/rocblas_gemm_double.log
./rocblas-bench -f gemm -r d -m 2048 -n 2048 -k 2048 --lda 2048 --ldb 2048 --ldc 2048 --transposeB N 2>&1 | tee -a $logs/rocblas_gemm_double.log
./rocblas-bench -f gemm -r d -m 4095 -n 4095 -k 4095 --lda 4095 --ldb 4095 --ldc 4095 --transposeB T 2>&1 | tee -a $logs/rocblas_gemm_double.log
./rocblas-bench -f gemm -r d -m 4095 -n 4095 -k 4095 --lda 4095 --ldb 4095 --ldc 4095 --transposeB N 2>&1 | tee -a $logs/rocblas_gemm_double.log
./rocblas-bench -f gemm -r d -m 4096 -n 4096 -k 4096 --lda 4096 --ldb 4096 --ldc 4096 --transposeB T 2>&1 | tee -a $logs/rocblas_gemm_double.log
./rocblas-bench -f gemm -r d -m 4096 -n 4096 -k 4096 --lda 4096 --ldb 4096 --ldc 4096 --transposeB N 2>&1 | tee -a $logs/rocblas_gemm_double.log
./rocblas-bench -f gemm -r d -m 5503 -n 5503 -k 5503 --lda 5503 --ldb 5503 --ldc 5503 --transposeB T 2>&1 | tee -a $logs/rocblas_gemm_double.log
./rocblas-bench -f gemm -r d -m 5503 -n 5503 -k 5503 --lda 5503 --ldb 5503 --ldc 5503 --transposeB N 2>&1 | tee -a $logs/rocblas_gemm_double.log
./rocblas-bench -f gemm -r d -m 5504 -n 5504 -k 5504 --lda 5504 --ldb 5504 --ldc 5504 --transposeB T 2>&1 | tee -a $logs/rocblas_gemm_double.log
./rocblas-bench -f gemm -r d -m 5504 -n 5504 -k 5504 --lda 5504 --ldb 5504 --ldc 5504 --transposeB N 2>&1 | tee -a $logs/rocblas_gemm_double.log
./rocblas-bench -f gemm -r d -m 5760 -n 5760 -k 5760 --lda 5760 --ldb 5760 --ldc 5760 --transposeB T 2>&1 | tee -a $logs/rocblas_gemm_double.log
./rocblas-bench -f gemm -r d -m 5760 -n 5760 -k 5760 --lda 5760 --ldb 5760 --ldc 5760 --transposeB N 2>&1 | tee -a $logs/rocblas_gemm_double.log
./rocblas-bench -f gemm -r d -m 6143 -n 6143 -k 6143 --lda 6143 --ldb 6143 --ldc 6143 --transposeB T 2>&1 | tee -a $logs/rocblas_gemm_double.log
./rocblas-bench -f gemm -r d -m 6143 -n 6143 -k 6143 --lda 6143 --ldb 6143 --ldc 6143 --transposeB N 2>&1 | tee -a $logs/rocblas_gemm_double.log
./rocblas-bench -f gemm -r d -m 6144 -n 6144 -k 6144 --lda 6144 --ldb 6144 --ldc 6144 --transposeB T 2>&1 | tee -a $logs/rocblas_gemm_double.log
./rocblas-bench -f gemm -r d -m 6144 -n 6144 -k 6144 --lda 6144 --ldb 6144 --ldc 6144 --transposeB N 2>&1 | tee -a $logs/rocblas_gemm_double.log
./rocblas-bench -f gemm -r d -m 7744 -n 7744 -k 7744 --lda 7744 --ldb 7744 --ldc 7744 --transposeB T 2>&1 | tee -a $logs/rocblas_gemm_double.log
./rocblas-bench -f gemm -r d -m 7744 -n 7744 -k 7744 --lda 7744 --ldb 7744 --ldc 7744 --transposeB N 2>&1 | tee -a $logs/rocblas_gemm_double.log
./rocblas-bench -f gemm -r d -m 8191 -n 8191 -k 8191 --lda 8191 --ldb 8191 --ldc 8191 --transposeB T 2>&1 | tee -a $logs/rocblas_gemm_double.log
./rocblas-bench -f gemm -r d -m 8191 -n 8191 -k 8191 --lda 8191 --ldb 8191 --ldc 8191 --transposeB N 2>&1 | tee -a $logs/rocblas_gemm_double.log
./rocblas-bench -f gemm -r d -m 8192 -n 8192 -k 8192 --lda 8192 --ldb 8192 --ldc 8192 --transposeB T 2>&1 | tee -a $logs/rocblas_gemm_double.log
./rocblas-bench -f gemm -r d -m 8192 -n 8192 -k 8192 --lda 8192 --ldb 8192 --ldc 8192 --transposeB N 2>&1 | tee -a $logs/rocblas_gemm_double.log

echo "===================== Running Tensorflow alexnet and resnet50=================================" 2>&1 | tee $logs/Tensorflow_sample.log

cd $dir/benchmarks/scripts/tf_cnn_benchmarks/
python3 tf_cnn_benchmarks.py --model=alexnet --num_gpus=1 --batch_size=512 --num_batches=10 --print_training_accuracy=True 2>&1 | tee -a $logs/Tensorflow_sample.log
python3 tf_cnn_benchmarks.py --model=resnet50 --num_gpus=1 --batch_size=128 --num_batches=10 --print_training_accuracy=True 2>&1 | tee -a $logs/Tensorflow_sample.log

echo "===================== Smoke Test finished================================="

