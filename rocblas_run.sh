NUMPATH="$HOME/Desktop/Numbers/workloads"

#Remove Temp files
echo AH64_uh1 | sudo rm -rf /tmp/*
echo AH64_uh1 | sudo rm -rf ~/.cache/*

#Remove old log file
cd $NUMPATH
echo AH64_uh1 | sudo rm -rf rocblas_*.log

cd $HOME/Desktop/rocm_tests/mathlibs/rocblas/build/release/clients/staging

#Performance calculation

echo "===== Running rocblas : GEMM - single Precision =====" 2>&1 | tee $NUMPATH/rocblas_GEMM_single.log
/opt/rocm/bin/rocm-smi --setperflevel high
./rocblas-bench -f gemm -r s -m 2047 -n 2047 -k 2047 --lda 2047 --ldb 2047 --ldc 2047 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_single.log
./rocblas-bench -f gemm -r s -m 2047 -n 2047 -k 2047 --lda 2047 --ldb 2047 --ldc 2047 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_single.log
./rocblas-bench -f gemm -r s -m 2048 -n 2048 -k 2048 --lda 2048 --ldb 2048 --ldc 2048 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_single.log
./rocblas-bench -f gemm -r s -m 2048 -n 2048 -k 2048 --lda 2048 --ldb 2048 --ldc 2048 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_single.log
./rocblas-bench -f gemm -r s -m 4095 -n 4095 -k 4095 --lda 4095 --ldb 4095 --ldc 4095 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_single.log
./rocblas-bench -f gemm -r s -m 4095 -n 4095 -k 4095 --lda 4095 --ldb 4095 --ldc 4095 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_single.log
./rocblas-bench -f gemm -r s -m 4096 -n 4096 -k 4096 --lda 4096 --ldb 4096 --ldc 4096 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_single.log
./rocblas-bench -f gemm -r s -m 4096 -n 4096 -k 4096 --lda 4096 --ldb 4096 --ldc 4096 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_single.log
./rocblas-bench -f gemm -r s -m 5503 -n 5503 -k 5503 --lda 5503 --ldb 5503 --ldc 5503 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_single.log
./rocblas-bench -f gemm -r s -m 5503 -n 5503 -k 5503 --lda 5503 --ldb 5503 --ldc 5503 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_single.log
./rocblas-bench -f gemm -r s -m 5504 -n 5504 -k 5504 --lda 5504 --ldb 5504 --ldc 5504 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_single.log
./rocblas-bench -f gemm -r s -m 5504 -n 5504 -k 5504 --lda 5504 --ldb 5504 --ldc 5504 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_single.log
./rocblas-bench -f gemm -r s -m 5760 -n 5760 -k 5760 --lda 5760 --ldb 5760 --ldc 5760 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_single.log
./rocblas-bench -f gemm -r s -m 5760 -n 5760 -k 5760 --lda 5760 --ldb 5760 --ldc 5760 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_single.log
./rocblas-bench -f gemm -r s -m 6143 -n 6143 -k 6143 --lda 6143 --ldb 6143 --ldc 6143 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_single.log
./rocblas-bench -f gemm -r s -m 6143 -n 6143 -k 6143 --lda 6143 --ldb 6143 --ldc 6143 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_single.log
./rocblas-bench -f gemm -r s -m 6144 -n 6144 -k 6144 --lda 6144 --ldb 6144 --ldc 6144 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_single.log
./rocblas-bench -f gemm -r s -m 6144 -n 6144 -k 6144 --lda 6144 --ldb 6144 --ldc 6144 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_single.log
./rocblas-bench -f gemm -r s -m 7744 -n 7744 -k 7744 --lda 7744 --ldb 7744 --ldc 7744 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_single.log
./rocblas-bench -f gemm -r s -m 7744 -n 7744 -k 7744 --lda 7744 --ldb 7744 --ldc 7744 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_single.log
./rocblas-bench -f gemm -r s -m 8191 -n 8191 -k 8191 --lda 8191 --ldb 8191 --ldc 8191 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_single.log
./rocblas-bench -f gemm -r s -m 8191 -n 8191 -k 8191 --lda 8191 --ldb 8191 --ldc 8191 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_single.log
./rocblas-bench -f gemm -r s -m 8192 -n 8192 -k 8192 --lda 8192 --ldb 8192 --ldc 8192 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_single.log
./rocblas-bench -f gemm -r s -m 8192 -n 8192 -k 8192 --lda 8192 --ldb 8192 --ldc 8192 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_single.log


echo "===== Running rocblas : GEMM - Double Precision =====" 2>&1 | tee $NUMPATH/rocblas_GEMM_double.log
/opt/rocm/bin/rocm-smi --setperflevel high
./rocblas-bench -f gemm -r d -m 2047 -n 2047 -k 2047 --lda 2047 --ldb 2047 --ldc 2047 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_double.log
./rocblas-bench -f gemm -r d -m 2047 -n 2047 -k 2047 --lda 2047 --ldb 2047 --ldc 2047 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_double.log
./rocblas-bench -f gemm -r d -m 2048 -n 2048 -k 2048 --lda 2048 --ldb 2048 --ldc 2048 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_double.log
./rocblas-bench -f gemm -r d -m 2048 -n 2048 -k 2048 --lda 2048 --ldb 2048 --ldc 2048 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_double.log
./rocblas-bench -f gemm -r d -m 4095 -n 4095 -k 4095 --lda 4095 --ldb 4095 --ldc 4095 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_double.log
./rocblas-bench -f gemm -r d -m 4095 -n 4095 -k 4095 --lda 4095 --ldb 4095 --ldc 4095 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_double.log
./rocblas-bench -f gemm -r d -m 4096 -n 4096 -k 4096 --lda 4096 --ldb 4096 --ldc 4096 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_double.log
./rocblas-bench -f gemm -r d -m 4096 -n 4096 -k 4096 --lda 4096 --ldb 4096 --ldc 4096 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_double.log
./rocblas-bench -f gemm -r d -m 5503 -n 5503 -k 5503 --lda 5503 --ldb 5503 --ldc 5503 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_double.log
./rocblas-bench -f gemm -r d -m 5503 -n 5503 -k 5503 --lda 5503 --ldb 5503 --ldc 5503 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_double.log
./rocblas-bench -f gemm -r d -m 5504 -n 5504 -k 5504 --lda 5504 --ldb 5504 --ldc 5504 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_double.log
./rocblas-bench -f gemm -r d -m 5504 -n 5504 -k 5504 --lda 5504 --ldb 5504 --ldc 5504 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_double.log
./rocblas-bench -f gemm -r d -m 5760 -n 5760 -k 5760 --lda 5760 --ldb 5760 --ldc 5760 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_double.log
./rocblas-bench -f gemm -r d -m 5760 -n 5760 -k 5760 --lda 5760 --ldb 5760 --ldc 5760 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_double.log
./rocblas-bench -f gemm -r d -m 6143 -n 6143 -k 6143 --lda 6143 --ldb 6143 --ldc 6143 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_double.log
./rocblas-bench -f gemm -r d -m 6143 -n 6143 -k 6143 --lda 6143 --ldb 6143 --ldc 6143 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_double.log
./rocblas-bench -f gemm -r d -m 6144 -n 6144 -k 6144 --lda 6144 --ldb 6144 --ldc 6144 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_double.log
./rocblas-bench -f gemm -r d -m 6144 -n 6144 -k 6144 --lda 6144 --ldb 6144 --ldc 6144 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_double.log
./rocblas-bench -f gemm -r d -m 7744 -n 7744 -k 7744 --lda 7744 --ldb 7744 --ldc 7744 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_double.log
./rocblas-bench -f gemm -r d -m 7744 -n 7744 -k 7744 --lda 7744 --ldb 7744 --ldc 7744 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_double.log
./rocblas-bench -f gemm -r d -m 8191 -n 8191 -k 8191 --lda 8191 --ldb 8191 --ldc 8191 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_double.log
./rocblas-bench -f gemm -r d -m 8191 -n 8191 -k 8191 --lda 8191 --ldb 8191 --ldc 8191 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_double.log
./rocblas-bench -f gemm -r d -m 8192 -n 8192 -k 8192 --lda 8192 --ldb 8192 --ldc 8192 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_double.log
./rocblas-bench -f gemm -r d -m 8192 -n 8192 -k 8192 --lda 8192 --ldb 8192 --ldc 8192 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_double.log


echo "===== Runninng rocblas : GEMM - Half Precision =====" 2>&1 | tee $NUMPATH/rocblas_GEMM_half.log
/opt/rocm/bin/rocm-smi --setperflevel high
./rocblas-bench -f gemm -r h -m 2047 -n 2047 -k 2047 --lda 2047 --ldb 2047 --ldc 2047 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_half.log
./rocblas-bench -f gemm -r h -m 2047 -n 2047 -k 2047 --lda 2047 --ldb 2047 --ldc 2047 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_half.log
./rocblas-bench -f gemm -r h -m 2048 -n 2048 -k 2048 --lda 2048 --ldb 2048 --ldc 2048 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_half.log
./rocblas-bench -f gemm -r h -m 2048 -n 2048 -k 2048 --lda 2048 --ldb 2048 --ldc 2048 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_half.log
./rocblas-bench -f gemm -r h -m 4095 -n 4095 -k 4095 --lda 4095 --ldb 4095 --ldc 4095 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_half.log
./rocblas-bench -f gemm -r h -m 4095 -n 4095 -k 4095 --lda 4095 --ldb 4095 --ldc 4095 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_half.log
./rocblas-bench -f gemm -r h -m 4096 -n 4096 -k 4096 --lda 4096 --ldb 4096 --ldc 4096 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_half.log
./rocblas-bench -f gemm -r h -m 4096 -n 4096 -k 4096 --lda 4096 --ldb 4096 --ldc 4096 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_half.log
./rocblas-bench -f gemm -r h -m 5503 -n 5503 -k 5503 --lda 5503 --ldb 5503 --ldc 5503 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_half.log
./rocblas-bench -f gemm -r h -m 5503 -n 5503 -k 5503 --lda 5503 --ldb 5503 --ldc 5503 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_half.log
./rocblas-bench -f gemm -r h -m 5504 -n 5504 -k 5504 --lda 5504 --ldb 5504 --ldc 5504 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_half.log
./rocblas-bench -f gemm -r h -m 5504 -n 5504 -k 5504 --lda 5504 --ldb 5504 --ldc 5504 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_half.log
./rocblas-bench -f gemm -r h -m 5760 -n 5760 -k 5760 --lda 5760 --ldb 5760 --ldc 5760 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_half.log
./rocblas-bench -f gemm -r h -m 5760 -n 5760 -k 5760 --lda 5760 --ldb 5760 --ldc 5760 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_half.log
./rocblas-bench -f gemm -r h -m 6143 -n 6143 -k 6143 --lda 6143 --ldb 6143 --ldc 6143 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_half.log
./rocblas-bench -f gemm -r h -m 6143 -n 6143 -k 6143 --lda 6143 --ldb 6143 --ldc 6143 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_half.log
./rocblas-bench -f gemm -r h -m 6144 -n 6144 -k 6144 --lda 6144 --ldb 6144 --ldc 6144 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_half.log
./rocblas-bench -f gemm -r h -m 6144 -n 6144 -k 6144 --lda 6144 --ldb 6144 --ldc 6144 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_half.log
./rocblas-bench -f gemm -r h -m 7744 -n 7744 -k 7744 --lda 7744 --ldb 7744 --ldc 7744 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_half.log
./rocblas-bench -f gemm -r h -m 7744 -n 7744 -k 7744 --lda 7744 --ldb 7744 --ldc 7744 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_half.log
./rocblas-bench -f gemm -r h -m 8191 -n 8191 -k 8191 --lda 8191 --ldb 8191 --ldc 8191 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_half.log
./rocblas-bench -f gemm -r h -m 8191 -n 8191 -k 8191 --lda 8191 --ldb 8191 --ldc 8191 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_half.log
./rocblas-bench -f gemm -r h -m 8192 -n 8192 -k 8192 --lda 8192 --ldb 8192 --ldc 8192 --transposeB T 2>&1 | tee -a $NUMPATH/rocblas_GEMM_half.log
./rocblas-bench -f gemm -r h -m 8192 -n 8192 -k 8192 --lda 8192 --ldb 8192 --ldc 8192 --transposeB N 2>&1 | tee -a $NUMPATH/rocblas_GEMM_half.log


echo "===== Runninng rocblas : GEMV - Single Precision =====" 2>&1 | tee $NUMPATH/rocblas_GEMV_single.log
/opt/rocm/bin/rocm-smi --setperflevel high
./rocblas-bench -f gemv -r s -m 1000 -n 1000 --lda 1000 --transposeA T 2>&1 | tee -a $NUMPATH/rocblas_GEMV_single.log
./rocblas-bench -f gemv -r s -m 1000 -n 1000 --lda 1000 --transposeA N 2>&1 | tee -a $NUMPATH/rocblas_GEMV_single.log
./rocblas-bench -f gemv -r s -m 1000 -n 1000 --lda 1000 --transposeA T 2>&1 | tee -a $NUMPATH/rocblas_GEMV_single.log
./rocblas-bench -f gemv -r s -m 1000 -n 1000 --lda 1000 --transposeA N 2>&1 | tee -a $NUMPATH/rocblas_GEMV_single.log
./rocblas-bench -f gemv -r s -m 1000 -n 1000 --lda 1000 --transposeA T 2>&1 | tee -a $NUMPATH/rocblas_GEMV_single.log
./rocblas-bench -f gemv -r s -m 1000 -n 1000 --lda 1000 --transposeA N 2>&1 | tee -a $NUMPATH/rocblas_GEMV_single.log
./rocblas-bench -f gemv -r s -m 10001 -n 10001 --lda 10002 --transposeA T 2>&1 | tee -a $NUMPATH/rocblas_GEMV_single.log
./rocblas-bench -f gemv -r s -m 10001 -n 10001 --lda 10002 --transposeA N 2>&1 | tee -a $NUMPATH/rocblas_GEMV_single.log
./rocblas-bench -f gemv -r s -m 10001 -n 10001 --lda 10002 --transposeA T 2>&1 | tee -a $NUMPATH/rocblas_GEMV_single.log
./rocblas-bench -f gemv -r s -m 10001 -n 10001 --lda 10002 --transposeA N 2>&1 | tee -a $NUMPATH/rocblas_GEMV_single.log
./rocblas-bench -f gemv -r s -m 10001 -n 10001 --lda 10002 --transposeA T 2>&1 | tee -a $NUMPATH/rocblas_GEMV_single.log
./rocblas-bench -f gemv -r s -m 10001 -n 10001 --lda 10002 --transposeA N 2>&1 | tee -a $NUMPATH/rocblas_GEMV_single.log


echo "===== Runninng rocblas : GEMV - Double Precision =====" 2>&1 | tee $NUMPATH/rocblas_GEMV_double.log
/opt/rocm/bin/rocm-smi --setperflevel high
./rocblas-bench -f gemv -r d -m 1000 -n 1000 --lda 1000 --transposeA T 2>&1 | tee -a $NUMPATH/rocblas_GEMV_double.log
./rocblas-bench -f gemv -r d -m 1000 -n 1000 --lda 1000 --transposeA N 2>&1 | tee -a $NUMPATH/rocblas_GEMV_double.log
./rocblas-bench -f gemv -r d -m 1000 -n 1000 --lda 1000 --transposeA T 2>&1 | tee -a $NUMPATH/rocblas_GEMV_double.log
./rocblas-bench -f gemv -r d -m 1000 -n 1000 --lda 1000 --transposeA N 2>&1 | tee -a $NUMPATH/rocblas_GEMV_double.log
./rocblas-bench -f gemv -r d -m 1000 -n 1000 --lda 1000 --transposeA T 2>&1 | tee -a $NUMPATH/rocblas_GEMV_double.log
./rocblas-bench -f gemv -r d -m 1000 -n 1000 --lda 1000 --transposeA N 2>&1 | tee -a $NUMPATH/rocblas_GEMV_double.log
./rocblas-bench -f gemv -r d -m 10001 -n 10001 --lda 10002 --transposeA T 2>&1 | tee -a $NUMPATH/rocblas_GEMV_double.log
./rocblas-bench -f gemv -r d -m 10001 -n 10001 --lda 10002 --transposeA N 2>&1 | tee -a $NUMPATH/rocblas_GEMV_double.log
./rocblas-bench -f gemv -r d -m 10001 -n 10001 --lda 10002 --transposeA T 2>&1 | tee -a $NUMPATH/rocblas_GEMV_double.log
./rocblas-bench -f gemv -r d -m 10001 -n 10001 --lda 10002 --transposeA N 2>&1 | tee -a $NUMPATH/rocblas_GEMV_double.log
./rocblas-bench -f gemv -r d -m 10001 -n 10001 --lda 10002 --transposeA T 2>&1 | tee -a $NUMPATH/rocblas_GEMV_double.log
./rocblas-bench -f gemv -r d -m 10001 -n 10001 --lda 10002 --transposeA N 2>&1 | tee -a $NUMPATH/rocblas_GEMV_double.log


echo "===== Runninng rocblas : GER - Single Precision =====" 2>&1 | tee $NUMPATH/rocblas_GER_single.log
/opt/rocm/bin/rocm-smi --setperflevel high
./rocblas-bench -f ger -r s -m 1000 -n 1000 --lda 1000 2>&1 | tee -a $NUMPATH/rocblas_GER_single.log
./rocblas-bench -f ger -r s -m 10001 -n 10001 --lda 10002 2>&1 | tee -a $NUMPATH/rocblas_GER_single.log
./rocblas-bench -f ger -r s -m 1000 -n 1000 --lda 1000 2>&1 | tee -a $NUMPATH/rocblas_GER_single.log
./rocblas-bench -f ger -r s -m 10001 -n 10001 --lda 10002 2>&1 | tee -a $NUMPATH/rocblas_GER_single.log
./rocblas-bench -f ger -r s -m 1000 -n 1000 --lda 1000 2>&1 | tee -a $NUMPATH/rocblas_GER_single.log
./rocblas-bench -f ger -r s -m 10001 -n 10001 --lda 10002 2>&1 | tee -a $NUMPATH/rocblas_GER_single.log


echo "===== Runninng rocblas : GER - Double Precision =====" 2>&1 | tee $NUMPATH/rocblas_GER_double.log
/opt/rocm/bin/rocm-smi --setperflevel high
./rocblas-bench -f ger -r d -m 1000 -n 1000 --lda 1000 2>&1 | tee -a $NUMPATH/rocblas_GER_double.log
./rocblas-bench -f ger -r d -m 10001 -n 10001 --lda 10002 2>&1 | tee -a $NUMPATH/rocblas_GER_double.log
./rocblas-bench -f ger -r d -m 1000 -n 1000 --lda 1000 2>&1 | tee -a $NUMPATH/rocblas_GER_double.log
./rocblas-bench -f ger -r d -m 10001 -n 10001 --lda 10002 2>&1 | tee -a $NUMPATH/rocblas_GER_double.log
./rocblas-bench -f ger -r d -m 1000 -n 1000 --lda 1000 2>&1 | tee -a $NUMPATH/rocblas_GER_double.log
./rocblas-bench -f ger -r d -m 10001 -n 10001 --lda 10002 2>&1 | tee -a $NUMPATH/rocblas_GER_double.log


echo "===== Runninng rocblas : DOT - Single Precision =====" 2>&1 | tee $NUMPATH/rocblas_DOT_single.log
/opt/rocm/bin/rocm-smi --setperflevel high
./rocblas-bench -f dot -r s -n 1000 2>&1 | tee -a $NUMPATH/rocblas_DOT_single.log
./rocblas-bench -f dot -r s -n 100001 2>&1 | tee -a $NUMPATH/rocblas_DOT_single.log
./rocblas-bench -f dot -r s -n 10000002 2>&1 | tee -a $NUMPATH/rocblas_DOT_single.log


echo "===== Runninng rocblas : DOT - Double Precision =====" 2>&1 | tee $NUMPATH/rocblas_DOT_double.log
/opt/rocm/bin/rocm-smi --setperflevel high
./rocblas-bench -f dot -r d -n 1000 2>&1 | tee -a $NUMPATH/rocblas_DOT_double.log
./rocblas-bench -f dot -r d -n 100001 2>&1 | tee -a $NUMPATH/rocblas_DOT_double.log
./rocblas-bench -f dot -r d -n 10000002 2>&1 | tee -a $NUMPATH/rocblas_DOT_double.log

