#!/bin/bash
current=`pwd`
dir=/root/driver
logs=/dockerx

echo "======================running clinfo==============================" 2>&1 | tee $logs/clinfo.log
/opt/rocm/opencl/bin/x86_64/clinfo 2>&1 | tee -a $logs/clinfo.log
echo "======================running rocminfo=============================" 2>&1 | tee $logs/rocminfo.log
/opt/rocm/bin/rocminfo 2>&1 | tee -a $logs/rocminfo.log

echo "======================running hipinfo=============================" 2>&1 | tee $logs/hipinfo.log
cd $dir/src/hip/samples/1_Utils/hipInfo
make
./hipInfo 2>&1 | tee -a $logs/hipinfo.log

echo "======================running rocm-bandwidth validation=================" 2>&1 | tee $logs/rocm-bw.log
cd  $dir/bin
./rocm_bandwidth_test -v 2>&1 | tee -a $logs/rocm-bw.log
echo "==============running Unidirectional bandwidth data=================" 2>&1 | tee -a $logs/rocm-bw.log
./rocm_bandwidth_test -a 2>&1 | tee -a $logs/rocm-bw.log
echo "==============running Bidirectional Bandwidth data=================" 2>&1 | tee -a $logs/rocm-bw.log
./rocm_bandwidth_test -A 2>&1 | tee -a $logs/rocm-bw.log






