#!/bin/bash
# check if caffe2 is  build and installed properly. success => all good; failure => not built and installed properly.
echo "===============If success, caffe2 installed properly======="
 cd /home/caffe2/build
 python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
  
# run the unit tests
echo "=========================start caffe2 unittests====================="
cd /home/caffe2/
 .jenkins/caffe2/test.sh 2>&1 | tee /dockerx/caffe2_tests.log 
  
  echo "=========================end caffe2 unittests====================="
#Run the Benchmark

echo "=========================start caffe2 benchmark====================="
cd /home/caffe2/build
wget https://raw.githubusercontent.com/pramenku/rocm-scripts/master/convnet_benchmarks_dpm.py
chmod 775 convnet_benchmarks_dpm.py

####################################1-GPU#############################################
echo "========================= caffe2 Alexnet 1024 model====================="
python convnet_benchmarks_dpm.py --model AlexNet  --num_gpus 1 --batch_size 1024 --iterations 10 |& tee -a caffe2-alexnet-1024_1.log

echo "========================= caffe2 Inception 128 model====================="
python convnet_benchmarks_dpm.py --model Inception  --num_gpus 1 --batch_size 128 --iterations 10 |& tee -a caffe2-Inception-128_1.log

echo "========================= caffe2 Resnet50 64 model====================="
python convnet_benchmarks_dpm.py --model Resnet50  --num_gpus 1 --batch_size 64 --iterations 10 |& tee -a caffe2-Resnet50-64_1.log

echo "========================= caffe2 Resnet101 64  model====================="
python convnet_benchmarks_dpm.py --model Resnet101  --num_gpus 1 --batch_size 64 --iterations 10 |& tee -a caffe2-Resnet101-64_1.log

echo "========================= caffe2 Resnext101 64 model====================="
python convnet_benchmarks_dpm.py --model Resnext101  --num_gpus 1 --batch_size 64 --iterations 10 |& tee -a caffe2-Resnext101-64_1.log

echo "========================end caffe2 benchmark====================="

cd /home/caffe2
.jenkins/caffe2/bench.sh 2>&1 | tee /dockerx/caffe2_bench.log

