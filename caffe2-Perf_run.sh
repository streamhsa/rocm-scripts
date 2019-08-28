#!/bin/bash
 
# check if caffe2 is  build and installed properly. success => all good; failure => not built and installed properly.
  echo "===============If success, caffe2 installed properly======="
  cd /root/caffe2/build
  python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
  
  
echo "=========================start caffe2 benchmark====================="
cd /root/caffe2/build
wget https://raw.githubusercontent.com/pramenku/rocm-scripts/master/convnet_benchmarks_dpm.py
chmod 775 convnet_benchmarks_dpm.py
apt-get update -y && apt-get install -y dos2unix
dos2unix convnet_benchmarks_dpm.py

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


###################################2-GPU#################################################
#echo "========================= caffe2 Alexnet 2048 model====================="
#python convnet_benchmarks_dpm.py --model AlexNet  --num_gpus 2 --batch_size 2048 --iterations 10 |& tee -a caffe2-alexnet-1024_2.log

#echo "========================= caffe2 Inception 256 model====================="
#python convnet_benchmarks_dpm.py --model Inception  --num_gpus 2 --batch_size 256 --iterations 10 |& tee -a caffe2-Inception-128_2.log

#echo "========================= caffe2 Resnet50 128 model====================="
#python convnet_benchmarks_dpm.py --model Resnet50  --num_gpus 2 --batch_size 128 --iterations 10 |& tee -a caffe2-Resnet50-64_2.log

#echo "========================= caffe2 Resnet101 128 model====================="
#python convnet_benchmarks_dpm.py --model Resnet101  --num_gpus 2 --batch_size 128 --iterations 10 |& tee -a caffe2-Resnet101-64_2.log

#echo "========================= caffe2 Resnext101 128 model====================="
#python convnet_benchmarks_dpm.py --model Resnext101  --num_gpus 2 --batch_size 128 --iterations 10 |& tee -a caffe2-Resnext101-64_2.log


###################################4-GPU#################################################
#echo "========================= caffe2 Alexnet 4096 & 256 model====================="
#python convnet_benchmarks_dpm.py --model AlexNet  --num_gpus 4 --batch_size 4096 --iterations 10 |& tee -a caffe2-alexnet-1024.log 
#python convnet_benchmarks_dpm.py --model AlexNet  --num_gpus 4 --batch_size 256 --iterations 10 |& tee -a caffe2-alexnet-64.log

#echo "========================= caffe2 Inception  512 & 256 model====================="
#python convnet_benchmarks_dpm.py --model Inception  --num_gpus 4 --batch_size 512 --iterations 10 |& tee -a caffe2-Inception-128.log
#python convnet_benchmarks_dpm.py --model Inception  --num_gpus 4 --batch_size 256 --iterations 10 |& tee -a caffe2-Inception-64.log

#echo "========================= caffe2 Resnet50 256 & 128 model====================="
#python convnet_benchmarks_dpm.py --model Resnet50  --num_gpus 4 --batch_size 256 --iterations 10 |& tee -a caffe2-Resnet50-64.log
#python convnet_benchmarks_dpm.py --model Resnet50  --num_gpus 4 --batch_size 128 --iterations 10 |& tee -a caffe2-Resnet50-32.log

#echo "========================= caffe2 Resnet101 256 & 128 model====================="
#python convnet_benchmarks_dpm.py --model Resnet101  --num_gpus 4 --batch_size 256 --iterations 10 |& tee -a caffe2-Resnet101-64.log
#python convnet_benchmarks_dpm.py --model Resnet101  --num_gpus 4 --batch_size 128 --iterations 10 |& tee -a caffe2-Resnet101-32.log

#echo "========================= caffe2 Resnext101 64 & 32 model====================="
#python convnet_benchmarks_dpm.py --model Resnext101  --num_gpus 4 --batch_size 64 --iterations 10 |& tee -a caffe2-Resnext101-64.log
#python convnet_benchmarks_dpm.py --model Resnext101  --num_gpus 4 --batch_size 32 --iterations 10 |& tee -a caffe2-Resnext101-32.log

echo "========================end caffe2 benchmark====================="
