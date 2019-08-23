#!/bin/sh

cd /root/pytorch/build
wget  https://raw.githubusercontent.com/wiki/ROCmSoftwarePlatform/pytorch/micro_benchmarking_pytorch.py
wget  https://raw.githubusercontent.com/wiki/ROCmSoftwarePlatform/pytorch/fp16util.py 
apt-get update -y && apt-get install -y dos2unix
chmod 775 micro_benchmarking_pytorch.py fp16util.py 

pip install torchvision==0.2.2.post3

echo "========================= pytorch resnet50====================="
python micro_benchmarking_pytorch.py --network resnet50 --batch-size 256 --iterations 10
echo "========================= pytorch resnet101====================="
python micro_benchmarking_pytorch.py --network resnet101 --batch-size 128 --iterations 10
echo "========================= pytorch resnet152====================="
python micro_benchmarking_pytorch.py --network resnet152 --batch-size 128 --iterations 10
echo "========================= pytorch Alexnet====================="
python micro_benchmarking_pytorch.py --network alexnet --batch-size 1024 --iterations 10
echo "========================= pytorch Squeezenet===================="
python micro_benchmarking_pytorch.py --network SqueezeNet --batch-size 128 --iterations 10 
echo "========================= pytorch Inceptionv3====================="
python micro_benchmarking_pytorch.py --network inception_v3 --batch-size 256 --iterations 10
echo "========================= pytorch densenet121====================="
python micro_benchmarking_pytorch.py --network densenet121 --batch-size 128 --iterations 10
echo "========================= pytorch vgg16====================="
python micro_benchmarking_pytorch.py --network vgg16 --batch-size 128 --iterations 10 
echo "========================= pytorch vgg19===================="
python micro_benchmarking_pytorch.py --network vgg19 --batch-size 128 --iterations 10

echo "========================= pytorch  resnet50 fp16====================="
python micro_benchmarking_pytorch.py --network resnet50 --batch-size 256 --iterations 10 --fp16 1
echo "========================= pytorch  resnet101 fp16====================="
python micro_benchmarking_pytorch.py --network resnet101 --batch-size 128 --iterations 10 --fp16 1
echo "========================= pytorch  resnet152 fp16====================="
python micro_benchmarking_pytorch.py --network resnet152 --batch-size 128 --iterations 10 --fp16 1
echo "========================= pytorch  alexnet fp16====================="
python micro_benchmarking_pytorch.py --network alexnet --batch-size 1024 --iterations 10 --fp16 1
echo "========================= pytorch  squeeznet fp16====================="
python micro_benchmarking_pytorch.py --network SqueezeNet --batch-size 128 --iterations 10 --fp16 1
echo "========================= pytorch inceptionv3 fp16====================="
python micro_benchmarking_pytorch.py --network inception_v3 --batch-size 256 --iterations 10 --fp16 1
echo "========================= pytorch  densenet121 fp16====================="
python micro_benchmarking_pytorch.py --network densenet121 --batch-size 128 --iterations 10 --fp16 1
echo "========================= pytorch  vgg16 fp16====================="
python micro_benchmarking_pytorch.py --network vgg16 --batch-size 128 --iterations 10 --fp16 1
echo "========================= pytorch  vgg19 fp16====================="
python micro_benchmarking_pytorch.py --network vgg19 --batch-size 128 --iterations 10 --fp16 1
