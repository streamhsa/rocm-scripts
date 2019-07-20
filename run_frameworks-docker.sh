#!/bin/bash
current=`pwd`
dir=/root/driver
logs=/dockerx

cd $dir/hipcaffe
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

./build/examples/cpp_classification/classification.bin \
        models/bvlc_reference_caffenet/deploy.prototxt \
    models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
    data/ilsvrc12/imagenet_mean.binaryproto \
    data/ilsvrc12/synset_words.txt \
    examples/images/cat.jpg 2>&1 | tee $logs/hipcaffe_caffenet.log


./examples/mnist/train_lenet.sh --gpu 0 2>&1 | tee $logs/hipcaffe_mnist.log
./build/tools/caffe train --solver=examples/cifar10/cifar10_quick_solver.prototxt --gpu 0 2>&1 | tee $logs/hipcaffe_cifar10.log

cd $dir/deepbench/code/amd/bin/

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib:/opt/rocm/rccl/lib
echo "===================gemm bench=========" | tee $logs/deepbench_run.log
./gemm_bench 2>&1 | tee -a $logs/deepbench_run.log
echo "===================conv bench=========" | tee -a $logs/deepbench_run.log
./conv_bench 2>&1 | tee -a $logs/deepbench_run.log
echo "===================rnn bench=========" | tee -a $logs/deepbench_run.log
./rnn_bench | tee -a $logs/deepbench_run.log
echo "===================rccl_single_all_reduce=========" | tee -a $logs/deepbench_run.log
./rccl_single_all_reduce 1 | tee -a $logs/deepbench_run.log


cd $dir/hip-mlopen/MLOpen/build
export MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0
make check -j16 2>&1| tee $logs/mlopen-ut.log
unset MIOPEN_CONV_PRECISE_ROCBLAS_TIMING

