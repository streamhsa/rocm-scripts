#!/bin/bash
# run_soumith_benchmarks is not applicable for TF1.8 and higher. refer https://github.com/soumith/convnet-benchmarks/issues/138
#This script is based on python3 . When TF was build with python3, use python3 . When TF was built with python2, replace python3 to python2
#export HIP_VISIBLE_DEVICES=0
#export HSA_ENABLE_SDMA=0

cwd=`pwd`
BASEDIR=/root


mkdir -p /dockerx/tf-rccllogs

MODELDIR="$BASEDIR/models"
BENCHDIR="$BASEDIR/benchmarks"
LOGDIR="/dockerx/tf-rccllogs"

run_tf_cnn_benchmarks()
{
    echo "=======================tf_cnn_benchmarks-BS1-64==============="
        cd $BENCHDIR
 #       cd $BENCHDIR/scripts/tf_cnn_benchmarks/
#       sed -i 's/import cPickle/import pickle/g' datasets.py
        cd $BENCHDIR
#     MODELS="alexnet"
       export TF_ROCM_FUSION_ENABLE=1
        MODELS="alexnet googlenet inception3 inception4 lenet overfeat resnet50 resnet152_v2 trivial vgg11 vgg16 vgg19 resnet101 resnet50_v1.5"
	NGPUS=1
	ITERATIONS=50
	BATCH_SIZE="1 2 4 8 16 32 64"

	for j in ${BATCH_SIZE[@]}
	do
	for i in ${MODELS[@]}
	do
         /usr/bin/python3 ./scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model=$i \
         --print_training_accuracy=True \
         --num_batches=${ITERATIONS} --all_reduce_spec=nccl --variable_update=replicated \
         --num_gpus=${NGPUS} --batch_size=$j  2>&1 | tee $LOGDIR/tfrccl-$i-$j.txt
         done
         done

}

run_tf_cnn_benchmarks_128()
{
    echo "=======================tf_cnn_benchmarks_BS128==============="
        cd $BENCHDIR
    export TF_ROCM_FUSION_ENABLE=1
    MODELS="alexnet googlenet inception3 lenet overfeat resnet50 trivial vgg11 vgg16 vgg19 resnet50_v1.5"
        NGPUS=1
        ITERATIONS=50
        BATCH_SIZE=128

        for j in ${BATCH_SIZE[@]}
        do
        for i in ${MODELS[@]}
        do
    /usr/bin/python3 ./scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model=$i \
    --print_training_accuracy=True \
    --num_batches=${ITERATIONS} --all_reduce_spec=nccl --variable_update=replicated \
    --num_gpus=${NGPUS} --batch_size=$j  2>&1 | tee $LOGDIR/tfrccl-$i-$j.txt
    done
    done

}

run_tf_cnn_benchmarks_256()
{
    echo "=======================tf_cnn_benchmarks_BS256==============="
        cd $BENCHDIR
    export TF_ROCM_FUSION_ENABLE=1
    MODELS="alexnet googlenet resnet50_v1.5"
        NGPUS=1
        ITERATIONS=50
        BATCH_SIZE=256

        for j in ${BATCH_SIZE[@]}
        do
        for i in ${MODELS[@]}
        do
    /usr/bin/python3 ./scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model=$i \
    --print_training_accuracy=True \
    --num_batches=${ITERATIONS} --all_reduce_spec=nccl --variable_update=replicated \
    --num_gpus=${NGPUS} --batch_size=$j  2>&1 | tee $LOGDIR/tfrccl-$i-$j.txt
    done
    done

}


run_tf_cnn_benchmarks_512()
{
    echo "=======================tf_cnn_benchmarks_BS512==============="
        cd $BENCHDIR
    export TF_ROCM_FUSION_ENABLE=1
    MODELS="alexnet resnet50_v1.5"
        NGPUS=1
        ITERATIONS=50
        BATCH_SIZE=512

        for j in ${BATCH_SIZE[@]}
        do
        for i in ${MODELS[@]}
        do
    /usr/bin/python3 ./scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model=$i \
    --print_training_accuracy=True \
    --num_batches=${ITERATIONS} --all_reduce_spec=nccl --variable_update=replicated \
    --num_gpus=${NGPUS} --batch_size=$j  2>&1 | tee $LOGDIR/tfrccl-$i-$j.txt
    done
    done
} 


run_tf_cnn_benchmarks
run_tf_cnn_benchmarks_128
run_tf_cnn_benchmarks_256
run_tf_cnn_benchmarks_512


