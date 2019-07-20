#!/bin/bash
#sudo docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video -v $HOME/dockerx:/dockerx  -v $HOME/dataset/imagenet/tf:/imagenet sunway513/hiptensorflow:rocm2.1-tf1.12-imagenet-eval
#export HCC_SERIALIZE_KERNEL=0x3
export  MIOPEN_DEBUG_DISABLE_FIND_DB=1
export TF_ROCM_FUSION_ENABLE=1
rocm=346
#export HSA_ENABLE_SDMA=0
cd /root
rm -rf benchmarks-imagenet
git clone -b cnn_tf_v1.12_compatible https://github.com/tensorflow/benchmarks.git ~/benchmarks-imagenet
cd /root/benchmarks-imagenet
#git checkout -b sep7 6a33b4a4b5bda950bb7e45faf13120115cbfdb2f
#sed -i 's|from tensorflow.contrib import nccl|#from tensorflow.contrib import nccl|g' ./scripts/tf_cnn_benchmarks/variable_mgr.py
#sed -i 's|import cPickle|import pickle|g' ./scripts/tf_cnn_benchmarks/datasets.py

#mv -f /imagenet/train_benchmarks_resnet50 /imagenet/train_benchmarks_resnet50-$rocm

#MODEL=( alexnet googlenet inception3 inception4 lenet overfeat resnet50 resnet152_v2 trivial vgg11 vgg16 )

MODEL=( resnet50 )
NGPUS=4
BATCH_SIZE=128
#ITERATIONS=200000
EPOCH=91

for i in ${MODEL[@]}
do
#python3 ./scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model=$i \
#  --data_name=imagenet --data_dir=/imagenet --train_dir=/imagenet/train_benchmarks_$i \
#  --save_summaries_steps 10 --save_model_secs=3600 --print_training_accuracy=True \
#  --variable_update=parameter_server --local_parameter_device=cpu --num_batches=${ITERATIONS} \
#  --num_gpus=${NGPUS} --batch_size=${BATCH_SIZE} 2>&1 | tee /dockerx/LongRuns/tf-lr-$i-$BATCH_SIZE-$rocm.txt

python3 scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --data_format=NCHW --batch_size=${BATCH_SIZE} --model=$i --optimizer=momentum --variable_update=parameter_server --local_parameter_device=cpu  --nodistortions --num_gpus=${NGPUS} --num_epochs=${EPOCH} --weight_decay=1e-4 --data_dir=/imagenet --train_dir=/imagenet/train_benchmarks_$i --all_reduce_spec='' --per_gpu_thread_count=1 --save_summaries_steps=1 --summary_verbosity=1 --num_warmup_batches=0 --print_training_accuracy=True --eval=True 2>&1 | tee /dockerx/LongRuns/tf-lr-$i-$BATCH_SIZE-$rocm-accuracy.txt

done

