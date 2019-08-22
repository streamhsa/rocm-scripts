#!/bin/bash
export MIOPEN_DEBUG_AMD_WINOGRAD_RXS_FP16=1
export TF_ROCM_FUSION_ENABLE=1

cwd=`pwd`
BASEDIR=/root

export BENCHDIR="$BASEDIR/benchmarks"
export LOGDIR=/dockerx

download_tensorflow_benchmarks()
{
    cd $BASEDIR
	rm -rf benchmarks
    git clone -b cnn_tf_v1.12_compatible https://github.com/tensorflow/benchmarks&& cd benchmarks
	#git checkout -b may22 ddb23306fdc60fefe620e6ce633bcd645561cb0d
	#sed -i 's|from tensorflow.contrib import nccl|#from tensorflow.contrib import nccl|g' ./scripts/tf_cnn_benchmarks/variable_mgr.py
	cd ..
    pushd benchmarks
    popd
}

function tf_cnn_benchmarks()
{
	cd $BENCHDIR
  model=$1
  num_gpus=$2
  bsz=$3
  f_only=$4
  num_batches=100
  display_every=10
  #echo "Model:${model}_${bsz}_${f_only}_${num_gpus}"
  /usr/bin/python3 scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --num_gpus=$num_gpus --batch_size=$bsz --model=$model --variable_update=parameter_server --local_parameter_device=cpu --forward_only=${f_only} --print_training_accuracy=True --num_batches=${num_batches} --display_every=${display_every} --use_fp16=True 2>&1 | tee -a tf_cnn_benchmarks_FP16_log.txt
  cp -rf tf_cnn_benchmarks_FP16_log.txt $LOGDIR
  #python3 tf_cnn_benchmarks.py --num_gpus=$num_gpus --batch_size=$bsz --model=$model --forward_only=${f_only} --print_training_accuracy=True --num_batches=${num_batches} --display_every=${display_every} --use_fp16=True --xla=True 2>&1 | tee $log_file
  #grep -E "total images/sec" $log_file
}

function tf_test()
{
  models=$1
  gpu_array=$2
  batch_array=$3
  modes=$4
  for net in ${models[@]}
  do
    for num_gpus in ${gpu_array[@]}
    do
      for batch in ${batch_array[@]}
      do
        for f_only in ${modes[@]}
        do
          tf_cnn_benchmarks $net $num_gpus $batch $f_only
        done
      done
    done
  done
}

gpu_array=1
modes=('False')
models=('resnet50' 'resnet101' 'vgg19')
batch_array=64

#download_tensorflow_benchmarks
tf_test $models $gpu_array $batch_array $modes

cd $LOGDIR
cnn_bms=`grep -E "total images/sec" tf_cnn_benchmarks_FP16_log.txt | wc -l`

echo "[STEPS]" > Results.ini
echo "Number=1" >> Results.ini
echo " " >> Results.ini

echo "[STEP_007]" >> Results.ini
echo "Description= cnn_bms" >> Results.ini
if [ $cnn_bms -eq 9 ]; then
  echo "Status=Passed" >> Results.ini
else
  echo "Status=Failed" >> Results.ini
fi

cp -rf Results.ini ../
