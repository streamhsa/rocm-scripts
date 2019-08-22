
#!/bin/bash
# run_soumith_benchmarks is not applicable for TF1.8 and higher. refer https://github.com/soumith/convnet-benchmarks/issues/138
#This script is based on python3 . When TF was build with python3, use python3 . When TF was built with python2, replace python3 to python2
#export HIP_VISIBLE_DEVICES=0
#export HSA_ENABLE_SDMA=0

cwd=`pwd`
BASEDIR=/root

mkdir -p /dockerx/tf-logs

export MODELDIR="$BASEDIR/models"
export BENCHDIR="$BASEDIR/benchmarks"
export LOGDIR="/dockerx/tf-logs"

download_tensorflow_models()
{
    cd $BASEDIR
        rm -rf models
    git clone https://github.com/tensorflow/models.git
    # FIXME:  workaround to support TF v1.0.1
    pushd models
    popd
}
download_tensorflow_benchmarks()
{
    cd $BASEDIR
        rm -rf benchmarks
        git clone -b cnn_tf_v1.13_compatible https://github.com/tensorflow/benchmarks && cd benchmarks
        #git checkout -b may22 ddb23306fdc60fefe620e6ce633bcd645561cb0d
        #sed -i 's|from tensorflow.contrib import nccl|#from tensorflow.contrib import nccl|g' ./scripts/tf_cnn_benchmarks/variable_mgr.py
        cd ..
    pushd benchmarks
    popd
}
run_convolutional_quick_test()
{
    echo "=======================convolutional_quick_test==============="
        cd $MODELDIR
    rm -f out.txt sum.txt
    numtests=10
    for i in $(eval echo "{1..$numtests}")
    do
        /usr/bin/python3 ./tutorials/image/mnist/convolutional.py --self_test > out.txt 2>&1
        tail -n1 out.txt 2>&1 | tee -a sum.txt
    done
    pass_cnt=`grep 'test_error 0.0' sum.txt | wc -l`
    printf "convolutional.py pass count = %d / $numtests\n", $pass_cnt
        cp -rf out.txt sum.txt $LOGDIR

    #Total Avg. Execution time : 10mins
        # Expected "final" output:
    #  convolutional.py pass count = 10 / 10
}
run_tutorials_image_mnist()
{
    echo "=======================tutorials_image_mnist==============="
    cd $MODELDIR/tutorials/image/mnist
        rm -f tutorials_image_mnist.txt
    /usr/bin/python3 ./convolutional.py 2>&1 | tee -a tutorials_image_mnist.txt
    cp -rf tutorials_image_mnist.txt $LOGDIR

        #Total Avg. Execution time : 20mins
    # Expected "final" output:
    #   Step 8500 (epoch 9.89), 7.2 ms
    #   Minibatch loss: 1.609, learning rate: 0.006302
    #   Minibatch error: 0.0%
    #   Validation error: 1.0%
    #   Test error: 0.8%
}
run_tutorials_image_cifar10()
{
    echo "=======================tutorials_image_cifar10==============="
        cd $MODELDIR/tutorials/image/cifar10
        rm -f tutorials_image_cifar10.txt
    /usr/bin/python3 ./cifar10_train.py --max_steps=5000 2>&1 | tee -a tutorials_image_cifar10.txt
    /usr/bin/python3 ./cifar10_eval.py --run_once=True 2>&1 | tee -a tutorials_image_cifar10.txt
    cp -rf tutorials_image_cifar10.txt $LOGDIR

        #Total Avg. Execution time : 30mins
    # Expected "final" output:
    #   2017-10-10 18:24:38.971358: precision @ 1 = 0.775
}
run_resnet_on_cifar10()
{
     echo "=======================resnet_on_cifar10==============="
        cd $MODELDIR/research/resnet
        rm -f resnet_on_cifar10.txt
    # Get the data
    curl -o cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
    tar -xzf cifar-10-binary.tar.gz
    ln -s ./cifar-10-batches-bin ./cifar10
    # Run it
    # NOTE:  here we're using the `timeout` command to limit the length of the run (as
    # there doesn't appear to be a flag for this).  So we're really measuring the accuracy
    # given X seconds of run time.
    timeout 480s /usr/bin/python3 ./resnet_main.py --train_data_path=cifar10/data_batch* \
                            --log_root=/tmp/resnet_model \
                            --train_dir=/tmp/resnet_model/train \
                            --dataset='cifar10' \
                            --num_gpus=1 2>&1 | tee -a resnet_on_cifar10.txt
    cp -rf resnet_on_cifar10.txt $LOGDIR

        #Total Avg. Execution time : 40mins
    # Expected "final" output:
    #   INFO:tensorflow:loss = 2.62153, step = 1, precision = 0.0703125
    #   INFO:tensorflow:global_step/sec: 1.53655
    #   INFO:tensorflow:loss = 1.77221, step = 101, precision = 0.359375
    #   INFO:tensorflow:global_step/sec: 1.55515
    #   INFO:tensorflow:loss = 1.50121, step = 201, precision = 0.5625
    #   INFO:tensorflow:global_step/sec: 1.55693
    #   INFO:tensorflow:loss = 1.33547, step = 301, precision = 0.609375
    #   INFO:tensorflow:global_step/sec: 1.55474
    #   INFO:tensorflow:loss = 1.31107, step = 401, precision = 0.578125
    #   INFO:tensorflow:global_step/sec: 1.55877
    #   INFO:tensorflow:loss = 1.16523, step = 501, precision = 0.726562
}
run_imagenet_classify()
{
    echo "=======================imagenet_classify==============="
        #  Details:  https://github.com/ROCmSoftwarePlatform/hiptensorflow/blob/hip-amd-nccl/tensorflow/g3doc/tutorials/image_recognition/index.md
    cd $MODELDIR/tutorials/image/imagenet
        rm -f imagenet_classify.txt
    /usr/bin/python3 ./classify_image.py 2>&1 | tee -a imagenet_classify.txt
        cp -rf imagenet_classify.txt $LOGDIR

    #Total Avg. Execution time : 15mins
        # Expected "final" output:
    #   giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca (score = 0.89107)
    #   indri, indris, Indri indri, Indri brevicaudatus (score = 0.00779)
    #   lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens (score = 0.00296)
    #   custard apple (score = 0.00147)
    #   earthstar (score = 0.00117)
}
run_slim_lenet()
{
    echo "=======================slim_lenet==============="
        cd $MODELDIR/research/slim
         rm -f slim_lenet.txt
    chmod u+x ./scripts/train_lenet_on_mnist.sh
    sed -i 's/python/python3/g' scripts/train_lenet_on_mnist.sh
    ./scripts/train_lenet_on_mnist.sh 2>&1 | tee -a slim_lenet.txt
        cp -rf slim_lenet.txt $LOGDIR

        #Total Avg. Execution time : 20mins
        # Expected "final" output:
        #2018-03-30 03:34:03.920083: I tensorflow/core/kernels/logging_ops.cc:79] eval/Accuracy[0.9894]
        #2018-03-30 03:34:03.920534: I tensorflow/core/kernels/logging_ops.cc:79] eval/Recall_5[1]
        #INFO:tensorflow:Finished evaluation at 2018-03-30-03:34:03
}
run_slim_cifarnet()
{
    echo "=======================slim_cifarnet==============="
        cd $MODELDIR/research/slim
    chmod u+x ./scripts/train_cifarnet_on_cifar10.sh
    sed -i 's/python/python3/g' scripts/train_cifarnet_on_cifar10.sh
    ./scripts/train_cifarnet_on_cifar10.sh  2>&1 | tee -a slim_cifarnet.txt
        cp -rf slim_cifarnet.txt $LOGDIR

          #Total Avg. Execution time : 40mins
        # Expected "final" output: Ran for 2000
       
#I0822 10:24:38.175445 140455272515328 learning.py:507] global step 1990: loss = 57.4365 (0.504 sec/step)
#I0822 10:24:43.236582 140455272515328 learning.py:507] global step 2000: loss = 57.4384 (0.507 sec/step)
#I0822 10:24:43.238091 140455272515328 learning.py:777] Stopping Training.
#I0822 10:24:43.238430 140455272515328 learning.py:785] Finished training! Saving model to disk.

}
run_tf_cnn_benchmarks()
{
    echo "=======================tf_cnn_benchmarks==============="
        cd $BENCHDIR
 #       cd $BENCHDIR/scripts/tf_cnn_benchmarks/
#       sed -i 's/import cPickle/import pickle/g' datasets.py
        cd $BENCHDIR
#     MODELS="alexnet"
    MODELS="alexnet googlenet inception3 inception4 lenet overfeat resnet50 resnet152_v2 trivial vgg11 vgg16 vgg19 resnet101 resnet50_v1.5"
	NGPUS=1
	ITERATIONS=500
#	BATCH_SIZE=( 1 2 4 8 16 32 64 128 256 )
        BATCH_SIZE=64

	for j in ${BATCH_SIZE[@]}
	do
	for i in ${MODELS[@]}
	do
    /usr/bin/python3 ./scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model=$i \
    --print_training_accuracy=True \
    --num_batches=${ITERATIONS} --variable_update=parameter_server --local_parameter_device=cpu \
    --num_gpus=${NGPUS} --batch_size=$j  2>&1 | tee $LOGDIR/tf-$i-$j.txt
    done
    done
 

    #grep -E "Model:|total images/sec" tf_cnn_benchmarks_log.txt
    #Total Avg. Execution time : 300mins
    # Expected "final" output:
    # (TITAN X results shown below)
    #   AlexNet     @   2500    images/sec: 2495.8 +/- 1.3 (jitter = 45.6)      6.178   0.006   0.029
    #   GoogleNet   @   2500    images/sec: 422.4 +/- 0.2 (jitter = 7.4)        3.647   0.031   0.156
    #   inception3  @   2500    images/sec: 123.1 +/- 0.1 (jitter = 1.5)        0.436   1.000   1.000
    #   lenet       @   2550    images/sec: 14362.7 +/- 22.4 (jitter = 827.9)   3.500   0.031   0.156
    #   overfeat    @   2500    images/sec: 541.1 +/- 0.2 (jitter = 8.9)        3.748   0.031   0.156
    #   resnet50    @   2500    images/sec: 190.8 +/- 0.0 (jitter = 0.9)        0.912   1.000   1.000
    #   trivial     @   2500    images/sec: 7949.9 +/- 7.3 (jitter = 354.5)     6.674   0.406   0.406
    #   vgg11       @   2500    images/sec: 250.6 +/- 0.1 (jitter = 2.0)        4.457   0.031   0.094
}

run_flower_alexnet()
{
    echo "=======================flower-alexnet==============="
    mkdir -p /dockerx/flower-alexnet
    rm -rf /dockerx/flower-alexnet/*
    wget https://raw.githubusercontent.com/pramenku/rocm-scripts/master/preprocessing_factory.py
    cp -rf preprocessing_factory.py /root/models/research/slim/preprocessing/
       
        cd $MODELDIR/research/slim
     chmod u+x download_and_convert_data.py
    python3 download_and_convert_data.py --dataset_name=flowers --dataset_dir=/dockerx/flower-alexnet
   
     python3 train_image_classifier.py --train_dir=/tmp/flowers-models/alexnet --dataset_name=flowers --dataset_split_name=train --dataset_dir=/dockerx/flower-alexnet --model_name=alexnet_v2 --max_number_of_steps=2000 --batch_size=128 --num_clones=1 2>&1 | tee  flower-alexnet.txt

         cp -rf flower-alexnet.txt $LOGDIR

        #Total Avg. Execution time : 40mins
        # Expected "final" output: Ran for 2000
       
#I0822 10:24:38.175445 140455272515328 learning.py:507] global step 1990: loss = 57.4365 (0.504 sec/step)
#I0822 10:24:43.236582 140455272515328 learning.py:507] global step 2000: loss = 57.4384 (0.507 sec/step)
#I0822 10:24:43.238091 140455272515328 learning.py:777] Stopping Training.
#I0822 10:24:43.238430 140455272515328 learning.py:785] Finished training! Saving model to disk.

}

#download_tensorflow_models
#download_tensorflow_benchmarks

run_convolutional_quick_test
run_tutorials_image_mnist
run_tutorials_image_cifar10
run_resnet_on_cifar10
run_imagenet_classify
run_slim_lenet
run_slim_cifarnet
run_flower_alexnet
#run_tf_cnn_benchmarks


cd $LOGDIR

cnn_bms=`grep -E "Model:|total images/sec" tf_cnn_benchmarks_log.txt | wc -l`
soumith=`grep 'across 100 steps' output*.log | wc -l`
slim=`grep 'Finished evaluation' slim_*.txt | wc -l`
classify=`grep 'earthstar' imagenet_classify.txt | wc -l`
resnet_cifar10=`grep 'precision' resnet_on_cifar10.txt | wc -l`
image_cifar10=`grep 'precision' tutorials_image_cifar10.txt | wc -l`
image_mnist=`grep 'Test error' tutorials_image_mnist.txt | wc -l`
conv_test=`grep 'test_error' sum.txt | wc -l`

echo "[STEPS]" > Results.ini
echo "Number=7" >> Results.ini
echo " " >> Results.ini

echo "[STEP_001]" >> Results.ini
echo "Description= convolutional_quick_test" >> Results.ini
if [ $conv_test -eq 10 ]; then
  echo "Status=Passed" >> Results.ini
else
  echo "Status=Failed" >> Results.ini
fi

echo "[STEP_002]" >> Results.ini
echo "Description= tutorials_image_mnist" >> Results.ini
if [ $image_mnist -eq 1 ]; then
  echo "Status=Passed" >> Results.ini
else
  echo "Status=Failed" >> Results.ini
fi

echo "[STEP_003]" >> Results.ini
echo "Description= tutorials_image_cifar10" >> Results.ini
if [ $image_cifar10 -eq 1 ]; then
  echo "Status=Passed" >> Results.ini
else
  echo "Status=Failed" >> Results.ini
fi

echo "[STEP_004]" >> Results.ini
echo "Description= resnet_on_cifar10" >> Results.ini
if [ $resnet_cifar10 -ge 3 ]; then
  echo "Status=Passed" >> Results.ini
else
  echo "Status=Failed" >> Results.ini
fi

echo "[STEP_005]" >> Results.ini
echo "Description= imagenet_classify" >> Results.ini
if [ $classify -eq 1 ]; then
  echo "Status=Passed" >> Results.ini
else
  echo "Status=Failed" >> Results.ini
fi

echo "[STEP_006]" >> Results.ini
echo "Description= slim tests" >> Results.ini
if [ $slim -eq 2 ]; then
  echo "Status=Passed" >> Results.ini
else
  echo "Status=Failed" >> Results.ini
fi

#echo "[STEP_007]" >> Results.ini
#echo "Description= soumith bms" >> Results.ini
#if [ $soumith -eq 8 ]; then
#  echo "Status=Passed" >> Results.ini
#else
#  echo "Status=Failed" >> Results.ini
#fi

echo "[STEP_007]" >> Results.ini
echo "Description= cnn_bms" >> Results.ini
if [ $cnn_bms -eq 20 ]; then
  echo "Status=Passed" >> Results.ini
else
  echo "Status=Failed" >> Results.ini
fi

cp -rf Results.ini ../


