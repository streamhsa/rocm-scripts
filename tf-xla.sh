#!/bin/bash
current=`pwd`

LOGDIR=/dockerx
BENCHDIR=/root/benchmarks
BASEDIR=/root
#BUILD=346

download_tensorflow_benchmarks()
{
        cd $BASEDIR
        rm -rf benchmarks
        git clone -b cnn_tf_v1.12_compatible https://github.com/tensorflow/benchmarks
}

run_tf_xla_concat_gpu()
{
cd /root/tensorflow
bazel test --compilation_mode=dbg   --cache_test_results=no --config=opt --config=rocm --test_tag_filters=-no_gpu,-benchmark-test,-no_oss,-no_rocm //tensorflow/compiler/xla/tests:concat_test_gpu  --test_output=streamed --verbose_test_summary --verbose_failures 2>&1 | tee $LOGDIR/test-concat_gpu-xla-$BUILD.log

}
:
run_tf_xla_cnn_bm()
{
    echo "=======================tf_cnn_benchmarks==============="
        cd $BENCHDIR
#       sed -i 's/import cPickle/import pickle/g' datasets.py
        cd $BENCHDIR
#     MODELS="alexnet"
    #MODELS="alexnet googlenet inception3 inception4 lenet overfeat resnet50 resnet152_v2 trivial vgg11 vgg16 vgg19 resnet101 resnet50_v1.5"
        MODELS="resnet50 resnet101 vgg19"
		NGPUS=1
 #       ITERATIONS=50
#        BATCH_SIZE=( 1 2 4 8 16 32 64 128 )
#        BATCH_SIZE=64

 #       for j in ${BATCH_SIZE[@]}
 #       do
        for i in ${MODELS[@]}
        do
    /usr/bin/python3 ./scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model=$i \
    --print_training_accuracy=True \
    --num_gpus=${NGPUS} --xla 2>&1 | tee $LOGDIR/tfxla-$i.txt
  #  done
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


download_tensorflow_benchmarks

run_tf_xla_concat_gpu
run_tf_xla_cnn_bm


