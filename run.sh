#!/bin/bash
./run_mathlibs-docker.sh
./tf_tests-1gpu.sh
./tf-fp16-1gpu.sh
./alexnet-flower_1gpu.sh
./run_frameworks-docker.sh
#./7sk.sh
