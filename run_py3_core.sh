#!/usr/bin/env bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================

set -e
set -x
#install_bootstrap_deb_packages.sh
apt-get update
apt-get install -y --no-install-recommends pciutils \
    apt-transport-https ca-certificates software-properties-common
apt-get clean
rm -rf /var/lib/apt/lists/*

#install_pip_packages.sh
#pip3 install --upgrade pip==18.1
pip3 install wheel==0.31.1
pip3 install --upgrade setuptools==39.1.0
pip3 install virtualenv
pip3 install --upgrade six==1.12.0
pip3 install future>=0.17.1
pip3 install --upgrade absl-py
pip3 install --upgrade werkzeug==0.11.10
pip3 install --upgrade bleach==2.0.0
pip3 install --upgrade markdown==2.6.8
pip3 install --upgrade protobuf==3.6.1
rm -rf /usr/lib/python3/dist-packages/six*
pip3 install --upgrade numpy==1.14.5
pip3 install --upgrade scipy==1.1.0
pip3 install scikit-learn==0.18.1
pip3 install pandas==0.19.2
pip3 install psutil
pip3 install py-cpuinfo
pip3 install pylint==1.6.4
pip3 install pycodestyle portpicker grpcio 
pip3 install --upgrade astor
pip3 install --upgrade gast
pip3 install --upgrade termcolor
pip3 install keras_applications==1.0.6 --no-deps
pip3 install keras_preprocessing==1.0.5 --no-deps
pip3 install --upgrade h5py==2.8.0
pip3 install tf-estimator-nightly==1.14.0.dev2019061801 --no-deps
pip3 install --upgrade argparse
#install_golang.sh
GOLANG_URL="https://storage.googleapis.com/golang/go1.10.linux-amd64.tar.gz"
sudo mkdir -p /usr/local
wget -q -O - "${GOLANG_URL}" | sudo tar -C /usr/local -xz


N_JOBS=$(grep -c ^processor /proc/cpuinfo)
N_GPUS=$(lspci|grep 'VGA'|grep 'AMD/ATI'|wc -l)

echo ""
echo "Bazel will use ${N_JOBS} concurrent build job(s) and ${N_GPUS} concurrent test job(s)."
echo ""

# Run configure.
export PYTHON_BIN_PATH=`which python3`
export CC_OPT_FLAGS='-mavx'

export TF_NEED_ROCM=1
export TF_GPU_COUNT=${N_GPUS}
export HIP_HIDDEN_FREE_MEM=320

#yes "" | $PYTHON_BIN_PATH configure.py

# Run bazel test command. Double test timeouts to avoid flakes.
bazel test --config=rocm --test_tag_filters=-no_oss,-oss_serial,-no_gpu,-no_rocm,-benchmark-test -k \
    --test_lang_filters=py --jobs=${N_JOBS} --test_timeout 600,900,2400,7200 \
    --build_tests_only --test_output=errors --local_test_jobs=${TF_GPU_COUNT} --config=opt \
    --test_sharding_strategy=disabled \
    --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute -- \
    //tensorflow/... -//tensorflow/compiler/... -//tensorflow/contrib/... \
