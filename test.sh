#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Skip tests in environments where they are not built/applicable
if [[ "${BUILD_ENVIRONMENT}" == *-android* ]]; then
  echo 'Skipping tests'
  exit 0
fi

# Find where cpp tests and Caffe2 itself are installed
if [[ "$BUILD_ENVIRONMENT" == *cmake* ]]; then
  # For cmake only build we install everything into /usr/local
  cpp_test_dir="$INSTALL_PREFIX/cpp_test"
  ld_library_path="$INSTALL_PREFIX/lib"
else
  # For Python builds we install into python
  # cd to /usr first so the python import doesn't get confused by any 'caffe2'
  # directory in cwd
  python_installation="$(dirname $(dirname $(cd /usr && $PYTHON -c 'import os; import caffe2; print(os.path.realpath(caffe2.__file__))')))"
  caffe2_pypath="$python_installation/caffe2"
  cpp_test_dir="$python_installation/torch/test"
  ld_library_path="$python_installation/torch/lib"
fi

################################################################################
# C++ tests #
################################################################################

################################################################################
# Python tests #
################################################################################
if [[ "$BUILD_ENVIRONMENT" == *cmake* ]]; then
  exit 0
fi


# Collect additional tests to run (outside caffe2/python)
EXTRA_TESTS=()

# CUDA builds always include NCCL support
if [[ "$BUILD_ENVIRONMENT" == *-cuda* ]]; then
  EXTRA_TESTS+=("$caffe2_pypath/contrib/nccl")
fi

rocm_ignore_test=()
if [[ $BUILD_ENVIRONMENT == *-rocm* ]]; then
  # Currently these tests are failing on ROCM platform:

  # Unknown reasons, need to debug
  rocm_ignore_test+=("--ignore $caffe2_pypath/python/operator_test/piecewise_linear_transform_test.py")
  rocm_ignore_test+=("--ignore $caffe2_pypath/python/operator_test/checkpoint_test.py")

  # On ROCm, RCCL (distributed) development isn't complete.
  # https://github.com/ROCmSoftwarePlatform/rccl
  rocm_ignore_test+=("--ignore $caffe2_pypath/python/data_parallel_model_test.py")
  rocm_ignore_test+=("--ignore $caffe2_pypath/python/dataio_test.py") 
fi

# NB: Warnings are disabled because they make it harder to see what
# the actual erroring test is
echo "Running Python tests.."
if [[ "$BUILD_ENVIRONMENT" == *py3* ]]; then
  # locale setting is required by click package with py3
  export LC_ALL=C.UTF-8
  export LANG=C.UTF-8
fi
pip install --user pytest-sugar
"$PYTHON" \
  -m pytest \
  -x \
  -v \
  --disable-warnings \
  --junit-xml="$pytest_reports_dir/result.xml" \
  --ignore "$caffe2_pypath/python/test/executor_test.py" \
  --ignore "$caffe2_pypath/python/operator_test/matmul_op_test.py" \
  --ignore "$caffe2_pypath/python/operator_test/pack_ops_test.py" \
  --ignore "$caffe2_pypath/python/mkl/mkl_sbn_speed_test.py" \
  ${rocm_ignore_test[@]} \
  "$caffe2_pypath/python" \
  "${EXTRA_TESTS[@]}"

#####################
# torchvision tests #
#####################
if [[ "$BUILD_ENVIRONMENT" == *onnx* ]]; then
  pip install -q --user git+https://github.com/pytorch/vision.git
  pip install -q --user ninja
  # JIT C++ extensions require ninja, so put it into PATH.
  export PATH="/var/lib/jenkins/.local/bin:$PATH"
  if [[ "$BUILD_ENVIRONMENT" == *py3* ]]; then
    pip install -q --user onnxruntime
  fi
  "$ROOT_DIR/scripts/onnx/test.sh"
fi
