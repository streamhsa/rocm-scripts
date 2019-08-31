current=`pwd`
LOGDIR=$current
export PYTORCH_TEST_WITH_ROCM=1 
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
echo "========================= py-autograd====================="
python /root/pytorch/test/test_autograd.py -v 2>&1 | tee $LOGDIR/py-autograd.log
echo "========================= py-cuda====================="
python /root/pytorch/test_cuda.py -v 2>&1 | tee $LOGDIR/py-cuda.log	
echo "========================= py-dataloader ====================="
python /root/pytorch/test_dataloader.py	-v 2>&1 | tee $LOGDIR/py-dataloader.log
echo "========================= py-distributions====================="
python /root/pytorch/test_distributions.py -v 2>&1 | tee $LOGDIR/py-distributions.log
echo "========================= py-indexing====================="
python /root/pytorch/test_indexing.py -v 2>&1 | tee $LOGDIR/py-indexing.log
echo "========================= py-jit====================="
python /root/pytorch/test_jit.py -v 2>&1 | tee $LOGDIR/py-jit.log	
echo "========================= py-nn====================="
python /root/pytorch/test_nn.py	-v 2>&1 | tee $LOGDIR/py-nn.log	
echo "========================= py-optim===================="
python /root/pytorch/test_optim.py	-v 2>&1 | tee $LOGDIR/py-optim.log
echo "========================= py-sparse====================="
python /root/pytorch/test_sparse.py	-v 2>&1 | tee $LOGDIR/py-sparse.log	
echo "========================= py-torch====================="
python /root/pytorch/test_torch.py -v 2>&1 | tee $LOGDIR/py-torch.log
echo "========================= py-utils====================="
python /root/pytorch/test_utils.py -v 2>&1 | tee $LOGDIR/py-utils.log

echo "========================= py-test_cuda_primary_ctx====================="
python /root/pytorch/test_cuda_primary_ctx.py -v 2>&1 | tee $LOGDIR/test_cuda_primary_ctx.log
echo "========================py-test_indexing_cuda====================="
python /root/pytorch/test_indexing_cuda.py -v 2>&1 | tee $LOGDIR/test_indexing_cuda.log	

echo "========================= py-test_numba_integration ====================="
python /root/pytorch/test_numba_integration.py	-v 2>&1 | tee $LOGDIR/py-test_numba_integration.log

echo "========================= py-test_type_info====================="
python /root/pytorch/test_type_info.py -v 2>&1 | tee $LOGDIR/py-test_type_info.log

echo "========================= py-test_type_hints====================="
python /root/pytorch/test_type_hints.py -v 2>&1 | tee $LOGDIR/py-test_type_hints.log

echo "========================= py-test_expecttest====================="
python /root/pytorch/test_expecttest.py -v 2>&1 | tee $LOGDIR/py-test_expecttest.log	

echo "========================= py-test_docs_coverage====================="
python /root/pytorch/test_docs_coverage.py -v 2>&1 | tee $LOGDIR/py-test_docs_coverage.log	
