#!/bin/bash

set -ex

install_ubuntu() {
    apt-get -y update
    apt-get install -y wget unzip
    apt-get install -y libopenblas-dev

    # Need the libc++1 and libc++abi1 libraries to allow torch._C to load at runtime
    apt-get install libc++1
    apt-get install libc++abi1


 apt-get clean
 rm -rf /var/lib/apt/lists/*


mkdir /root/roc-master && cd /root/roc-master
current=`pwd`
cd $current
#wget -np -nd -r -l 1 -A deb  http://10.216.151.220/artifactory/rocm-osdb-deb/compute-roc-master-10820/
wget -np -nd -r -l 1 -A deb http://10.216.151.220/artifactory/rocm-osdb-deb/compute-rocm-dkms-no-npi-410/

wget http://13.82.220.49/rocm/apt/debian/pool/main/r/rocm-profiler/rocm-profiler_5.6.7262_amd64.deb
wget http://13.82.220.49/rocm/apt/debian/pool/main/c/cxlactivitylogger/cxlactivitylogger_5.6.7259_amd64.deb
dpkg -i hsakmt-roct*.deb hsakmt-roct-dev*.deb
dpkg -i hsa-ext-rocr-dev*.deb hsa-rocr-dev*.deb
dpkg -i rocr_debug_agent*.deb


dpkg -i rocm-opencl*.deb rocm-opencl-dev*.deb rocm-smi*.deb rocm-utils*.deb rocminfo*.deb rocm-clang-ocl*.deb hip_*.deb hcc*.deb hsa-amd-aqlprofile*.deb rocprofiler-dev*.deb rocm-cmake*.deb rocm-device-libs*.deb comgr*.deb rocm-profiler*.deb cxlactivitylogger*.deb miopengemm*.deb MIOpen-HIP*.deb rocblas*.deb hipblas*.deb rocsparse*.deb hipsparse*.deb rocrand*.deb rocfft*.deb rocprim*.deb hipcub*.deb rocthrust*.deb rccl*.deb

rm -rf $current/*

cd $current
wget https://phoenixnap.dl.sourceforge.net/project/half/half/1.12.0/half-1.12.0.zip
unzip *.zip
cp -rf $current/include/half.hpp /opt/rocm/include

#git clone --recurse-submodules https://github.com/ROCmSoftwarePlatform/Thrust
#rm -rf /opt/rocm/include/thrust
#cp -rf Thrust/thrust /opt/rocm/include/

apt-get clean all
rm -rf /tmp/* 

}

install_centos() {


  yum update -y
  yum install -y wget
  yum install -y openblas-devel

  yum install -y epel-release
  yum install -y dkms kernel-headers-`uname -r` kernel-devel-`uname -r`

  echo "[ROCm]" > /etc/yum.repos.d/rocm.repo
  echo "name=ROCm" >> /etc/yum.repos.d/rocm.repo
  echo "baseurl=http://repo.radeon.com/rocm/yum/rpm/" >> /etc/yum.repos.d/rocm.repo
  echo "enabled=1" >> /etc/yum.repos.d/rocm.repo
  echo "gpgcheck=0" >> /etc/yum.repos.d/rocm.repo

  yum update -y

  yum install -y \
                   rocm-dev \
                   rocm-libs \
                   rocm-utils \
                   rocfft \
                   miopen-hip \
                   miopengemm \
                   rocblas \
                   rocm-profiler \
                   cxlactivitylogger \
                   rocsparse \
                   hipsparse \
                   rocrand \
                   rccl

  pushd /tmp
  rpm -i --replacefiles ./hcc-1.2.18473-Linux.rpm ./hip_base-1.5.18462.rpm ./hip_hcc-1.5.18462.rpm ./hip_doc-1.5.18462.rpm ./hip_samples-1.5.18462.rpm
  popd

  # Cleanup
  yum clean all
  rm -rf /var/cache/yum
  rm -rf /var/lib/yum/yumdb
  rm -rf /var/lib/yum/history

  # Needed for now, will be replaced once hip-thrust is packaged for CentOS


}
 
# Install Python packages depending on the base OS
if [ -f /etc/lsb-release ]; then
  install_ubuntu
elif [ -f /etc/os-release ]; then
  install_centos
else
  echo "Unable to determine OS..."
  exit 1
fi
