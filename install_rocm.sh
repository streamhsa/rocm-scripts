#!/bin/bash

set -ex

install_ubuntu() {
    apt-get update
    apt-get install -y wget
    apt-get install -y libopenblas-dev

    # Need the libc++1 and libc++abi1 libraries to allow torch._C to load at runtime
    apt-get install libc++1
    apt-get install libc++abi1

# Install required python2 packages
apt-get -y update --fix-missing --allow-insecure-repositories && DEBIAN_FRONTEND=noninteractive apt-get install -y git dkms cmake flex bison aria2 check xsltproc cifs-utils vim ssh dos2unix  python-pip-whl libpci3 libelf1 g++-multilib gcc-multilib libunwind-dev libnuma-dev bzip2 sudo file zip

 apt-get clean
 rm -rf /var/lib/apt/lists/*


mkdir /home/roc-master && cd /home/roc-master
current=`pwd`
#wget http://10.236.104.70/job/compute-roc-master/9969/artifact/artifacts/compute-roc-master-9969.tar.bz2
#wget http://172.27.226.104/artifactory/rocm-generic-local/amd/compute-psdb/compute-psdb-29641.tar.bz2
cd $current
wget -np -nd -r -l 1 -A deb  http://172.27.226.104/artifactory/rocm-osdb-deb/compute-roc-master-10175/
 #tar -jxvf compute*
#sed -i 's/compute-artifactory.amd.com/172.27.226.104/g' deb.meta4
#aria2c deb.meta4

#cd ./deb 

#cp -rf amd_comgr/*.deb hip/*.deb ocl_lc/*.deb  meta/rocm-utils*.deb  rocr_debug_agent/*.deb devicelibs/*.deb hcc/*.deb clang-ocl/*.deb hsa-amd-aqlprofile/*.deb rocm-cmake/*.deb rocminfo/*.deb rocm-smi/*.deb rocprofiler/*.deb rocr/*.deb rocr_ext/*.deb roct/*.deb $current

wget http://13.82.220.49/rocm/apt/debian/pool/main/r/rocm-profiler/rocm-profiler_5.6.7259_amd64.deb
wget http://13.82.220.49/rocm/apt/debian/pool/main/c/cxlactivitylogger/cxlactivitylogger_5.6.7259_amd64.deb
dpkg -i hsakmt-roct*.deb hsakmt-roct-dev*.deb
dpkg -i hsa-ext-rocr-dev*.deb hsa-rocr-dev*.deb
dpkg -i rocr_debug_agent*.deb


dpkg -i rocm-opencl*.deb rocm-opencl-dev*.deb rocm-smi*.deb rocm-utils*.deb rocminfo*.deb rocm-clang-ocl*.deb hip*.deb hcc*.deb  hsa-amd-aqlprofile*.deb rocprofiler-dev*.deb rocm-cmake*.deb rocm-device-libs*.deb comgr*.deb rocm-profiler*.deb cxlactivitylogger*.deb miopengemm*.deb MIOpen-HIP*.deb rocblas*.deb hipblas*.deb rocsparse*.deb hipsparse*.deb rocrand*.deb rocfft*.deb 

rm -rf $current/*

###########################Uncomment If you want to build HIP from Github##############
#cd $current
#git clone https://github.com/ROCm-Developer-Tools/HIP
#cd $current/HIP
#git reset --hard 9a5dc9fe24ca78976f9366303d336a005c90aabc
#patch -p1 src/hip_peer.cpp /root/driver/0001-Revert-Fixed-issue-of-GPU-device-losing-access-to-ho.patch
#echo y | ./install.sh

#########################################################

rm -rf ~/.cache/*

sh -c 'echo HIP_PLATFORM=hcc >> /etc/environment'
export HIP_PLATFORM=hcc
#sh -c 'echo HCC_AMDGPU_TARGET=gfx900 >> /etc/environment'

 echo "gfx900" > /opt/rocm/bin/target.lst
 echo "gfx906" >> /opt/rocm/bin/target.lst
# echo "gfx803" >> /opt/rocm/bin/target.lst

mkdir -p /opt/rocm/debians
curl http://13.82.220.49/rocm/apt/debian/pool/main/h/hip-thrust/hip-thrust_1.8.2_all.deb -o /opt/rocm/debians/hip-thrust.deb
dpkg -i /opt/rocm/debians/hip-thrust.deb


cd $current
wget https://phoenixnap.dl.sourceforge.net/project/half/half/1.12.0/half-1.12.0.zip
unzip *.zip
cp -rf $current/include/half.hpp /opt/rocm/include

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
  wget https://github.com/RadeonOpenCompute/hcc/releases/download/roc-1.9.2-pytorch-eap/hcc-1.2.18473-Linux.rpm
  wget https://github.com/ROCm-Developer-Tools/HIP/releases/download/roc-1.9.2-pytorch-eap/hip_base-1.5.18462.rpm
  wget https://github.com/ROCm-Developer-Tools/HIP/releases/download/roc-1.9.2-pytorch-eap/hip_doc-1.5.18462.rpm
  wget https://github.com/ROCm-Developer-Tools/HIP/releases/download/roc-1.9.2-pytorch-eap/hip_hcc-1.5.18462.rpm
  wget https://github.com/ROCm-Developer-Tools/HIP/releases/download/roc-1.9.2-pytorch-eap/hip_samples-1.5.18462.rpm
  rpm -i --replacefiles ./hcc-1.2.18473-Linux.rpm ./hip_base-1.5.18462.rpm ./hip_hcc-1.5.18462.rpm ./hip_doc-1.5.18462.rpm ./hip_samples-1.5.18462.rpm
  popd

  # Cleanup
  yum clean all
  rm -rf /var/cache/yum
  rm -rf /var/lib/yum/yumdb
  rm -rf /var/lib/yum/history

  # Needed for now, will be replaced once hip-thrust is packaged for CentOS
  git clone --recursive https://github.com/ROCmSoftwarePlatform/Thrust.git /data/Thrust
  rm -rf /data/Thrust/thrust/system/cuda/detail/cub-hip
  git clone --recursive https://github.com/ROCmSoftwarePlatform/cub-hip.git /data/Thrust/thrust/system/cuda/detail/cub-hip
  ln -s /data/Thrust/thrust /opt/rocm/include/thrust


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
