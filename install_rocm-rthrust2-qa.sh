#!/bin/bash

set -ex

install_ubuntu() {
    apt-get -y update
    apt-get install -y wget unzip
    apt-get install -y libopenblas-dev

    # Need the libc++1 and libc++abi1 libraries to allow torch._C to load at runtime
    apt-get install libc++1
    apt-get install libc++abi1

    sh -c 'echo deb [arch=amd64 trusted=yes] http://compute-artifactory.amd.com/artifactory/list/rocm-osdb-deb/ compute-rocm-dkms-no-npi 478 > /etc/apt/sources.list.d/rocm.list'

    apt-get update --allow-insecure-repositories

    DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
                   rocm-dev \
                   rocm-libs \
                   rocm-utils \
                   rocfft \
                   miopen-hip \
                   miopengemm \
                   rocblas \
                   rocsparse \
                   hipsparse \
                   rocrand \
		   hipcub \
                   rocthrust \
                   rccl


mkdir /root/roc-master && cd /root/roc-master
current=`pwd`
cd $current
wget http://13.82.220.49/rocm/apt/debian/pool/main/r/rocm-profiler/rocm-profiler_5.6.7262_amd64.deb
wget http://13.82.220.49/rocm/apt/debian/pool/main/c/cxlactivitylogger/cxlactivitylogger_5.6.7259_amd64.deb
dpkg -i *.deb

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
  yum install -y wget unzip
  yum install -y openblas-devel

  yum install -y epel-release
  yum install -y dkms kernel-headers-`uname -r` kernel-devel-`uname -r`

  echo "[ROCm]" > /etc/yum.repos.d/rocm.repo
  echo "name=ROCm" >> /etc/yum.repos.d/rocm.repo
  echo "baseurl=http://10.216.151.220/artifactory/rocm-osdb-rpm/compute-rocm-dkms-no-npi-478" >> /etc/yum.repos.d/rocm.repo
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
                   rccl \
                   hipcub \
                   rocthrust


mkdir /root/roc-master && cd /root/roc-master
current=`pwd`
cd $current
wget http://13.82.220.49/rocm/yum/rpm/cxlactivitylogger-5.6.7259-gf50cd35.x86_64.rpm
wget http://13.82.220.49/rocm/yum/rpm/rocm-profiler-5.6.7262-g93fb592.x86_64.rpm
rpm -i --replacefiles rocm-profiler-5.6.7262-g93fb592.x86_64.rpm cxlactivitylogger-5.6.7259-gf50cd35.x86_64.rpm

wget https://phoenixnap.dl.sourceforge.net/project/half/half/1.12.0/half-1.12.0.zip
unzip *.zip
cp -rf $current/include/half.hpp /opt/rocm/include

#git clone --recurse-submodules https://github.com/ROCmSoftwarePlatform/Thrust
#rm -rf /opt/rocm/include/thrust
#cp -rf Thrust/thrust /opt/rocm/include/

rm -rf /tmp/*


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
