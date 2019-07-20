#!/bin/sh
echo y | apt-get update
echo y | pip3 install scipy jupyter matplotlib tqdm pillow moviepy opencv-python pandas pygtp sgf nltk sox soundfile sklearn numpy
echo y | apt-get install python3-tk

current=/dockerx
mkdir -p $current/7sk

cd $current/7sk && git clone https://github.com/7SK/CapsNet-Tensorflow

cd $current/7sk/CapsNet-Tensorflow

python3 download_data.py

python3 main.py 2>&1 | tee $current/7sk/CapsNet-Tensorflow.log

python3 main.py --is_training=False 2>&1 | tee -a $current/7sk/CapsNet-Tensorflow.log

cp -rf results/test_acc.csv $current/7sk/ 

cd $current/7sk && git clone https://github.com/7SK/DCGAN-tensorflow

cd $current/7sk/DCGAN-tensorflow
python3 download.py mnist

python3 main.py --dataset mnist --input_height=28 --output_height=28 --train 2>&1 | tee  $current/7sk/DCGAN-tensorflow.log

python3 main.py --dataset mnist --input_height=28 --output_height=28 2>&1 | tee -a $current/7sk/DCGAN-tensorflow.log

 

cd $current/7sk && git clone https://github.com/7SK/ResNeXt-in-tensorflow

cd $current/7sk/ResNeXt-in-tensorflow

python3 cifar10_train.py --version=’test’ 2>&1 | tee $current/7sk/ResNeXt-in-tensorflow.log

 

cd $current/7sk && git clone https://github.com/7SK/models

cd $current/7sk/models/research/minigo

python3 minigo.py --board_size=9 --batch_size=256 2>&1 | tee $current/7sk/models-minigo.log

python3 minigo.py --board_size=9 --batch_size=256 –validation 2>&1 | tee -a $current/7sk/models-minigo.log

python3 minigo.py --board_size=9 --batch_size=256 –test 2>&1 | tee -a $current/7sk/models-minigo.log

cd $current/7sk/models/research/deep_speech

echo y | apt-get install sox libsox-fmt-mp3

./run_deep_speech.sh 2>&1 | tee $current/7sk/deep_speech.log
 

cd $current/7sk && git clone https://github.com/7SK/pix2pix-tensorflow

cd $current/7sk/pix2pix-tensorflow

python3 tools/download-dataset.py facades

python3 pix2pix.py  --mode train  --output_dir facades_train  --max_epochs 200  --input_dir facades/train --which_direction BtoA  2>&1 | tee $current/7sk/pix2pix-tensorflow.log

python3 pix2pix.py  --mode test  --output_dir facades_test  --input_dir facades/val  --checkpoint facades_train 2>&1 | tee $current/7sk/pix2pix-tensorflow.log

 

 echo y | pip3 uninstall jupyter
echo y | pip3 install ipython
cd $current/7sk && git clone https://github.com/7sk/fast-wavenet

cd $current/7sk/fast-wavenet

export PYTHONPATH=$(pwd)/wavenet:$PYTHONPATH

python3 demo.py 2>&1 | tee $current/7sk/fast-wavenet.log
