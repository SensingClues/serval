# SERVAL

Sound Event Recognition-based Vigilance for Alerting and Localisation

Rework of the original setup in 2020, old code is archived in the folder 20200209-old-code.

## Installation

See the notebooks for more details.

We use [this](https://github.com/igor-panteleev/youtube-8m#running-on-your-own-machine) now

Old original [EARS project](https://github.com/karoldvl/EARS/tree/master/ears) for setup details. 

## Use docker with tensorflow
We use docker images on a GPU enabled host machine.

    docker run --gpus all -it -p 8888:8888 -v /home/hugo/data/git/serval:/tf/serval tensorflow/tensorflow:latest-gpu-py3-jupyter

    docker run --gpus all -it -p 8888:8888 -v /home/hugo/data/git/serval:/tf/serval tensorflow/tensorflow:1.15.2-gpu-py3-jupyter

We are using the latest tensorflow version 1.15.2 , so we do NOT use tensorflow 2.x yet.

## Description of application

please vistit our website [sensingclues/serval](https://sensingclues.com/serval/) for a description of the real-life application.

## Useful other info

To train classification model next resources have been used:

    [Google AudioSet](https://research.google.com/audioset/)
    [YouTube-8M model](https://github.com/google/youtube-8m)
    [Tensorflow vggish model](https://github.com/tensorflow/models/tree/master/research/audioset)
    [Original Device hive blog](https://www.iotforall.com/tensorflow-sound-classification-machine-learning-applications/)
    [Igor's Youtube8M startercode](https://www.iotforall.com/tensorflow-sound-classification-machine-learning-applications/)

You can try to train model with more steps/samples to get more accuracy.

### LICENCE
Copyright for portions of project SERVAL are held by 2017 Karol J. Piczak as part of project [EARS](https://github.com/karoldvl/EARS). All other copyright for project SERVAL are held by Sensing Clues 2017.

GPL-3.0 © Sensing Clues 2017 
