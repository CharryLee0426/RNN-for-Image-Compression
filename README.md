# Recurrent Neural Network for Image Compression

## 1. Introduction

This is the repo of our recurrent neural network for Image Compression optimized for human visual system. In this work, an architecture consist of a RNN-based encoder and decoder, a binarizer, and a neural network for entropy coding was designed. Compared with jpeg standard and some improved jpeg versions, the model can get 4.3%-8.8% AUC depending on the perceptual metric like MS--SSIM. The work shows that the trained model can outform JPEG at image compression across most bitrates on the rate-distortion curve on the Kodak dataset images no matter with or without the aid of entropy coding.

## 2. Prerequosites

1. The test script must run on linux, so linux/wsl is needed;
2. The train and test process depends on CUDA, so Nvdia GPU and CUDA are needed;
3. At least 25GB storage is needed if you want to train by yourself. MSCOCO 2017 was used for training in the work;

## 3. Test

## 4. Train

## 5. Results

## 6. Acknowledgement

## 7. References