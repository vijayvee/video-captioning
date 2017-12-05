# Automated Video Captioning using S2VT

## Introduction
This repository contains my implementation of a video captioning system. This system takes as input a **video** and generates a **caption** describing the event in the video. 

I took inspiration from [Sequence to Sequence -- Video to Text](https://vsubhashini.github.io/s2vt.html), a video captioning work proposed by researchers at the University of Texas, Austin.

## S2VT - Architecture and working

Attached below is the **architecture diagram of S2VT** as given in their [paper](http://www.cs.utexas.edu/users/ml/papers/venugopalan.iccv15.pdf).

![Arch_S2VT](images/Arch_S2VT.png)

The **working** of the system while generating a caption for a given video is represented below diagrammatically.

![S2VT_Working](images/S2VT.png)

## Sample results

Attached below are a few screenshots from caption generation for videos from the **validation set**.

![Result1](images/Res1.png)

![Result2](images/Res2.png)

# Dataset

Even though S2VT was trained on MSVD, M-VAD and MPII-MD, I have trained my system **only on MSVD**, which can be downloaded [here.](https://www.microsoft.com/en-us/download/details.aspx?id=52422)

# Demo

A **demo of my system** can be found [here](https://www.youtube.com/watch?v=tmLzgFdI7Xg)
