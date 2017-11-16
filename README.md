# CapsNet

## Geoffrey Hinton's paper 
* [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)  
* [MATRIX CAPSULES WITH EM ROUTING](https://openreview.net/pdf?id=HJWLfGWRb)


## Result

#### 1. Reconstruction) 
> Just run runner.recons_random() as show in Run  

![Reconstruction](./readme/result_no_y.bmp)

#### 2. Paper 5.1: What the individual dimensions of a capsule represent
> Just run runner.recons_random_slow() as show in Run  

**Each row shows there construction when one of the 16 dimensions in the DigitCaps representation is tweaked by intervals of 1/63 in the range [−0.5,0.5], the leftmost col was not tweaked.**  
![Reconstruction](./readme/random_2_0.01587.bmp)


## Run
![main](./readme/main.png)


## Data
* mnist

## Reference
* [naturomics/CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow)
