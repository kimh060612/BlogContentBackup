---
title: FastSLAM Part.A
tags:
  - Robotics
  - SLAM
categories:
  - SLAM
mathjax: true
---

## Fast SLAM paper review

### Index

1. Abstract
2. Introduction
3. SLAM Problem Definition
4. FastSLAM with Known Correspondences  
   1. Factored Representation
   2. Particle Filter Path Estimation
   3. Landmark Localization Estimation
   4. Calculating Importance Weights
   5. Efficient Implementation
   6. Data Association
5. Experimental Results
6. Conclusion

### Abstract

SLAM 알고리즘은 자율주행 로봇에 꼭 필요한 알고리즘이다. 하지만 현실에 가까운 Environment에서는 많은 landmark를 필요로 하는데, 여태까지의 알고리즘은 여기까지 Scale-up하는데 무리가 있다. 예를 들어, Kalman filter 기반의 알고리즘의 time complexity는 각 센서에서 감지한 landmark의 개수에 제곱에 비례한다.  
본 논문에서는 FastSLAM 알고리즘을 제시한다. 해당 알고리즘은 재귀적으로 robot pose와 landmark의 위치에 대한 full posterior distribution을 landmark의 개수에 log 값에 비례하는 시간 복잡도로 추정한다. 이 알고리즘은 5만개의 landmark들에 대해서 성공적으로 동작을 했다.

### Introduction

본 논문 이전의 SLAM의 경우에는 대체적으로 EKF를 활용하여 robot의 pose에 대한 landmark의 위치의 posterior distribution을 구하는 것이었다. 하지만 이러한 방식에는 큰 단점이 있었는데, 바로 시간 복잡도가 landmark 점의 개수의 제곱에 비례한다는 것이다. 이러한 시간 복잡도 때문에 scale up된 환경에서 SLAM을 구동시키는 것이 매우 어려워졌다. 따라서 이번 논문에서는 bayesian 기법을 활용하여 

<p align="center"><img src="https://kimh060612.github.io/img/BayesianSLAM.png" width="100%"></p>

### SLAM Problem Definition

SLAM Problem은 Probabilistic Markov Chain으로써 가장 잘 표현된다. 로봇의 시간 $t$에서의 pose가 $s_t$라고 하겠다.  
로봇의 pose $s_t$는 로봇의 위치와 로봇이 향하고 있는 방향으로 표현된다. 
이때, 확률의 법칙으로 설명했을때, *motion model*을 주로 표현할때는 다음과 같이 표현한다. 

$$
p(s_t | u_t, s_{t - 1})
\tag{definition 1}
$$

$s_t$는 odometry $u_t$와 이전 pose $s_{t-1}$을 변수로 가지는 확률 함수이다.   
로봇의 환경은 $K$개의 immobile한 landmark를 가지고 있다. 각각의 Landmark point들은 위치로써 특정이 되며, 다음과 같이 표기하도록 하겠다.

$$
\theta_k, k = 1,2,3, \dots , K
\tag{definition 2}
$$





### FastSLAM with Known Correspondences  