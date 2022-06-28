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

본 논문 이전의 SLAM의 경우에는 대체적으로 EKF를 활용하여 robot의 pose에 대한 landmark의 위치의 posterior distribution을 구하는 것이었다.  
하지만 이러한 방식에는 큰 단점이 있었는데, 바로 시간 복잡도가 landmark 점의 개수의 제곱에 비례한다는 것이다.   
이러한 시간 복잡도 때문에 scale up된 환경에서 SLAM을 구동시키는 것이 매우 어려워졌다. 따라서 이번 논문에서는 bayesian 기법을 활용하여 SLAM에 접근한다. 밑의 Figrue 1을 보면 SLAM을 잘 표현할 수 있는 generative probabilistic model(Dynamic Bayesian Network)가 그려져 있다.  

<p align="center"><img src="https://kimh060612.github.io/img/BayesianSLAM.png" width="100%"></p>

위의 Figure 1을 보면 SLAM 문제 자체가 중요한 conditional independency를 나타내고 있다는 것이 자명하다.  
대표적으로 robot path의 정보가 독립적으로 각각의 landmark의 measure를 생성한다는 점이다. 

이러한 점들을 종합해서 본 논문에서는 FastSLAM 이라는 알고리즘을 제시한다.  
이 알고리즘은 SLAM 문제를 다음과 같이 분해하여 해결한다. 

* robot localization problem
* collection of landmark estimation problem conditioned on robot pose estimation

해당 논문에서는 Particle Filter기법과 tree based data structure를 혼합하여 시간 복잡도를 $O(M\log K)$로 줄여냈다. 
이제부터 그 방법을 알아보도록 할 것이다. 

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

그리고 우리의 경우에는 robot이 감지할 수 있는 Landmark의 measurement의 개수는 1 timestep 당 1라고 가정한다. 이는 오로지 수학적인 편의성을 위한 것이다. 

Sensor measurement는 measurement model이라는 확률적인 법칙에 의해 지배된다. 이 model은 다음과 같이 표현할 수 있다.

$$
p(z_t | s_t, \theta, n_t)
\tag{definition 3}
$$

여기서 $\theta$의 경우, 위에서 언급한 landmark point의 집합이고, $n_t$의 경우 시간 $t$에서 로봇이 감지한 landmark의 index이다.   
여기서 $n_t$같은 경우에는 종종 *correspondence*라고도 불리운다. 이러한 landmark 같은 경우에는 서로 identifiable하다고 가정한다. 

일반적으로 이러한 정의들을 놓고 보았을때, SLAM은 모든 landmark의 location $\theta$와 로봇의 pose $s_t$를 measurements $z^t = z_1,\dots,z_t$와 odometry $u^t=u_1,\dots,u_t$를 통해서 결정하는 것이다.  
확률을 통해서 표기를 했을때, 이는 posterior distribution $p(s^t, \theta| z^t, u^t)$라고 할 수 있다. 

만약, correspondence가 알려져 있다면, SLAM은 더욱 간단하게 정의될 수 있다. 

$$
p(s^t, \theta| z^t, u^t, n^t)
\tag{definition 4}
$$

앞서 언급했듯이, 모든 landmark estimation 문제는 로봇의 경로 $s^t$와 correspondence $n^t$를 알고 있다면 서로 독립이 된다. 이러한 가정은 FastSLAM 알고리즘에서의 공리로써 다음 장에서 사용되게 된다. 

### FastSLAM with Known Correspondences  

