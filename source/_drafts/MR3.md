---
title: Modern Robotics Chapter 3) Rigid-Body Motions
categories: 
    - Robotics
tags: 
    - Physics
    - Robotics
---

## Modern Robotics Chapter 3) Rigid-Body Motions

*이 글은 Modern Robotics를 참조하여 작성된 글이며 여기 나오는 사진들의 출처는 대부분이 동일한 책에 있습니다.*

### Index

1. Introduction
2. Rigid-Body Motions in the plane
3. Rotations and Angular Velocities  
   1. Rotation Matrices
   2. Angular Velocities
   3. Exponential Coordinate Representation
4. Rigid-Body Motions and Twists  
   1. Homogeneous Transformation Matrices
   2. Twists
   3. Exponential Coordinate Representation of Rigid-Body Motions
5. Wrenches

### Introduction

*Modern robotics article*

앞으로 이 카테고리에는 Modern Robotics의 정리글이 올라갈 것이다. 편의상 Chapter 2는 따로 정리하지 않고, Chapter 3 부터 정리가 시작될 것이다. 예상 독자들은 선형대수학, 미분적분학, 미분방정식, 일반 물리를 어느 정도 알고 있다는 전제 하에 글을 작성할 것이며 고등학교 수준의 기하와 벡터는 어렵지 않게 받아들일 수 있다고 생각할 것이다.

이번 Chapter에서는 Robot의 움직임에 대해서 상세하게 논의할 것이다.    
robot의 움직임을 어떻게 다양한 좌표계의 관점에서 표현을 할 수 있는지와 그에 대한 수학적인 notation을 익히는 것이 이번 장의 주요 목적이다.   
우선 그러기 위해서는 몇가지 정의를 사전에 알려주고 넘어가도록 하겠다. 

우리는 어떠한 곳이라도 좋으니 절대적인 기준 원점이 있다고 전제를 깔고 가겠다. (예를 들어서 벽 모서리 같은 곳) 이러한 기준 원점을 기준으로 펼쳐지는 좌표계를 우리는 reference frame이라고 부르도록 하고 $\{s\}$라고 표현하도록 하자.

그리고 우리는 robot의 관점에서 각 질량점들을 표현할 좌표계를 하나 더 정의해야한다. 이 좌표계를 body frame이라고 부르고, $\{b\}$ 라고 표현하도록 하겠다. 이 좌표계의 원점은 robot의 무게 중심이든 어디든 좋다. 하지만 수학적/물리적으로 논리를 전개하기 좋은 곳을 원점으로 잡는것이 편할 것이다. (예: 무게 중심)

먼저 이렇게 notation을 깔고 갔을때, robot의 움직임을 어떻게 수학적으로 표현할 수 있는지를 논의할 것이다. 먼저 평면에서의 움직임을 어떻게 표현할 수 있는지를 알아보겠다. 그 다음에 수학적으로 robot의 움직임을 표현하는데 중요한 회전 행렬에 대해서 알아보고, 이를 이용하여 3차원에서 어떻게 robot의 움직임을 표현할 수 있는지를 알아볼 것이다.

여기서 중요한 대전제를 다시 한번 더 표현하도록 하겠다.

1. 모든 frame들은 stationary하다
2. 여기서 표현되는 모든 frame들은 right handed 기준이다. 

### Rigid-Body Motions in the plane

우선 평면에서의 움직임을 정의하기 전에, 2차원 공간에서 기준 좌표계의 단위 벡터의 표현부터 짚고 넘어가겠다. 

우선 앞으로 우리가 쓸 reference frame의 단위 벡터의 표현부터 정의하겠다.  
reference frame의 $x$-축: $\hat{\bold{x}}_{\bold{s}}$  
reference frame의 $y$-축: $\hat{\bold{y}}_{\bold{s}}$

그리고 동일하게, 우리가 쓸 body frame의 단위 벡터 표현은 다음과 같다.  
body frame의 $x$-축: $\hat{\bold{x}}_{\bold{b}}$  
body frame의 $y$-축: $\hat{\bold{y}}_{\bold{b}}$

<p align="center"><img src="https://kimh060612.github.io/img/Chap3-Fig1.png" width="100%"></p>

위 Fig 3.3을 보자. 여기서 표현되는 vector $p$를 표현하는데 있어서 우리는 $(p_x, p_y)$와 같은 표현이 익숙할 것이다. 하지만 이렇게 표현을 하게 된다면 우리는 어떤 frame 기준의 벡터인지를 알 수가 없다. 보통 일반 물리 같은 수업을 들었다면 다음과 같이 표현 방법도 알고 있을 것이다.

$$
\vec{p} = p_x\hat{\bold{i}} + p_y\hat{\bold{j}}
$$

우리도 이와 비슷하게 위에서 정의한 단위 벡터를 사용하여 벡터를 표현할 것이다.

$$
\vec{p} = p_x\hat{\bold{x}}_{\bold{s}} + p_y\hat{\bold{y}}_{\bold{s}}
$$

이렇게 표현하는 장점은 어떤 frame을 채택한 벡터인지를 명확하게 표현할 수 있다는 것이다. 

그리고 위의 Fig 3.3에서 또 볼만한 포인트는 body frame을 reference frame을 통해서 표현할 수 있다는 것이다. body frame이 reference frame 보다 $\theta$만큼 기울어져 있다고 하자. 그렇다면 다음과 같이 body frame을 표현할 수 있다.

$$

\hat{\bold{x}}_{\bold{b}} = \cos \theta \hat{\bold{x}}_{\bold{s}} + \sin \theta \hat{\bold{y}}_{\bold{s}} \\
\hat{\bold{y}}_{\bold{b}} = -\sin \theta \hat{\bold{x}}_{\bold{s}} + \cos \theta \hat{\bold{y}}_{\bold{s}} \\

\tag{definition 1}
$$

하지만 모든 벡터를 이따구로 표현하면 진짜 더럽게 수식이 길어질 것이다. 편의를 위해서 다음과 같은 notation으로 벡터를 표현하면 무조건 reference frame에 있는 벡터라고 가정하자.

$$
\vec{p} = \begin{bmatrix}
           p_{x} \\
           p_{y} 
         \end{bmatrix}
$$

그렇다면 body frame의 두 단위 벡터들은 다음과 같은 행렬 $P$로 표현이 가능하다.

$$
P =
\begin{bmatrix}
    \hat{\bold{x}}_{\bold{b}} \\
    \hat{\bold{y}}_{\bold{b}} 
\end{bmatrix}
=
\begin{bmatrix}
    \cos \theta \space \sin \theta \\
    -\sin \theta \space \cos \theta 
\end{bmatrix}
$$

이때 이 P는 회전 행렬의 한 예시이다. 나중에 한 챕터를 다 써서 이 회전 행렬에 대해서 자세히 알아볼 것이다.

우선, 회전 행렬에 대해서 자세히 알아보기 전에 reference frame을 통해서 body frame을 표현하는 방법을 알아보자.   
이것이 왜 필요한지를 설명해 보자면, 우리가 가장 수학적으로 표현하기 쉬운 reference frame을 통해서 상대적으로 표현하기 어려운 body frame을 치환할 수 있다면 수학적으로 robot의 움직임을 표현하기 상당히 수월해질 것이다. 또한, 이 방법으로 좌표계 끼리의 자유로운 표현 방법을 익혀서 나중에는 굳이 reference frame이 아니더라도 상대적으로 표현하기 어려운 frame을 쉬운 frame으로 옮겨서 처리하는 것이 가능할 것이다.

사족이 길었는데, 다음 그림을 보면서 설명하도록 하겠다.

<p align="center"><img src="https://kimh060612.github.io/img/Chap3-Fig2.png" width="100%"></p>

다음 그림을 보면 