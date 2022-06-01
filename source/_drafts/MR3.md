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
    \cos \theta & -\sin \theta \\
    \sin \theta & \cos \theta 
\end{bmatrix}
$$

이때 이 P는 회전 행렬의 한 예시이다. 나중에 한 챕터를 다 써서 이 회전 행렬에 대해서 자세히 알아볼 것이다.

우선, 회전 행렬에 대해서 자세히 알아보기 전에 reference frame을 통해서 body frame을 표현하는 방법을 알아보자.   
이것이 왜 필요한지를 설명해 보자면, 우리가 가장 수학적으로 표현하기 쉬운 reference frame을 통해서 상대적으로 표현하기 어려운 body frame을 치환할 수 있다면 수학적으로 robot의 움직임을 표현하기 상당히 수월해질 것이다. 또한, 이 방법으로 좌표계 끼리의 자유로운 표현 방법을 익혀서 나중에는 굳이 reference frame이 아니더라도 상대적으로 표현하기 어려운 frame을 쉬운 frame으로 옮겨서 처리하는 것이 가능할 것이다.

사족이 길었는데, 다음 그림을 보면서 설명하도록 하겠다.

<p align="center"><img src="https://kimh060612.github.io/img/Chap3-Fig2.png" width="100%"></p>

다음 그림을 보면 위 말이 어느 정도는 납득이 갈 것이다.  
우리는 frame $\{b\}$와$\{c\}$ 를 reference frame $\{s\}$를 통해서 표현할 수 있다.

이때, 표현에 필요한 정보는 2가지가 있다.

1. reference frame에 비해 얼마나 회전했는지를 나타내는 회전 행렬
2. reference frame에 대해서 얼마나 떨어져 있는지를 나타내는 위치 벡터

이렇게 2가지를 알고 있다면 하나의 frame을 표현할 수 있으므로, 우리는 이를 묶어서 하나의 tuple 형식으로 생각할 수 있다. 위 그림에서 highlight가 되어 있는 것처럼 말이다.

자, 그렇다면 우리는 한번 위 Fig 3.4에서 나온 정보들을 하나 하나 파해쳐 보자.

위 그림에서 나온 $(Q, q)$에 대해서 먼저 설명해 보자면, frame $\{b\}$를 기준으로 frame $\{c\}$를 표현하는 tuple이다.  
그렇다면, reference frame $\{s\}$에서 다음이 성립할 것이다.

$$
\vec{r} = P\vec{q} + \vec{p} \\  
R = PQ

\tag{definition 2}
$$

보는 것과 같이, 행렬 $P$는 단순히 body frame을 나타내는 역할 뿐만 아니라 body frame위의 벡터를 reference frame 위로 끌고 오는 역할도 한다.

위 그림을 그대로 예시로 들어보면 $(Q, q)$를 다음과 같이 표현할 수 있을 것이다.

$$
\vec{q} = \begin{bmatrix}
           -1 \\
           -1 
         \end{bmatrix}_{\bold{b}}
$$

$$
Q = 
\begin{bmatrix}
    \cos \psi & -\sin \psi \\
    \sin \psi & \cos \psi 
\end{bmatrix},
\psi = 90\degree
$$

또한, $(P, p)$를 다음과 같이 표현 가능할 것이다.

$$
\vec{p} = \begin{bmatrix}
           2 \\
           2 
         \end{bmatrix}_{\bold{s}}
$$

$$
P = 
\begin{bmatrix}
    \cos \theta & -\sin \theta \\
    \sin \theta & \cos \theta 
\end{bmatrix},
\theta = 90\degree
$$

그렇다면 위 식이 납득이 갈 것이다. $90\degree + 90\degree$를 회전해서 $\phi = 180\degree$가 될 것이고, $\vec{q}$를 $90\degree$회전 시켜서 $\{s\}$로 끌고 와서 reference frame $\{s\}$에 맞추면 딱 $(R, r)$을 표현할 수 있다.

왜 라디안이 아니고 degree로 표현하냐고? ~~~(내맘이야 썅)~~~
 
이러한 tuple $(P, p)$를 통해서 우리는 다음과 같은 것들이 가능하다.

1. $\{s\}$를 통한 강체의 configuration의 표현
2. $\{s\}$로의 좌표계 변환
3. vector나 frame의 displacement

definition 2의 방정식은 흔히 rigid-body displacement라고도 불리는데, 여기서 굳이 displacement라는 단어를 쓰는 이유가 있다.

displacement의 단어의 뜻이 원래 "강제 철거"라는 뉘양스인데, 무슨 뜻인지 감각적으로 이해를 할 수 있지만 완전히 대체할 한국 단어가 마땅치 않아서 이 단어를 채택하도록 한다.

그리고 위 지식을 바탕으로 2차원 평면에서 강체의 움직임에 관해 일부 논의를 해보자.

<p align="center"><img src="https://kimh060612.github.io/img/Chap3-Fig3.png" width="100%"></p>

위 Fig 3.5에서 회색으로 묶여있는 것들을 강체로 표현한 것이다. (a) 처럼 각 frame들에 대해서 이동에 대한 notation을 할 수 있지만, (b) 처럼 점 하나를 잡고 거기에 대해서 회전했다라고 표현할 수도 있다.   
이와 같은 운동의 형태를 *screw motion*이라고 한다.

이러한 screw motion은 단 3가지 물리량으로 운동을 표현할 수 있다는 장점이 있다. 회전 각(위 그림에서는 $\beta$), 중심점의 $x$, $y$ 좌표로 말이다.

이를 표현할 수 있는 자세한 notation인 exponential coordinates 같은 상위 개념들은 나중에 하나의 소 챕터로 제대로 다루도록 하고 여기서는 넘어가도록 하겠다.

### Rotations and Angular Velocities

#### Rotation Matrices

앞서, 우리는 회전 행렬을 2차원 공간에서 다루었다. 이제 회전 행렬을 3차원 공간에서 다루어 보도록 하겠다. 그 전에, 회전 행렬의 수학적인 특성에 대해서 먼저 짚고 넘어가자.

먼저 말해두겠지만, 수학적인 특성은 그냥 참고만 하라. 이 포스트는 수학 포스트가 아니다. 진짜 극단적으로는 수학적 파트는 안 읽고 넘어가도 좋다. 대신에 이러한 notation들에 익숙해야만 한다. 이 주제의 본질은 수학이 아니라 "어떻게 로봇의 움직임을 표현할 것인가"에 있다. 따라서 여기에 너무 매몰되서 본질을 잃지 말고 가볍게 "이런게 있구나" 하고 넘긴 다음에 본질을 이해하려고 애쓰자. 그때가서 모르는게 생겼다면 그때가서 이걸 다시 읽어도 전혀 늦지 않다.

서론이 길었다. 바로 시작하자.

먼저 회전 행렬의 특성을 정리하자면 다음과 같다.

0. 모든 원소는 실수이다.
2. 각 열벡터의 L2-norm은 1이다.
3. 각 열벡터끼리 내적하면 무조건 0이다.
4. $R^T = R^{-1}$이다. 즉, $R^TR = I$이다.
5. $\det R = 1$이다.

이러한 특성을 만족시키는 $\mathbb{R}^{3 \times 3}$ 행렬을 Special orthogonal group($SO(3)$)라고 부르도록 하겠다.

이를 표현함에 있어서 행렬 $R$이 $SO(3)$의 조건을 만족시키면 다음과 같이 표현한다.

$$
R \in SO(3)
$$

그리고 회전 행렬의 성질을 정리해보자.

1. closure: 두 $SO(3)$에 속하는 $A$, $B$가 있으면 $AB$ 또한 $SO(3)$에 속한다.
2. associativity: $(AB)C$ = $A(BC)$
3. identity element existence: $I \in SO(3)$, $AI = IA = A$
4. inverse element existence: $\exist A^{-1} \rightarrow AA^{-1} = A^{-1}A = I$

이렇게 4가지의 성질을 만족한다. 이때, 성질 4는 성질 5와 맏물려서 다음이 성립한다.

*$R \in SO(3)$일때, $R^{-1} = R^T$이며 두 행렬 모두 $SO(3)$에 속한다.*

나머지 성질들은 독자들이 알아서 찾아보기를 권한다. 물론, 책의 p.70~71에 자세히 적혀 있다.

그리고 이러한 회전 행렬을 어디에 사용하는가가 중요한 논점이다. 크게 다음 3가지 경우에 걸쳐서 사용할 수 있다.

1. 방향의 표현
2. frame의 변경
3. vector나 frame의 회전

어차피 1,2 는 정말 뻔한 내용이라서 여기에서는 다루지 않을 예정이다. 대신에 우리는 3번에 대한 논의를 조금 해보도록 하겠다.

<p align="center"><img src="https://kimh060612.github.io/img/Chap3-Fig4.png" width="100%"></p>

우리는 위 그림 Fig 3.8과 같이 축 $\hat{\omega}$에 대해서 $\theta$ 만큼 회전시키는 회전 행렬을 다음과 같이 표현하도록 하겠다.

$$
R = \text{Rot}(\hat{\omega}, \theta)
\tag{definition 3}
$$

그리고 frame $\{s\}$에 대한 frame $\{c\}$의 방향의 표현을 $R_{sc}$와 같이 하도록 하자.

이러한 회전 행렬은 다음과 같이 표현할 수 있다.

<p align="center"><img src="https://kimh060612.github.io/img/Chap3-Fig5.png" width="100%"></p>

(이걸 Latex로 쓰라고? 에바쎄바 ㅋㅋ)

어떠한가, 이를 실제로 사용하는 것은 조금 무식해 보인다. 나중에는 이를 grace하게 사용하기 위해서 exponential coordinates라는 것을 배울 것이다. 실제로는 "이런 방식으로 표현할 수 있다" 라는 정도로만 알고 넘어가자.  

그리고 premultiply되느냐, postmultiply되느냐에 따라서 결과는 완전히 달라진다. 이 점을 유의하자.

#### Angular Velocities

회전하는 정도를 회전 행렬로 알 수 있다면 시간에 따라 얼마나 회전하는지도 알아야만 한다(각속도). 그래야 "운동을 분석한다"라고 할 수 있기 때문이다. 물론, 나중에는 각가속도도 알아야 하는데, 그건 나중에 하도록 하고 먼저 각속도부터 알아보도록 하자.

<p align="center"><img src="https://kimh060612.github.io/img/Chap3-Fig6.png" width="100%"></p>

먼저 우리는 어떠한 방향으로 시간에 따라 얼마나 회전하는지를 알아야 한다.그러려면 회전하는 축을 알아야하고 그에 따라 얼마나 회전하는지를 알아야한다.

종합적으로, 각속도를 다음과 같이 표현할 수 있다.

$$
\bold{w} = \hat{w} \dot{\theta} \\
\dot{\theta} = \lim_{\Delta t \rightarrow \infty} \frac{\Delta \theta}{\Delta t}
$$

그리고 각 축이 어떤 속도로 움직이고 있는지는 다음과 같이 표현이 가능하다.

$$

\dot{\hat{\bold{x}}} = \bold{w} \times \hat{\bold{x}} \\
\dot{\hat{\bold{y}}} = \bold{w} \times \hat{\bold{y}} \\
\dot{\hat{\bold{z}}} = \bold{w} \times \hat{\bold{z}} 

\tag{definition 4}
$$

우리는 이러한 vector들을 고정된 reference frame 위에서 정의된 각속도 $w_{s}$에 대해서 시간에 따라 변화하고 있는 body frame의 회전 행렬 $R(t)$의 변화($\dot{R}(t)$)를 표현하고 싶은 것이다.

이때, 각속도 $w_s \in \mathbb{R}^3$이라는 것을 잊지 말자.
그리고 definition 4를 활용하면 다음이 성립한다.

$$
R(t) = [ r_1 \space \space \space r_2 \space \space \space r_3 ] 
\tag{definition 5}
$$

$$
\dot{r}_i = w_s \times r_i 
\tag{equation 1}
$$

$$
\dot{R}(t) = w_s \times R(t)
\tag{equation 2}
$$

결과적으로 equation 2에서 cross product를 제거하고 좀 더 편한 notation을 소개해 보도록 하겠다. $x \in \mathbb{R}^3$ 이고 $x = [x_1 \space x_2 \space x_3]$ 일때, 다음을 정의해보자.

$$

[x] = 
\begin{bmatrix*}[l]
    0 & -x_3 & x_2 \\
    x_3 & 0 & -x_1 \\
    -x_2 & x_1 & 0 \\
\end{bmatrix*}

\tag{definition 6}
$$

위 definition 6을 skew-symmetric matrix라고 한다. 이 행렬을 곱하는 것은 3차원 벡터에서 cross product를 하는 것과 같다.  
즉, 다음이 성립한다.

$$
\dot{R}(t) = w_s \times R(t) = [w_s]R(t) \\
\therefore [w_s] = \dot{R}R^{-1}
$$

이러한 skew-symmetric production은 다음과 같은 성질들이 있다.

1. $[x] = -[x]^T$
2. $R \in SO(3) \rightarrow$ $R[w]R^T = [Rw]$

이를 body frame의 관점에서 다시 한번 작성해볼 수 있다. 한번 유도해볼 것을 추천한다.

#### Exponential Coordinate Representation

이러한 


