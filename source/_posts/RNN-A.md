---
title: Recurrent Neural Network 강의 내용 part.A
date: 2022-03-05 19:26:50
categories: 
    - Deep Learning
tags: 
    - Deep Learning
    - Mathematics
mathjax: true
---

*본 포스트는 Hands-on Machine learning 2nd Edition, CS231n, Tensorflow 공식 document를 참조하여 작성되었음을 알립니다.*

### Index  
  
1. Definition of Recurrent Neural Network(RNN)
2. Back Propagation of RNN
3. Partice

여기서는 1,2는 Part. A이고 3은 Part. B에서 다루도록 하겠다.
<br>

### Definition of Recurrent Neural Network(RNN)

-------------------------------------------------------

RNN(Recurrent Neural Network)은 시계열 데이터를 처리하는데 특화된 신경망 구조이다.
이는 전의 입력이 연속해서 다음 입력에 영향을 주는 신경망 구조로써, 같은 구조가 계속 순환되어 나타나기 때문에 이러한 이름이 붙어져 있다. 
다음 그림을 보자. 

<p align="center"><img src="https://kimh060612.github.io/img/RNN.png" width="100%"></p>

하지만 이 그림으로는 자세한 구조까지는 잘 모르겠다. 그래서 필자가 직접 그린 그림을 보면서 설명하도록 하겠다. 말해두겠지만 필자는 그림을 심각하게 못 그리니 양해 바란다.

<p align="center"><img src="https://kimh060612.github.io/img/RNND.png" width="100%"></p>

이처럼 각 RNN Cell에 FCNN의 Unit이 들어가 있는 형식으로 구현된다. 이의 Feed Forward 연산을 생각해 보면 정말 간단하다.

$$
z_t = f(z_{t - 1}W_{re} + x_tW_{in} + b)
\tag{equation 1}
$$

$equation\space 1$에서 각 변수들의 정의는 다음과 같다.

> Definition 1

* $z_t$: time step $t$에 unit의 값에 activation function에 넣은 값
* $W_{re}$: Reccurent 가중치
* $W_{in}$: input layer에서의 가중치
* $x_t$: input vector

그림으로 정리하면 대충 다음과 같다.

<p align="center"><img src="https://kimh060612.github.io/img/RNND2.png" width="100%"></p>

결국 이렇게 Sequential한 데이터 $x$에 대해서 Sequential한 output $y$를 뽑을 수 있는 것이다.


### Back Propagation of RNN

-------------------------------------------------------

RNN의 Back Propagation 방법은 다음의 2가지가 있다. 

* Back Propagation Through Time - (BPTT)
* Real Time Recurrent Learning (RTRL)

여기서는 BPTT만을 다루도록 하겠다. RNN을 그렇게 자세하게 다루지 않는 이유는 FCNN의 연장선 느낌이 강해서이고 또한 현대에서는 딱히 잘 쓰이지 않기 때문이다. 그냥 지적 유희를 위해서 또는 기본기를 잘 닦기 위해서로만 읽어주기를 바란다.

또는 AutoGrad 계열의 알고리즘을 공부하는 사람들은 미분 그래프를 사용한 AutoGrad이전에는 이런 식으로 역전파 알고리즘을 구현했었구나 라고 역사책 읽는 느낌으로 읽어주면 매우 감사하겠다. 

이러한 RNN의 Back Propagation을 진행하기 위해서는 다음을 구하면 된다.

$$
\begin{aligned}
    \frac{\partial E}{\partial w^{out}_{ij}} =\space ? \\
    \frac{\partial E}{\partial w^{re}_{ij}} = \space ? \\
    \frac{\partial E}{\partial w^{in}_{ij}} = \space?
\end{aligned}
\tag{definition 2}
$$

이제 위 미분들에 chain rule을 적용해 보자. 그렇다면 다음과 같이 표현할 수 있다.

$$
\frac{\partial E}{\partial w^{out}_{ij}} = \sum^T_{t = 1} \frac{\partial E}{\partial v^{out}_{i}} \frac{\partial v^{out}_{i}}{\partial w^{out}_{ij}}
\tag{equation 2}
$$

$$
\frac{\partial E}{\partial w^{re}_{ij}} = \sum^T_{t = 1} \frac{\partial E}{\partial u^{t + 1}_{i}} \frac{\partial u^{t + 1}_{i}}{\partial w^{re}_{ij}}
\tag{equation 3}
$$

$$
\frac{\partial E}{\partial w^{in}_{ij}} = \sum^T_{t = 1} \frac{\partial E}{\partial u^{in}_{i}} \frac{\partial u^{in}_{i}}{\partial w^{out}_{ij}}
\tag{equation 4}
$$

자, 그렇다면 여기서 delta를 정의해서 일반화된 delta 규칙을 적용해 보아야 Back Propagation이 효율적으로 될 것이다. 

그렇다면 각 미분들에 대한 delta는 다음과 같이 정의될 수 있다.

$$
\delta^{out}_{k,t} = \frac{\partial E}{\partial v^t_k} = \frac{\partial E}{\partial y^t_k} \frac{\partial y^t_k}{\partial v^t_k} = \frac{\partial E}{\partial y^t_k} f'(v^t_k)
\tag{definition 3}
$$

input layer에서의 가중치는 output layer처럼 FCNN과 완전히 같다. 굳이 적어두지는 않도록 하겠다. 그렇다면 남은 것은 recurrent layer에서의 delta이다. 전개해보면 대략 다음과 같이 표현할 수 있다.

<p align="center"><img src="https://kimh060612.github.io/img/RNNDBPTT.png" width="100%"></p>

이를 수식으로 표현하면 다음과 같이 표현할 수 있다.

$$
\delta^{re}_{j,t} = \sum^{n_{out}}_{i = 1} \frac{\partial E}{\partial v^t_i} \frac{\partial v^t_i}{\partial u^t_j} + \sum^{n_{re}}_{i = 1} \frac{\partial E}{\partial u^{t + 1}_i} \frac{\partial u^{t + 1}_i}{\partial u^t_j}
\tag{definition 4}
$$

각 Summation Term들의 delta를 이용해서 표현해 보면 다음과 같다.

$$
\delta^{re}_{j,t} = \sum^{n_{out}}_{i = 1} \delta^{out}_{i,t} \frac{\partial v^t_i}{\partial u^t_j} + \sum^{n_{re}}_{i = 1} \delta^{re}_{i, t} \frac{\partial u^{t + 1}_i}{\partial u^t_j}  \\ 
= \sum^{n_{out}}_{i = 1} \delta^{out}_{i,t} w^{out}_{ij} f'(u^t_j) + \sum^{n_{re}}_{i = 1} \delta^{re}_{i, t} w^{re}_{ij} f'(u^t_j)
\tag{equation 4}
$$

즉, 이와 같이 RNN Cell의 기본 형태의 BPTT는 정말 단순하게도 FCNN의 연장선이다. 별게 없다.

그리고 이의 변형판으로 시간을 분할해서 Update하는 방법도 있다. 이를 Truncated BPTT라고 하는데 관심이 있다면 찾아보도록 하자. 이것도 진짜 별거 없다.

그저 위의 미분에서 시간 term을 잘라서 update 해주면 된다.

이걸로 RNN 또한 끝이 났다. 다음 파트에서는 이를 구현해 보도록 하겠다.

