---
title: Mathematical Statistics Chapter 2 Part. B
tags:
  - Mathematics
  - Statistics
categories:
  - Mathematical Statistics
mathjax: true
date: 2022-10-23 17:02:32
---


## Index

1. Introduction
2. Correlation Coefficient
3. Correlation for Multiple random variables
4. Transformation of Multiple random variables
5. linear combinaiton of random variables

## Introduction

이 장에서는 Multivariate Random Variable에서의 다양한 연산을 정의해볼 것이다. 상관 계수, 여러 변수에서의 변환, 랜덤 변수의 선형 결합 등등 여러 연산들을 정의할 것이다.

## Correlation Coefficient

이 장에서는 확률 변수들이 서로 의존일때 얼마나, 그리고 어떻게 그 변수들이 연관되어 있는지를 알아볼 것이다.   
이러한 의존성에는 어려가지 척도가 있다. 그 중에서 여기서는 $X$와 $Y$라는 확률 변수의 선형성을 측정하는 결합 분포의 모수 $\rho$를 소개할 것이다.   
이 절에서 다루는 모든 확률 변수는 기댓값이 있다고 가정한다.

이러한 모수를 다루기 전에 우리는 공분산이라는 것을 정의해볼 필요가 있다. 

$$
Cov(X,Y) = E[(X - \mu_X)(Y - \mu_Y)]
\tag{definition 1}
$$

위 식의 $Cov(X,Y)$를 우리는 $X$와 $Y$의 공분산이라고 한다. 위의 정의에서 $\mu_X$, $\mu_Y$는 각각 $X$와 $Y$의 기댓값이다.

위 식을 정리해서 다음과 같은 결과를 얻을 수도 있다.

$$
\begin{align}
    Cov(X, Y) &= E[XY - \mu_XY - \mu_YX + \mu_X\mu_Y] \\
    &= E[XY] - \mu_XE[Y] - \mu_YE[X] + \mu_X\mu_Y \\
    &= E[XY] - \mu_X\mu_Y
\end{align}
$$

위 식을 사용하는 것이 정의를 직접 활용해서 계산하는 것보다 종종 더 쉽기도 하다.   
그리고 위 공분산을 정규화하여 단위가 없는 척도를 만들 수도 있다. 이를 *상관계수*라고 한다.

$$
\rho = \frac{Cov(X,Y)}{\sigma_X\sigma_Y}
\tag{definition 2}
$$

위 정의에서 $\sigma_X, \sigma_Y$는 각각 $X$와 $Y$의 표준 편차이다.  
그리고 이러한 상관 계수에는 다음과 같은 성질이 있다.

1. 상관 계수 $\rho$가 존재하는 결합 분포를 가진 모든 확률 변수 $(X,Y)$에 대해서 $-1 \leq \rho \leq 1$ 이다.
2. $X,Y$가 독립인 확률변수이면 $Cov(X,Y) = 0$이고 따라서 상관 계수 $\rho = 0$이다. *이의 역은 성립하지 않지만 대우는 성립한다.* 
3. $(X, Y)$가 $X, Y$의 분산이 양수이고 유한할때 결합 분포를 가진다고 하자. 만약 $E[Y|X]$가 $X$에 대해서 선형이면 다음이 성립한다.
    $$
    \begin{align}
        E[Y|X] &= \mu_Y + \rho\frac{\sigma_Y}{\sigma_X}(X - \mu_X) \\
        E[Var(Y|X)] &= \sigma_Y^2(1 - \rho^2)
    \end{align}
    $$  

## Correlation for Multiple random variables

이 파트에서는 여러 확률 변수로 앞서 배운 내용을 확장해볼 것이다. 우선, 확률 변수 $n$개에 대한 공간을 정의해 보도록 하겠다.

> 표본 공간 $\mathcal{C}$에서의 확률 실험을 생각해보자. 확률 변수 $X_i$는 각 요소 $c \in \mathcal{C}$에 단 하나의 실수 $X_i(c) = x_i$를 부여한다고 하자. $(1 \leq i \leq n)$ 이에 대한 $n$ 차원 벡터를 $(X_1, \dotsb, X_n)$으로 표시하고 tuple로써 공간을 정의한다.

이러한 확률 벡터의 결합 분포에 대한 본질은 앞서 다룬 내용과 별반 다를게 없다. CDF, PDF, PMF의 성질도 그러하다. 이 파트에서는 이에 따른 특수한 정리를 다뤄볼 것이다. 

* 결합 분포의 적률 생성 함수(MGF)

    $-h_i \leq t_i \leq h_i$ $(i = 1,\dotsb, n)$에 대해서 다음이 존재한다고 하자.

    $$
    M(t_1, t_2, \dotsb, t_n) = E[e^{t_1X_1 + t_2X_2 + \dotsb+t_nX_n}]
    $$

    변수가 1~2개일때와 같이 위와 같은 MGF가 유일하게 결정된다.  
    그리고 이러한  MGF의 벡터 표현으로 다음과 같이 표현할 수 있다.

    $$
    \mathbf{t} \in B \subset \mathbb{R}^n \rightarrow M(\mathbf{t}) = E[e^{\mathbf{t}^T\mathbf{X}}]
    $$

    그리고 이 정리가 앞으로 유용할 것이다.

    정리) $X_1, X_2, \dotsb, X_n$을 상호 독립적인 확률 변수라고 하자. 그리고 모든 $i = 1,2,3,\dotsb,n$에 대하여 $-h_i \leq t_i \leq h_i$ 이고 $h_i > 0$일때 $X_i$는 MGF $M_i(t)$를 가진다고 하자. $k_1, k_2, \dotsb, k_n$이 상수일때, $T$를 다음과 같이 $X_i$들의 선형 결합이라고 정의해 보자.

    $$
    T = \sum_{i = 1}^n k_iX_i
    $$

    이러면 $T$는 다음과 같은 MGF를 가진다.

    $$
    M_T(t) = \prod_{i = 1}^n M_i(k_it), \quad -\min_i{h_i} < t_i < min_i{h_i}
    $$

    이의 따름 정리로 다음을 정의 가능하다.

    위의 따름정리) 만약 k_i = 1 이고, $X_i$가 $i$에 상관 없이 전부 공통의 MGF를 가지고 같은 범위의 $t$($h$에서)를 가지며 서로 I.I.D 일때, 다음이 성립한다.

    $$
    M_T(t) = [M(t)]^n, \quad -h < t < h
    $$

* 다변량 분산/공분산 행렬

    이 절에서는 선형 대수학을 모르는 사람들에게는 조금 힘든 절이 될 것이다. 필히 기초 선형 대수학을 공부하고 오도록 하자.

    일단, 앞으로 설명할 변수들의 Notation들 부터 설명하도록 하겠다.

    $$
    \mathbf{X} = 
    \begin{pmatrix}
        X_1 \\
        X_2 \\
        \vdots \\
        X_n
    \end{pmatrix}
    $$

    $\mathbf{X}$: $n$차 확률 벡터  
    $X_i$: $\mathbf{X}$의 원소  

    $$
    \begin{align}
        \mathbf{W} = [W_{ij}] \\
        \mathbf{W} \in \mathbb{R}^{m \times n}
    \end{align}
    $$

    $\mathbf{W}$: $m \times n$ 크기의 행렬  
    $W_{ij}$: $\mathbf{W}$의 $i$행, $j$열의 원소  

    위와 같은 표현에서 우리는 다음과 같이 기댓값 연산을 정의할 수 있다.

    $$
    E[\mathbf{X}] = 
    \begin{pmatrix}
        E[X_1] \\ 
        E[X_2] \\
        \vdots \\
        E[X_n]
    \end{pmatrix}
    $$

    이렇게 평균 벡터를 정의할 수 있다고 지난 챕터에서 정의했다. 그리고 평균 행렬 또한 같은 맥락으로 정의할 수 있다.

    $$
    \begin{align}
        E[\mathbf{W}] = [E(W_{ij})] \\
        E[\mathbf{W}] \in \mathbb{R}^{m \times n}
    \end{align}
    $$

    그리고 이에 따라서 기댓값 행렬/벡터의 연산은 행렬과 벡터의 연산의 성질과 기댓값 본연의 성질이 합쳐져서 다음과 같은 따름 정리가 성립한다.

    따름 정리) $\mathbf{W}_1, \mathbf{W}_2$라는 행렬이 있다고 가정하자. 두 행렬 모두 차원이 $\mathbb{R}^{m \times n}$이다.  
    또한, $\mathbf{A}_1, \mathbf{A}_2$를 $\mathbb{R}^{k \times m}$ 크기의 상수 행렬이라고 하고 행렬 $\mathbf{B}$를 크기가 $\mathbb{R}^{n \times l}$의 상수 행렬이라고 하자.

    그러면 다음이 성립한다.

    $$
    \begin{align}
        E[\mathbf{A}_1\mathbf{W}_1 + \mathbf{A}_1\mathbf{W}_2] &= \mathbf{A}_1E[\mathbf{W}_1] + \mathbf{A}_2E[\mathbf{W}_2] \\
        E[\mathbf{A}_1\mathbf{W}_1B] &= \mathbf{A}_1E[\mathbf{W}_1]B
    \end{align}
    $$

    자, 그렇다면 대망의 분산-공분산 행렬을 다음과 같이 정의할 수 있다.

    분산-공분산 행렬) $\mathbf{X} = (X_1, \dotsb, X_n)^T$을 $\sigma_i^2 = Var(X_i) < \infty$인 $n$차원 확률 벡터라고 하자. $\mathbf{X}$의 평균 벡터를 다음과 같이 $\mathbf{\mu}$라고 표현하자. 이에 대하여 분산-공분산 행렬은 다음과 같이 정의된다.

    $$
    Cov(\mathbf{X}) = E[(\mathbf{X} - \mathbf{\mu})(\mathbf{X} - \mathbf{\mu})^T] = [\sigma_{ij}]
    \tag{definition 4}
    $$

    위 행렬 $Cov(\mathbf{X})$에서 대각 성분은 $Var(X_i)$이고 비 대각 성분은 $Cov(X_i, X_j), (i \neq j)$이다.

    이러한 분산-공분산 행렬의 특징은 다음과 같다.

    > $\mathbf{X} = (X_1, \dotsb, X_n)^T$을 $\sigma_i^2 = Var(X_i) < \infty$인 $n$차원 확률 벡터라고 하자. $\mathbf{A}$가 상수의 $\mathbb{R}^{m \times n}$인 행렬이라고 하면 다음이 성립한다.

    $$
    \begin{align}
        Cov(\mathbf{X}) &= E[\mathbf{X}\mathbf{X}^T] - \mathbf{\mu}\mathbf{\mu}^T \\
        Cov(\mathbf{A}\mathbf{X}) &= \mathbf{A}Cov(\mathbf{X})\mathbf{A}^T
    \end{align}
    $$

    > 또한, 모든 분산-공분산 행렬은 양반정치(PSD, positive semi-definite) 행렬이다. 즉, 모든 벡터 $\mathbf{a} \in \mathbb{R}^n$에 대하여 $\mathbf{a}^TCov(\mathbf{X})\mathbf{a} \geq 0$이다.


## Transformation of Multiple random variables

자, 이제 진짜로 머리 터지는 부분에 돌입해보자. 우리는 이전 챕터에서 두 확률 변수가 주어졌을때, 한 변수로부터 다른 변수로의 분포 변환을 다루어 보았다. 이번에는 그것을 $n$차원으로 확장해서 다루어 보도록 하겠다. 

우선 $n$차원 공간 $\mathcal{S}$의 부분 집합 $A$에 대하여 다음과 같은 다중 적분을 고려해보자.

$$
\int \dotsb \int_A f(x_1, \dotsb, x_n)dx_1dx_2 \dotsb dx_n
$$

다음의 확률 변수와 역함수를 $\mathcal{S}$를 $y_1, \dotsb, y_n$이 있는 공간 $\mathcal{T}$로 시상하여 $A$를 집합 $B$로 시상하는 1:1 변환을 정의한다고 하자.

$$
\begin{align}
    y_1 = u_1(x_1, \dots, x_n), \dots, y_n = u_n(x_1, \dots, x_n) \\
    x_1 = w_1(y_1, \dots, y_n), \dots, x_n = w_n(y_1, \dots, y_n)
\end{align}
$$

역함수의 1계 편도함수는 모두 연속이고 다음 $n \times n$ 행렬식(야코비안이라고 한다, 정의 5번)이 $\mathcal{T}$에서 0이 아니라고 하면 다음을 얻을 수 있다.

$$
J = \begin{vmatrix}
    \frac{\partial x_1}{\partial y_1} \frac{\partial x_1}{\partial y_2} \dotsb \frac{\partial x_1}{\partial y_n} \\
    \frac{\partial x_2}{\partial y_1} \frac{\partial x_2}{\partial y_2} \dotsb \frac{\partial x_2}{\partial y_n} \\
    \vdots \quad \vdots \quad \qquad \vdots \\
    \frac{\partial x_n}{\partial y_1} \frac{\partial x_n}{\partial y_2} \dotsb \frac{\partial x_n}{\partial y_n} \\
\end{vmatrix}

\tag{definition 5}
$$


$$
\int \dotsb \int_A f(x_1, \dotsb, x_n)dx_1dx_2 \dotsb dx_n \\
= \int \dotsb \int_B f[w_1(y_1, \dots, y_n), \dotsb, w_n(y_1, \dots, y_n)]|J|dy_1dy_2 \dotsb dy_n
$$

이렇게 변수 변환이 깔끔하게 되는 경우는 실제로 그렇게 자주 발생하지 않는다. 예를 들어서 변환이 1:1로 안되는 경우가 대표적일 것이다. 이러한 경우의 자세한 문제적 예시는 Part. C에서 다뤄보도록 할 것이다.

## linear combinaiton of random variables

자, 여기서 여러분은 선형 결합이 뭔지부터 알아야 할 것이다. 예를 들어서 확률 벡터 $(X_1, \dotsb, X_n)^T$이 있다고 하자. 여기서 여러 확률 변수들의 선형 결합은 특수한 상수 $a_1, a_2, \dotsb, a_n$에 대해서 다음과 같이 정의된다. 

$$
T = \sum_{i = 1}^n a_iX_i
$$

여기서는 $X_i$들의 선형결합으로 표현되는 확률 변수 $T$의 평균, 분산의 표현에 대해서 공부한다.

* 평균

$$
E[T] = \sum_{i = 1}^n a_i\mu_i
$$

* 공분산에 대한 정리

$T$가 위와 같이 $X_i$로 정의되는 선형 결합이고, $W$는 확률 변수 $Y_1, Y_2, \dots, Y_n$과 특정 상수 $b_1, b_2, \dotsb, b_n$로 표현되는 선형 결합이라고 가정하자. 모든 확률 변수에 대해서 분산이 유한할때, 다음이 성립한다.

$$
Cov(T, W) = \sum_{i = 1}^n \sum_{j = 1}^m a_ib_jCov(X_i, Y_j)
$$

* 위 정리에 대한 따름 정리 1

위의 가정과 같이 $X_i$의 분산이 유한이면 다음이 성립한다.

$$
Var(T) = Cov(T, T) = \sum_{i = 1}^n a_i^2Var(X_i) + 2\sum_{i < j} a_ia_jCov(X_i, X_j)
$$

* 위 정리에 대한 따름 정리 2

$X_i$들 끼리 전부 독립이고 모든 $i = 1,2,\dots,n$에 대해서 $Var(X_i) = \sigma_i^2$이면 다음이 성립한다.

$$
Var(T) = \sum_{i = 1}^n a_i^2\sigma_i^2
$$

> 이 결과를 얻기 위해서 단지 모든 $X_i$와 $X_j$에 $(i \neq j)$ 대해서 독립이면 된다. 다음으로는 독립성 이외에도 확률 변수들이 동일한 분포를 가진다고 가정한다. 이러한 확률 변수의 모음을 *확률 표본* 이라고 한다. 이를 축약하여 I.I.D 라고도 한다!



    

