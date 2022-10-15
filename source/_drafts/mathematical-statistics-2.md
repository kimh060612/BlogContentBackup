---
title: Mathematical Statistics Chapter 2 Part. A
tags:
  - Mathematics
  - Statistics
categories:
  - Mathematical Statistics
mathjax: true
---

## Index

1. Introduction
2. Distirbution of Two Random Variables
3. Transformation: Binominal Random Variables
4. Conditional Distribution and Expectation
5. Independence Random Variables

## Introduction

이번 포스트에서는 Mathematical Statistics Chapter 2의 전반부의 이론을 설명할 것이다. 앞으로 나올 내용들의 초석이 되니 주의 깊게 공부할 필요가 있다. 또한, 이번 파트부터는 예제 문제를 풀어서 그 풀이까지 올리도록 할 것이다. 이번 Chapter 2의 경우에는 Part A, B, C로 나뉘어질 것이며 각각 전반부 이론, 후반부 이론, 실습 문제 및 Appendix로 구성될 것이다. 이번 파트를 초석으로 앞으로 괴랄할 내용들이 많이 나올 것이다. 쉽게 쉽게 넘길 수 있는 내용이지만, *기초는 쉬운게 아니라 중요한 것이다*. 절대 가볍게 생각하지 말도록 주의하자.

## Distirbution of Two Random Variables

우선, 두개 이상의 랜덤 변수를 쌍으로 묶어서 분포를 보고 싶은 경우가 있다고 하자. 예를 들어서, 동전 3개를 동시에 던져서 나오는 앞면/뒷면의 조합을 보는 사건이 있다고 가정해 보는 것이다. 이러한 경우에는 각 동전별로 랜덤 변수 (가령) $X$, $Y$, $Z$를 할당할 수 있다. 그렇다면 우리는 이 사건은 랜덤 변수의 쌍 $(X, Y, Z)$로 표현할 수 있을 것이다. 이렇게 확률 변수의 쌍을 우리는 확률 벡터(Random Vector)라고 부리기로 하였다.

> Random Vercot의 정의: 표본 공간이 $\mathcal{C}$인 확률 실험이 주어졌을때 $\mathcal{C}$의 각 원소 $c$에 단 하나의 수의 순서쌍 $X_1(c) = x_1, X_2(c) = x_2$를 대응시키는 두 확률 변수 $X_1, X_2$를 생각해보자. $(X_1, X_2)$를 확률 벡터(Random Vector)라고 한다. $(X_1, X_2)$의 공간은 순서쌍 $\mathcal{D} = \{ (x_1, x_2): x_1 = X_1(c), x_2 = X_2(c) \}$이다.

그렇다면 이러한 랜덤 벡터에서 사건의 확률을 어떻게 구할 수 있을까?   
기본적으로 여러 방법들이 있다. 이 챕터에서는 그것을 다룰 것이다. 

1. 기본적으로 각 랜덤 변수의 사건의 교집합의 확률을 구할 수 있다.
2. 여러 사건들이 얽힌 조건부 확률을 구할 수 있다.

기본적으로 이 2개의 방법을 중점으로 다룰 것이다.  
이중에서 해당 절에서는 첫번째를 집중해서 다룰 것이다. 

우선 각각의 확률 변수가 할당된 사건의 교집합에 대한 확률을 구해보도록 하자.   
$\mathcal{D}$를 랜덤 벡터 $(X_1, X_2)$에 대한 공간이라고 하자. $A$가 $\mathcal{D}$의 부분 집합이라고 했을때, 우리는 하나의 확률 변수때와 마찬가지로 이를 사건 $A$라고 한다.  
그리고 이 사건 $A$에 대한 확률을 $P_{X_1, X_2}[A]$라고 표현한다. 그렇다면 이러한 사건 $A$를 다음과 같이 정의해서 구한 확률을 우리는 하나의 확률 변수때와 마찬가지로 CDF(누적 분포 함수)로 정의할 수 있으며 다음과 같이 표현할 수 있다.

$$
F_{X_1, X_2}(x_1, x_2) = P[\{X_1 \leq x_1\} \cap \{ X_2 \leq x_2 \}]
\tag{definition 1}
$$

이렇게 정의한 CDF를 우리는 *결합 누적 분포 함수*라고 부른다. 여기서 공간 $\mathcal{D}$가 가산형이면 우리는 이산형 확률 벡터라고 하고 다음과 같이 *결합 확률 질량 함수*를 정의할 수 있다.

$$
p_{X_1, X_2}(x_1, x_2) = P[X_1 = x_1, X_2 = x_2]
\tag{definition 2}
$$

**단일 확률 변수에서와 마찬가지로 PMF는 CDF를 유일하게 결정짓는다는 특성을 가진다.** 또한, 다음 두 성질 또한 결합 PMF의 특징이다. 

1. $0 \leq p_{X_1, X_2} \leq 1${
2. $\mathop{\sum\sum}_{\mathcal{D}} p_{X_1, X_2} = 1$

여기서 사건 $B$의 경우에는 다음과 같이 확률을 표현할 수 있다.

$$
P[(X_1, X_2) \in B] = \mathop{\sum\sum}_{B} p_{X_1, X_2}(x_1, x_2)
$$

그리고 공간 $\mathcal{D}$가 연속형이라면 다음과 같이 적분으로 결합 CDF를 표현할 수 있다.

$$
F_{X_1, X_2}(x_1, x_2) = \int_{-\infty}^{x_1}\int_{-\infty}^{x_2} f_{X_1, X_2}(w_1, w_2) dw_2dw_1
\tag{definition 3}
$$

위 정의 3에서 $f_{X_1, X_2}$을 결합 확률 밀도 함수라고 한다. 그러면 확률이 0인 사건을 제외하고는 다음과 같이 정의된다.

$$
\frac{\partial^2 F_{X_1, X_2}(x_1, x_2)}{\partial x_1 \partial x_2} = f_{X_1, X_2}(x_1, x_2)
$$

결합 PDF 또한 다음과 같은 본질적인 성질이 있다.

1. $f_{X_1, X_2}(x_1, x_2) \geq 0$
2. $\int\int_{\mathcal{D}} f_{X_1, X_2}(x_1, x_2) dx_1dx_2 = 1$

이것 또한 마찬가지로 사건 $A \in \mathcal{D}$에 대하여 다음과 같이 확률을 구할 수 있다.

$$
P[(X_1, X_2) \in A] = \iint_A f_{X_1, X_2}(x_1, x_2)dx_1dx_2
$$

모두들 중적분쯤은 안다고 가정하고 이 글을 쓰지만 솔직히 양심에 겁나게 찔리기 때문에 중적분에 대한 논의는 Appendix에서 하도록 하겠다. (저자도 따로 사이트에 중적분 관련한 포스트를 올려놨는데 내가 뭐라고...)

### Marginal Distribution

간단하게 말해서 결합 분포는 다음과 같은 분포를 뜻한다.  
> 어떤 확률 벡터의 결합 분포로부터 하나의 확률 변수의 분포를 얻어내는 것.

예를 들어보겠다. 우리는 2개의 확률 변수 $(X_1, X_2)$의 확률 벡터와 그것의 결합 누적 분포를 가지고 있다고 가정하자. 이때 우리는 각각의 확률 변수의 분포를 보고 싶다. 어찌 해야겠는가?

답은 간단하다. 하나의 변수를 전 구간에서 적분해서 없애버리면 된다. 두 확룔 변수가 연속형일때 다음과 같이 $X_1$의 CDF를 구할 수 있다.

$$
F_{X_1}(x_1) = \int_{-\infty}^{x_1}\int_{-\infty}^{\infty} f_{X_1, X_2}(w_1, w_2)dw_1dw_2
$$

반대로 이산형이라면, Summation으로 구할 수 있다.

### Expectation

이 개념을 기댓값을 확장해서 생각해보자. 우선 우리는 확률 벡터의 기댓값을 구하기 전에 확률 벡터에서의 기댓값을 먼저 정의할 필요가 있다.  
벡터는 스칼라가 아니다. 평균을 구한다는 것이 output을 벡터로 내뱉을 것인가? 아니면 스칼라로 뱉을 것인가? 그것부터 알쏭달쏭하지 않은가?  
우리는 다음과 같이 기댓값을 구하는 방법 2가지를 보도록 할 것이다. 

1. 새로운 확률 변수 $Y$를 정의하여 평균을 구하는 방법 $\rightarrow$ output: scalar
2. 주변 분포를 구한 뒤에 각 변수의 평균을 구하는 방법 $\rightarrow$ output: vector

이러면 명확해 진다. 우리는 첫번째 방법부터 차례차례 알아볼 것이다.

새로운 확률 변수 $Y = g(X_1, X_2)$ ($g: \mathbb{R}^2 \rightarrow \mathbb{R}$)을 정의하여 다음과 같은 조건을 만족시킨다면 우리는 랜덤 변수 $Y$에 대해서 기댓값이 존재한다고 할 수 있다.

$$
\int_{-\infty}^{\infty}\int_{-\infty}^{\infty} |g(x_1,x_2)|f_{X_1,X_2}(x_1,x_2)dx_1dx_2 < \infty
$$

그렇다면 기댓값은 다음과 같이 정의할 수 있다.

$$
E(Y) = \int_{-\infty}^{\infty}\int_{-\infty}^{\infty} g(x_1,x_2)f_{X_1,X_2}(x_1,x_2)dx_1dx_2
\tag{definition 4}
$$

이산형이라면 여기서 적분을 Summation으로 바꿔주면 된다.  
그리고 특수한 평균이라고 하면 또 적률 생성 함수를 꼽을 수 있다. 그것은 다음과 같이 정의된다.

> $\mathbf{X} = (X_1, X_2)^T$이고, $h_1,h_2$가 양수일때 $|t_1| < h_1$, $|t_2| < h_2$이라고 하자. 이때, $E(e^{t_1X_1 + t_2X_2})$가 존재하면 이것을 $M_{X_1, X_2}(t_1, t_2)$라고 하고 이를 $\mathbf{X}$의 적률 생성 함수라고 한다.

이때, $\mathbf{t} = (t_1, t_2)^T$라면 다음과 같이 정의할 수 있다.

$$
M_{X_1, X_2}(\mathbf{t}) = E[e^{\mathbf{t}^T\mathbf{X}}]
\tag{definition 5}
$$

우리는 이렇게 첫번째 기댓값을 정의할 수 있었다. 그렇다면 결과가 벡터로 나오는 확률 벡터의 기댓값은 어떻게 구할까?

뭘 어떻게 구하나, 주변 분포 구해서 벌크로 각 랜덤 변수의 평균을 구하면 된다.  
이는 다음과 같이 표기한다.

$$
E[\mathbf{X}] = 
\begin{pmatrix}
    E(X_1) \\
    E(X_2)
\end{pmatrix}
\tag{definition 6}
$$


## Transformation: Binominal Random Variables








