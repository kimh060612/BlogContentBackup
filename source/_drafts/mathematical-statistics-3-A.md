---
title: Mathematical Statistics Chapter 3 Part. A
tags:
  - Mathematics
  - Statistics
categories:
  - Mathematical Statistics
mathjax: true
---

## Index

1. Binomial Distribution
2. Poison Distribution
3. Normal Distribution
4. MultiVariate Normal Distribution
5. $t$-Distribution & $F$-Distribution

## Binomial Distribution

여기서는 Bernoulli experiment에 관한 확률 분포인 이항 분포와 그것에 관련된 분포들에 대해서 설명을 한다. 우선, Bernoulli experiment가 무엇인지부터 알아야 한다.   
Bernoulli experiment는 확률 실험으로, 그 결과는 성공 또는 실패와 같이 서로 배타적으로 전체를 이루는 방법 중 하나로 분류할 수 있다. 일련의 Bernoulli trial은 Bernoulli experiment을 독립적으로 여러번 실행했을때 발생하고 이때 성공 확률 $p$는 동일하게 유지된다. 

다음과 같이 정의된 Bernoulli trial에서 얻은 확률 변수를 $X$라고 하자.

$$
X(\text{Success}) = 1, \space X(\text{failure}) = 0
$$

이러한 $X$의 pmf는 다음과 같이 정의할 수 있다.

$$
p(x) = p^x(1 - p)^{1 - x}
$$

이때, $X$가 Bernoulli Distribution을 따른다고 표현한다.  
이를 시행 횟수가 $n$번인 실험으로 확장하여 pmf를 정의해보면 다음과 같다.

$$
p(x) = 
\begin{cases}
    \begin{pmatrix}
        n \\
        x
    \end{pmatrix}
    p^x(1 - p)^{n - x} & x=0,1,2,...,n \\
                        \\
    0 & \text{otherwise}

\end{cases}
$$

이때, 위 수식중에서 이항 계수의 정의를 가볍게 적어보면 다음과 같다.

$$
\begin{pmatrix}
    n \\
    x
\end{pmatrix} = \frac{n!}{x!(n - x)!}
$$

이러한 pmf를 가지는 확률 변수 $X$를 두고 "$X$가 이항 분포를 따른다"라고 하며 이와 같은 분포를 이항 분포(Binomial Distribution)이라고 한다. 그리고 여기서 상수 $n,p$를 이항 분포의 모수라고 한다.

이항 분포의 MGF 또한 쉽게 구할 수 있다. 

$$
\begin{align}
    M(t) &= \sum_xe^{tx}p(x) \\
        &= \sum_x\begin{pmatrix}
                    n \\
                    x
                \end{pmatrix}
                e^{tx}p^x(1 - p)^{n - x} \\
        &= \sum_x \begin{pmatrix}
                    n \\
                    x
                \end{pmatrix} (pe^t)^x(1 - p)^{n-x} \\
        &= (1 - p + pe^t)^n
\end{align}
$$

여기서 관련 정리를 하나 더 보고 들어가자.

> $i = 1,2,\dots,m$일때, $X_i$가 이항분포 $b(n_i,p)$를 따르는 독립인 확률 변수들을 $X_1, X_2, \dots, X_m$이라고 하자. $Y=\sum_{i=1}^mX_i$라면 $Y$는 이항분포 $b(\sum_{i=1}^mn_i, p)$를 따른다.

이 절의 나머지 부분은 이항 분포와 관련된 몇가지 중요한 분포들을 소개한다.

1. 음이항과 기하분포

    우선 음이항분포는 무엇인지부터 설명하도록 하겠다.  
    음이항분포는 기본적으로 베르누이 실행에 토대를 두고 있는데, $r$번째 성공이 발생하기 전에 이 시행에서 얻은 실패의 횟수를 확률 변수 $Y$로 정의하고 이 분포를 정의한 것이다.   
    이때, $r = 1$인 특수한 경우를 기하 분포라고 한다.

    $$
    P_Y(y) = \begin{cases}
        \begin{pmatrix}
            y + r - 1 \\
            r - 1
        \end{pmatrix}
        p^r(1 - p)^y & y=0,1,2,... \\
                            \\
        0 & \text{otherwise}
    \end{cases}
    $$

2. 다항분포

    이항분포는 다음과 같이 다항 분포로 일반화될 수 있다. 여러 확률 변수의 tuple이 다음과 같이 정의되어 있다고 가정하자.  
    $(X_1, X_2, ..., X_k)$  
    이때, 다음이 만족한다고 가정하자.  
    1. $x_1 + x_2 + \dots + x_k = n$
    2. $p_1 + p_2 + \dots + p_k = 1$
    
    이 경우에는 위와 같은 tuple의 결합 분포는 다음과 같다.

    $$
    P(X_1=x_1, X_2=x_2, \dots,X_k=x_k) = \frac{n!}{x_1!x_2!\dotsb x_k!}p_1^{x_1}p_2^{x_2}\dots p_k^{x_k}
    $$

3. 초기하분포

    기본적으로 비복원 추출 문제를 해결할때 이 분포를 많이 사용한다. $(N, D, n)$을 모수로 한다. 예를 들어서 $N$개의 물품이 있고 $D$개의 불량이 있다고 하자. 이때, $n$번 표본에서 추출했을때 불량이 나올 개수를 확률 변수 $X$로 정의하면 다음과 같은 초 기하 분포의 pmf가 나온다.

    $$
    p(x) = \frac{\begin{pmatrix}
            N- D \\
            n - x
        \end{pmatrix} \begin{pmatrix}
            D \\
            x
        \end{pmatrix}}{\begin{pmatrix}
            N \\
            n
        \end{pmatrix}}, \space x = 0,1,\dots,n
    $$

## Poison Distribution

