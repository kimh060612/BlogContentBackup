---
title: Modern Robotics Chapter 2
categories: 
    - Robotics
tags: 
    - Physics
    - Robotics
---

## Modern Robotics Chapter 2

### Index

1. Introduction of Chapter 2
2. Degrees of Freedom of a Rigid Body
3. Degrees of Freedom of a Robot
   1. Robot Joints
   2. Gr$\ddot u$bler's formula
4. Configure Space: Topology and Representation
5. Configure and Velocity Constraints
6. Task Space and Workspace

### Introduction of Chapter 2

이 파트에서는 강체의 자유도는 무엇인가에 대해서 논의한다.   
Degree of Freedom(DoF)는 물체의 위치를 표현하는데 있어서 필요한 최소의 continuous range of real number의 물리 정보라고 할 수 있다.   
예를 들어서, 문의 현재 위치(혹은 상태)를 표현하는데 있어서 필요한 정보는 문의 힌지로부터 벌어진 각도($\theta$)이다. 

이를 명확하게 정의하면 다음과 같다.

> 정의 2.1) Robot의 configuration은 robot의 모든 지점에서의 위치의 specification이다.   
> 이때, 이러한 configuration을 표현하기 위해서는 최소 $n$개의 real-valued coordinates가 필요한데 이 $n$을 Degree of Freedom(DoF)라고 부른다. 
> 그리고 로봇의 가능한 모든 configuration을 정의하는  $n$-dimensional space를 우리는 $C$-space라고 부른다.

우리는 이번 장 전체에서 $C$-space에 대해서 공부를 진행할 것이다. 

### Degrees of Freedom of a Rigid Body

우리는 이번 장에서 강체에서 DoF를 구하는 방법을 논의할 것이다.   
예를 들어서 손바닥 위에 있는 동전의 DoF를 구하는 방법에 대해서 논의해 보도록 하겠다. 손바닥 위의 동전의 Configuration을 정확하게 설명하려면 다음과 같은 정보가 필요하다.

1. 동전의 위치 $(x,y)$
2. 동전이 향하고 있는 방향 $\theta$

따라서 동전의 DoF는 3이 된다.

이를 조금 더 체계적으로 정리해보자면 다음 방정식과 같이 정리할 수 있다.

$$
\text{degrees of freedom} = \text{(sum of freedoms of the points)} - \text{(number of independent constraints)}
$$

이는 다음과 같이도 표현할 수 있다.

$$
\text{degrees of freedom} = \text{(number of variables)} - \text{(number of independent equations)}
$$

일반적인 3차원 공간에서의 강체는 DoF가 6이다. 이러한 강체를 우리는 Spatial Rigid body라고 부른다. 비슷하게 2차원에서 정의되는 강체는 Planar Rigid Body라고 부른다.

우리가 앞으로 다룰 robot들의 경우 각 구성 요소들이 강체로 되어 있기 때문에 다음과 같은 수식이 성립한다.

$$
\text{degrees of freedom} = \text{(sum of freedoms of the bodies)} - \text{(number of independent constraints)}
$$

### Degrees of Freedom of a Robot

이번 장에서는 지난 장에서 배운 DoF의 개념을 Robot에 적용해 볼 것이다.

#### Robot Joints

Robot의 Joint라 하면 다음 6가지의 종류가 있다. 

<p align="center"><img src="https://kimh060612.github.io/img/RobotJoint.png" width="100%"></p>

각각의 Joint의 설명을 여기서 적는 것도 좋아보이겠지만, 다음으로 미루도록 하겠다. (**추후에 추가할 것**)

Joint는 Robot에서 각각의 강체들이 서로 연관된 움직임을 할 수 있도록 한다. 다른 의미로 보자면, 이어지는 2개의 강체들 사이에 constraints를 주는 역할을 하기도 한다. 

<p align="center"><img src="https://kimh060612.github.io/img/JointConstraints.png" width="100%"></p>

위의 표는 각각의 Joint들이 강체에 주는 Constraints와 DoF값을 나타낸 것이다.   

위 표를 보자. Spatial Rigid body의 경우, 6에서 Constraints 값을 빼면 $f$가 나온다.    

#### Gr$\ddot u$bler's formula

지금부터 소개할 방정식은 Joint가 결합된 robot에서 DoF를 구하는 방정식이다. 이를 Gr$\ddot u$bler's formula라고 한다.

*Proposition 2.2*: $N$개의 link로 이루어진 mechanism이 있다고 해보자. (Ground 또한 link라고 간주한다.) 여기서 $J$개의 joint가 있고 $m$이 강체의 DoF라고 하자. 그리고 $f_i$를 joint에 의해 주어지는 freedom, $c_i$를 그에 의해 주어지는 constraints라고 하자. 

이러한 상황에서 Gr$\ddot u$bler's formula에 의해 robot의 DoF를 구해보면 다음 equation과 같다.

$$
\text{DoF} = m(N - 1) - \sum^J_{i = 1} c_i \\
 = m(N - 1) - \sum^J_{i = 1} (m - f_i) \\
 = m(N - 1 - J) + \sum^J_{i = 1} f_i
$$

이 공식은 모든 joint의 constraints가 독립적일 때 유효하다. 만약 하나라도 독립적이지 않은 joint들이 있다면 lower bound가 성립한다.

