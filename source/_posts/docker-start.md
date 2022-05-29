---
title: Deep Dark Docker - Introduction
date: 2022-03-13 14:57:31
categories: 
    - Containerization
tags: 
    - Containerization
    - Lightweight Virtualization
---

### Index

1. Post의 주제 소개
2. 강의 내용

<br>

### Post의 주제 소개

-------------------------------------------------------

이 카테고리의 포스트는 Docker에 대해서 더 딥 다크한 부분까지 알고 싶어하는 사람들을 위해서 만들어 나갈 것이다. 먼저 그 전에 Docker에 대해서 잠깐 짚고 넘어가는 것이 좋을 것 같다. 

> Docker

Docker는 수 많은 Containerization Software 중 하나이다. open source이며 가장 유명하기도 하다. 그렇다면 Containerization은 무엇일까?

> Containerization

여기서는 RedHat 공식 홈페이지에서 나온 문구를 인용해 보도록 하겠다. 

*컨테이너화란 소프트웨어 코드를 라이브러리, 프레임워크 및 기타 종속성과 같은 필수 요소와 함께 패키지에 포함하여 각자의 "컨테이너"로 분리하는 것을 뜻합니다.*

간단하게 말하면 우리가 작성한 Application을 운영 체제 내에서 하나의 고립된 공간을 만들어 패키징 하는 것을 말한다. 

주로 사람들이 Virtual Machine하고 비교를 하곤 한다. 하지만 필자의 생각으로는 비교 자체가 조금 잘못되었다. Virtual Machine은 OS Kernel 바로 위에 Hypervisor라는 가상화 layer를 추가하여 별도의 OS를 올릴 수 있게 가상화를 하는 것이다. (밑의 그림이 아주 좋은 예시이다.)   
그에 반해서 Docker는 기존에 OS에서 돌아가는 Process의 일종이다. 하지만 그 Process가 사용할 수 있는 PID, Network Interface, CPU, Memory 등의 여러 자원들을 다른 Process들과 고립시켜서 마치 독립된 머신 위에서 돌아가는 것 "처럼" 만드는 것이다.   
둘은 엄밀히 다른 목적을 가지고 사용되어야 한다. 누가 더 좋냐의 문제가 아니라 때에 맞춰서 적절한 기술을 사용해야하는 것이다.

<p align="center"><img src="https://kimh060612.github.io/img/Virtualization.png" width="100%"></p>


그래서 이 카테고리에 해당하는 포스트들은 뭔 내용을 다루는가?  
답은 간단하다

> Docker는 어떻게 Process를 Isolation 하는가?

Docker가 어떤 기술을 사용해서 process를 고립시키는지 OS, Computer Network의 관점에서 서술할 것이다.   
여기서는 어떻게 Docker를 사용하는 방법 같은 것은 일체 다루지 않을 것이다. 이미 Docker를 능숙하게 다룰 수 있는 사람이 어떻게 하면 좀 더 효율적으로 Docker를 사용할 수 있는지에 대한 Insight를 제공하는 목적으로 서술될 것이다. 그래서 제목이 Deep Dark Docker이다. 

~~진짜 이딴걸 누가 볼까 싶다~~


### 강의 내용

-------------------------------------------------------

강의는 크게 다음과 같은 주제를 필두로 전개될 것이다. 

1. PID/Network/IPC Isolation - Linux Namespace
2. Network Isolation Advanced - Docker Network
3. Computing Resource Isolation - Cgorup
4. Docker File System Isolation - Docker Volume, linux Namespace

우선 Docker가 process id, network interface, IPC 같은 자원들을 어떻게 isolation하는지를 다룰 것이다. 이때 주요하게 쓰이는 기술이 바로 Linux Namespace이다. 따라서 이를 Deep Dark하게 다뤄볼 것이다.

그리고 Web Service를 구축하는데 있어서 가장 중요한 부분인 "어떻게 Network를 구성할 것이가"에 대한 내용을 다룰 것이다. 이는 Docker Network를 통해서 조금 실용적으로 다뤄볼까 한다. 

다음으로는 Docker가 어떻게 CPU, RAM Memory 같은 자원을 Container에게 할당하고 제한하는지를 알아볼 것이다. 이때 사용되는 기술이 Control Group인데, 이 또한 Deep Dark하게 알아보고자 한다.

그리고 마지막으로 Docker의 File System에 대해서 알아볼 것이다. 이는 비단 Docker가 어떻게 File System을 고립시키는지 알아보고 실습을 통해서 어떻게 하면 안전하게 docker file system을 관리할 수 있는지에 대해서 알아본다.

진짜 이쯤 설명하고 나니 누가 볼까 싶지만 일단 끄적여 보겠다. 필자도 많이 부족하므로 틀린 부분이 있다면 망설이지 말고 지적해 줬으면 한다.