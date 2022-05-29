---
title: 통신 Protocol 공부 - Network Basic part.A
date: 2022-03-13 15:35:50
categories: 
    - Networking
tags: 
    - Networking
    - Computer Science
---

### Index

1. Network Topology
2. Network Bandwidth/Latency
3. OSI 7 layer
4. TCP/IP Protocol Stack
5. IP/IPv4
6. Network ID & Host ID
7. CIDR Basic
8. Network Address Translation
9. Unicast/Multicast/Broadcast
10. Internet Control Message Protocol 
11. Routing Protocol
12. Domain Name Service

<br>

### Network Topology

-------------------------------------------------------

우선 Computer Networking이란 무엇인가부터 생각해보자. 간단하게 생각해서 Computer들 끼리 뭔가를 주고 받는 모습을 떠올릴 수 있다. 이때 이런 Computer들은 Node라고 앞으로 부르겠다. 이러한 Node들이 연결되어 있는 모양새가 여러가지가 있을 수 있다. 이를 topology라고 한다. 

<p align="center"><img src="https://kimh060612.github.io/img/Topology.png" width="100%"></p>

이러한 Topology는 크게 Mesh, Star, Bus, Ring형이 있다. 각각의 network topology 형태에 따라서 장, 단점이 있는데 굳이 여기서는 그런 것을 다루지는 않겠다. 그냥 이런 것이 있다 정도로만 알고 있었으면 좋겠다.

### Network Bandwidth/Latency

-------------------------------------------------------

네트워크의 Bandwidth(대역폭)은 일정 시간 안에 네트워크 연결을 통해서 전달할 수 있는 정보의 양이다. 기본적으로 단위는 bps(bit per second)를 사용한다.   
네트워크 latency는 네트워크 리소스 요청을 보내고 받기까지 걸리는 시간을 말한다. 

결국 어떤 네트워크 application의 성능은 2가지 factor의 영향을 받는다. 첫번째로 인터넷 자체의 대역폭 속도와 SW/HW의 영향을 받는 latency로 말이다. 무조건 대역폭이 높은 네트워크를 사용하는 것만이 네트워크 서비스의 성능을 높이는 길이 아니다. 자신이 만든 서비스의 latency를 낮추는 것 또한 매우 중요한 일이다.  


### OSI 7 layer

-------------------------------------------------------

정말 양산형 블로그답게, OSI 7 layer부터 끊으면서 네트워크 강의를 시작하도록 하겠다. 일단 사진부터 보고 가자.

<p align="center"><img src="https://kimh060612.github.io/img/OSI7.jpg" width="100%"></p>

OSI 7 layer부터는 할 말이 조금 많다. 여기서는 각 layer의 대한 설명과 어떻게 각 layer들끼리 상호작용하여 정보가 전송되는지에 관해서 설명해 보도록 하겠다. 먼저 각 계층에 대해서 설명해 보도록 하자면 다음과 같다.

* 7 계층 - Application layer: 실제 사용자 application과 상호작용을 하는 계층이다. 웹 브라우저 같은 SW들이 이에 속한다.
* 6 계층 - Presentation layer: 데이터의 Encoding, Decoding, Encryption, Decryption등의 역할을 진행한다.
* 5 계층 - Session layer: 네트워크 노드 간의 연결 수명 주기를 관리하는 layer이다. 네트워크 연결을 수립하고 연결 Timeout 관리, 연결 종료 등의 기능을 담당한다.
* 4 계층 - Transport layer: 전송의 안정성을 유지하면서 두 노드간의 데이터의 전송을 제어하고 조정한다. 데이터 전송의 신뢰성을 유지하기 위해서 에러 수정, 데이터 전송 속도 제어, 데이터 청킹 및 분할, 누락된 재전송, 수신 데이터 확인 등의 역할을 한다.
* 3 계층 - Network layer: 노드간의 데이터를 전송하는 역할을 한다. Network layer는 Routing, 주소 지정, multicasting 및 traffic 제어와 관련된 네트워크 관리를 위한 프로토콜의 중심이다. 
* 2계층 - Data Link 계층: 직접 연결된 두 노드 간의 데이터 전송을 처리한다. 뿐만 아니라 데이터 링크 계층에서의 프로토콜은 물리 계층에서의 에러를 식별하여 수정을 시도하기도 한다. 
* 1계층 - Physcial layer: 네트워크 스택에서 발생한 비트를 하위 물리적 매체가 제어할 수 있는 전기, 광학 또는 무선 신호로 변환하고 타 노드의 물리적 매체에서 받은 신호를 다시 비트로 전환하는 역할을 한다.

이렇게 간단하게 설명을 해 보았는데, 중요한 것은 이것들의 역할을 전부 외우고 있는 것이 아닌, 각각이 어떤 역할을 하고 있는지 이해하는 것이 중요하다.

간략하게만 나누면 위 7개의 계층을 2개의 계층으로 나눌 수 있다. 

1. Application layer (7 ~ 5 계층)
2. Data Flow layer (4 ~ 1 계층)

이렇게 나누는 이유는 데이터를 전송하는 계층들과 응용하는 계층들의 역할이 위와 같이 나뉘기 때문이다. 따라서 자신의 목표에 따라 어떤 layer의 프로토콜을 공부해야할지가 정해질 것이다. 

우선 데이터는 상위 계층에서 생성되어 하위 계층으로 전달된다. 그리고 최종적으로 1 layer에 도달하여 이진수로 변환되고 이 이진수는 다른 node의 1 layer에 도착한다. 이러한 데이터를 받은 다른 node의 1 layer는 정보를 처리하여 상위 레이어로 데이터를 전달한다. 이렇게 각 node의 사용자들이 데이터를 주고 받는 것이다. 

해당 과정에서 상위 계층에서 하위 계층으로 데이터를 전달하여 전송하는 과정을 "Encapsulation", 받아서 하위에서 상위 계층으로 전달하는 과정을 "Decapsulation"이라고 한다.

<p align="center"><img src="https://kimh060612.github.io/img/OSI7_N2N.png" width="100%"></p>

현대 네트워크는 대부분 패킷 기반의 네트워크이다. 

현재는 OSI 7 layer는 잘 쓰이지 않는다. 대신 TCP/IP 프로토콜 스택이 훨씬 많이 쓰이므로 다음에 그것을 다뤄보도록 하겠다. 

### TCP/IP Protocol Stack

-------------------------------------------------------

현재 인터넷으로 통신하는데 있어 가장 널리 사용되는 프로토콜이다. 이를 OSI 7 layer와 비교하며 설명하도록 하겠다.
밑의 그림에서 보는 것 처럼 OSI 7 layer와 비교되는 layer들이 대칭적으로 그려져 있다. 하나하나 설명을 해보도록 하겠다.

<p align="center"><img src="https://kimh060612.github.io/img/TCPIP.jpg" width="100%"></p>


* Application Layer: OSI 7 계층에서는 5~7 계층에 해당되는 layer이다. 대부분의 SW Application과 직접 소통으로 하며 통상적으로 개발자가 다루는 프로토콜은 대부분 해당 계층의 프로토콜이다.
* Transport Layer: OSI 7 계층에서는 4 계층에 해당되는 layer이다. 데이터의 전송을 처리하며 출발지에서 전송된 모든 데이터가 목적지로 데이터 무결성을 보장하며 올바르게 전송되도록 함. 해당 계층의 프로토콜로는 TCP/UDP 등이 있다.
* Internet/Network Layer: OSI 7 계층에서는 3 계층에 해당하는 layer이다. 출발지 노드와 목적지 노드 사이의 상위 계층에서 데이터 패킷을 라우팅하는 기능을 한다. IPv4, IPv6, BGP, ICMP, IGMP, IPsec 등이 이 계층에 해당하는 프로토콜이다.
* Link Layer: OSI 7 계층에서는 2~1 계층에 해당하는 layer이다. 해당 layer의 ARP(주소 결정 프로토콜)는 노드의 IP주소를 Network Interface의 MAC 주소로 변환한다. 

이 Protocol Stack 또한 OSI 7 layer와 마찬가지로 Encapsulation, Decapsulation 과정을 거쳐서 데이터를 전송하게 된다. 지금은 이런 것이 있다 정도로만 알아두면 된다.

### IP/IPv4

-------------------------------------------------------


