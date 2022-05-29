---
title: 통신 Protocol 공부 - Introduction
date: 2022-03-13 02:14:49
categories: 
    - Networking
tags: 
    - Networking
    - Computer Science
---

### Index

1. 통신 Protocol 공부 배경
2. 참고할 서적들

<br>

### Network Protocol 공부 배경

-------------------------------------------------------

필자가 DDH라는 스타트업에서 일하며서 가장 크게 느낀 것이 바로 현대적인 Computer Networking에 대한 절대적인 지식 부족이었다.  
탄탄한 기초를 가지고 있는 좋은 Engineer가 되기 위해서 TCP/IP에 대한 이해와 HTTP, gRPC에 대한 이해는 필수 불가결하다고 생각한다.  

필자의 통신 프로토콜 공부는 Go언어를 기반으로 진행될 예정이며, 다른 포스트들과는 다르게 실습과 이론을 나누지 않을 예정이다. 이것 만큼은 나누는 것이 오히려 큰 혼란을 불러 일으킬 것 같다.

다음과 같은 큰 주제를 바탕으로 공부가 진행될 것이다.

1. Network System의 개요
2. TCP Socket 통신의 기초
3. HTTP 기초 (HTTP/1.1 정의 및 응용)
4. HTTP 심화 (HTTP/2.0 및 gRPC 관련 내용)

위의 주제들은 1개당 1개의 포스트로 절대 안 끝나고 소 Chapter로 나뉘어 상세하게 다룰 것이다. 겸사겸사 하면서 Go 언어의 문법도 상세히 다룰 예정이다. (랄까 필자도 공부하면서 다룰 것이라 틀린 내용이 있으면 제발 지적좀 부탁한다.)


### 참고할 서적들

-------------------------------------------------------

1. [Go 언어를 활용한 네트워크 프로그래밍](http://www.kyobobook.co.kr/product/detailViewKor.laf?mallGb=KOR&ejkGb=KOR&barcode=9791191600643)
2. [HTTP 완벽 가이드](http://www.kyobobook.co.kr/product/detailViewKor.laf?mallGb=KOR&ejkGb=KOR&barcode=9788966261208)
