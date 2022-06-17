---
title: DesignPattern-PartA
tags:
  - Design Patterns
  - OOP
Category:
  - Design Patterns  
---

## Design Patterns Part. A - Singleton Pattern

해당 카테고리는 인트로 없이 노빠구로 간다. 왜냐면 이걸 인트로부터 시작하면 100% 미룰 것 같기 때문이다. 
필자가 벡엔드 엔지니어링을 하면서 점점 Design Pattern에 관한 필요성을 느끼게 되었다. 따라서 가장 기초적인 Signleton Pattern부터 NestJS에서 자주 쓰이는 Design Pattern들을 차근차근 정리하기 위해서 이렇게 글을 쓰게 되었다.

일단 설명과 예제 코드는 보여주는 방식으로 글을 진행할 생각이며, 예제 코드는 C++ 또는 TypeScript로 진행할 생각이다.   
때때로 적절한 NodeJS Backend Application을 직접 구현하여 예시를 들어줄 생각이다.  
필자가 Design Pattern의 수 많은 글을 보면서 든 생각이 대체적으로 너무 추상적이라는 것이다. 이를 보완하는 포스트를 작성하고 싶어서 이렇게 글을 끄적여 본다.    
그리고 필자가 가장 애정하는 NestJS를 예시로 많이 설명할 예정이니, 보면서 틀린 점이 있으면 언제든 지적 환영이다.

### Singleton Pattern이란?

Singleton Pattern이란, 전체 시스템에서 class하나 당 한개의 Instance만이 존재하게끔 보장하는 객체 생성 패턴이다. 