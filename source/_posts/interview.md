---
title: interview
tags:
  - Javascript
  - Operating System
  - Networking
  - DB
  - Algorithm
  - Data Structure
categories:
  - Interview
date: 2022-04-23 17:54:07
---


## 해당 포스트는 필자가 직무 Interview를 위해 스터디한 내용을 적어두는 곳입니다. 틀린 내용이 있을 수 있습니다.

## Index

* DS & Algorithm  
  * Linked List / Array 
    * Linked List for Array
    * Linked List for Tree
  * Stack  
  * Queue    
  * Sort  
    * Stable / Unstable Sort 
    * Bubble Sort
    * Insert Sort
    * Selection Sort
    * Quick Sort
    * Merge Sort
    * Count Sort
    * Radix Sort
  * Tree
    * Binary Tree (Vanila, Full, Complete)
    * Binary Search Tree  
    * Red Black Tree  
    * B Tree  
    * B+ Tree
  * Heap  
  * Hash Table  
  * Graph  
    * Graph Travarsal  
    * Minimum Spanning Tree  
      * Kruskal algorithm  
      * Prim algorithm  
* Operating System  
  * Process    
  * Thread  
  * Multi-processing / Multi-Threading  
  * Scheduling  
  * System Call  
  * Synchronization  
  * Memory
    * Memory Management Strategy
    * Locality of Cache
* Database  
  * SQL vs NoSQL    
  * Index  
  * Transaction  
  * Lock    
  * deadlock  
  * Redis  
  * High Avaliablity Strategy  
  * ORM  
* Web  
  * TCP / UDP
  * TCP 3way/4way hand shake  
  * TLS/SSL  
  * HTTP 1.1/2.0  
  * HTTP Cache  
  * RESTful API    
  * Sync / Async / Blocking / Non-Blocking    
  * Blocking IO / Non-Blocking IO    
* Javascript/Nodejs    
  * Javascript      
  * NodeJS 
* Object Oriented Programming Basic
  * Properties of OOP
  * 5 Principle  


## DS & Algorithm  

### Linked List / Array  

먼저 Array에 대해서 설명을 해보자면 Array는 논리적 저장 순서와 물리적 저장 순서가 일치한다. 따라서 index를 통해서 해당 element에 접근이 가능하다. 그렇기에 찾고자 하는 원소의 index를 알고 있으면 $O(1)$의 시간 복잡도로 해당 element에 접근할 수 있다. 즉 random access가 가능하다는  장점이 있다. 

하지만 insert와 delete 연산의 경우에는 이야기가 다르다. 어떠한 index의 element를 삽입/삭제 하는 경우에는 그 뒤의 원소들을 한칸씩 뒤/앞으로 shift해줘야 한다. 이 경우에 시간 복잡도가 O(n)이 된다.

Linked List는 이와는 다르다. 각각의 원소들은 자기 자신 다음에 어떤 원소인지만을 기억하고 있다. 따라서 이 부분만 다른 값으로 바꿔주면 삭제와 삽입을 O(1) 만에 해결할 수 있는 것이다.

하지만 Linked List 역시 한 가지 문제가 있다. 원하는 위치에 삽입을 하고자 하면 원하는 위치를 Search 과정에 있어서 첫번째 원소부터 다 확인해봐야 한다는 것이다. Array 와는 달리 논리적 저장 순서와 물리적 저장 순서가 일치하지 않기 때문이다. 이것은 일단 삽입하고 정렬하는 것과 마찬가지이다. 이 과정 때문에, 어떠한 원소를 삭제 또는 추가하고자 했을 때, 그 원소를 찾기 위해서 O(n)의 시간이 추가적으로 발생하게 된다.

결국 linked list 자료구조는 search 에도 O(n)의 time complexity 를 갖고, 삽입, 삭제에 대해서도 O(n)의 time complexity 를 갖는다. 그렇다고 해서 아주 쓸모없는 자료구조는 아니기에, 우리가 학습하는 것이다. 이 Linked List 는 Tree 구조의 근간이 되는 자료구조이며, Tree 에서 사용되었을 때 그 유용성이 드러난다.

### Stack

선형 자료구조의 일종으로 Last In First Out (LIFO). 즉, 나중에 들어간 원소가 먼저 나온다. 이것은 Stack 의 가장 큰 특징이다. 차곡차곡 쌓이는 구조로 먼저 Stack 에 들어가게 된 원소는 맨 바닥에 깔리게 된다. 그렇기 때문에 늦게 들어간 녀석들은 그 위에 쌓이게 되고 호출 시 가장 위에 있는 녀석이 호출되는 구조이다.

### Queue

선형 자료구조의 일종으로 First In First Out (FIFO). 즉, 먼저 들어간 놈이 먼저 나온다. Stack 과는 반대로 먼저 들어간 놈이 맨 앞에서 대기하고 있다가 먼저 나오게 되는 구조이다. 이를 구현하고 있는 Priority queue등을 사용할 수 있다.

### Sort

#### Stable / Unstable Sort

Stable Sort와 Unstable Sort의 차이는 *같은 값의 숫자더라도 그 상대적인 위치가 유지 되느냐 아니냐*이다.    
예를 들어서 다음과 같이 "3 *3* 4 2 1 5 **3**"가 있다고 해보자.  
만약 sorting된 결과가 다음과 같으면 stable sort이다.  
"1 2 3 *3* **3** 4 5"    
이 이외의 결과는 전부 unstable sort이다. 

#### Bubble Sort

서로 인접한 두 원소를 검사하여 정렬하는 알고리즘  
==> 인접한 2개의 레코드를 비교하여 크기가 순서대로 되어 있지 않으면 서로 교환한다.

기본적으로 $O(n^2)$의 시간 복잡도를 가진다. 

#### Insert Sort

자료 배열의 모든 요소를 앞에서부터 차례대로 이미 정렬된 배열 부분과 비교 하여, 자신의 위치를 찾아 삽입함으로써 정렬을 완성하는 알고리즘

기본적으로 $O(n^2)$의 시간 복잡도를 가진다. 

#### Selection Sort

해당 순서에 원소를 넣을 위치는 이미 정해져 있고, 어떤 원소를 넣을지 선택하는 알고리즘  
==> 첫 번째 순서에는 첫 번째 위치에 가장 최솟값을 넣는다.  
==> 두 번째 순서에는 두 번째 위치에 남은 값 중에서의 최솟값을 넣는다.  

기본적으로 $O(n^2)$의 시간 복잡도를 가진다. 

#### Quick Sort

분할 정복 알고리즘의 하나로 평균적으로 $O(n\log_2 n)$의 시간 복잡도를 가진다. 이는 불안정 정렬에 속한다.   
과정은 다음과 같다. 

1. 리스트 안의 한 요소를 선택한다. 이를 pivot이라고 하겠다.
2. pivot을 기준으로 그보다 작은 요소를 전부 pivot의 왼쪽으로, 큰 요소는 오른쪽으로 옮긴다. 
3. pivot을 제외한 왼쪽 리스트와 오른쪽 리스트를 동일한 방법으로 재 정렬한다. 
4. 이를 부분 리스트들이 더 이상 분할 불가능 할때까지 반복한다.

최악의 경우에는 시간 복잡도가 $O(n^2)$이 되며 이 경우는 대표적으로 리스트가 전부 정렬 되어 있는 경우이다. 

실질적으로는 pivot을 중간값으로 잡아서 알고리즘을 최적화 하는 방법도 있다. 정말 아무 값이나 pivot으로 잡아버리면 리스트가 불균등하게 분할되어 시간이 오래 걸리는 경우가 생긴다. 

이를 방지하기 위해서 C++에서는 quick sort와 heap sort를 합친 intro sort를 사용하고 있다. 이는 기본적으로 quick sort와 동일하게 동작하지만 재귀 호출의 깊이가 일정 수준 이상으로 깊어지면 그 다음부터는 쪼개진 list를 heap sort로 정렬하여 시간 복잡도를 개선하였다. (구체적으로는 재귀함수의 깊이가 $1.5\log_2 N$보다 깊어지면 heap sort를 사용하여 정렬을 진행한다.)

#### Merge Sort

분할 정복 알고리즘의 하나로 $O(n\log_2 n)$의 시간 복잡도를 가진다. quick sort와 달리 안정 정렬에 속한다.   
과정은 다음과 같다.

1. 리스트의 절반의 길이로 리스트를 2개로 분할한다.
2. 계속 이렇게 잘라 나가면서 각 부분 리스트들을 재귀적으로 합병 정렬을 이용해서 정렬한다.
3. 두 부분 리스트들을 하나의 정렬된 리스트들로 병합한다.

단점으로는 별도의 메모리 공간이 필요하다. 리스트를 병합하는 과정에서 추가적인 메모리 공간에 이를 덧씌우는 작업을 진행하는데, 이 과정에서 컴퓨팅 resource를 잡아 먹으므로 실제로는 quick sort가 최선의 경우에서는 더 좋은 성능을 내고 있다. 

#### Count Sort

등장하는 숫자의 개수를 counting 하면서 정렬을 하는 알고리즘이다. 
정렬 자체는 시간복잡도가 $O(N)$이지만, 공간 복잡도가 리스트의 원소에 따라 달라진다. 게다가, 리스트의 원소가 크면 클수록 무의미한 순회를 반복하기 때문에 비효율성이 발생되는 정렬 방법이다. 

#### Radix Sort

기수 정렬은 낮은 자리수부터 정렬해 가는 정렬이다. 비교 연산을 하지 않으며 시간 복잡도가 $O(dN)$으로 빠른 정렬중에 하나이다. 하지만 데이터의 자리수에 따라서 정렬 시간 복잡도에 영향을 미치며, 추가적인 Queue를 사용하며 메모리 공간을 차지 한다는 점이 단점이다. 

### Tree

Tree는 기본적으로 Stack이나 Queue와는 달리 계층적 구조를 표현하는 비선형 자료구조이다. Tree의 명확한 정의는 "Cycle이 없는 Graph"이다. 

다음은 기본적인 Tree에서의 용어는 Graph에서의 그것과 큰 차이는 없지만 몇가지 추가되는 것이 있다.

1. Root Node: 트리 구조에서 최상위 노드에 있는 것을 의미한다.
2. Leaf Node: 하위에 다른 노드들이 연결되어 있지 않은 Node를 의미한다.
3. Internal Node: 단말 노드를 제외한 모든 노드로 Root Node를 포함한다.

#### Binary Tree (Vanila, Full, Complete)

우선 기본적인 Binary Tree의 정의부터 살펴보도록 하겠다.  
Root Node를 중심으로 2개의 Sub Tree로 나뉘는 Tree를 뜻한다. 이때 해당 Sub tree 들도 Binary Tree의 조건을 만족해야 한다.   
이 재귀적인 정의가 성립하려면 공집합도 Binary tree이고 Node가 하나인 것도 Binary Tree이다.   

이러한 Binary Tree는 그 형태에 따라서 3가지 종류로 나뉜다.

1. Perfect Binary Tree: 모든 level이 꽉 찬 Tree   
2. Complete Binary Tree: 마지막 level 을 제외한 나머지 level 에 node 들이 가득 차있고, 마지막 level 에서 node 는 가장 왼쪽 부터 채워지는 형태가 Complete Binary Tree 이다.   
3. Full Binary Tree: 모든 노드가 0개 혹은 2개의 children 을 가지고 있는 Tree


#### Binary Search Tree

Binary Search Tree는 다음과 같이 정의된 Tree를 말한다.

1. BST의 Node에 저장된 Key는 유일하다
2. 부모의 Key는 왼쪽 자식 Node의 Key보다 크다.
3. 부모의 Key는 오른쪽 자식 Node의 Key보다 작다.
4. 왼쪽과 오른쪽 Sub Tree도 BST이다.

BST의 특장점은 트리 위에서 특정 Key를 검색하는 속도가 $O(N\log_2 N)$이라는 것이다.  
하지만 만약 한쪽으로만 계속 데이터가 저장되어 Skewed Tree가 된다면 성능에 영향을 미칠 수 있다. 대표적으로 한쪽으로만 데이터가 계속 저장되면 탐색에 필요한 시간 복잡도는 $O(N)$이 된다.

#### Red Black Tree

Red Black Tree는 BST에서 Insert/Delete 연산을 수행할때의 시간 복잡도를 줄이기 위해서 나타난 자료구조이다. 동일한 Node의 개수일때, Depth를 최소화하여 시간 복잡도를 줄이는 것이 핵심 Idea이다. 

RBT의 명확한 정의는 다음과 같다.

1. BST의 정의를 만족해야 한다.
2. 각 Node는 red 또는 Black의 색을 가진다.
3. Root Node는 무조건 Black이다.
4. 각 Leaf Node(NULL)는 무조건 Black이다.
5. 어떤 Node의 색이 Red라면 두 Children의 색은 모두 Black이다.
6. 각 Node에 대해서 Node로 부터 descendant leaves 까지의 단순 경로는 모두 같은 수의 black nodes 들을 포함하고 있다. 

이러한 RBT는 다음과 같은 특징을 가진다.

1. Binary Search Tree 이므로 BST 의 특징을 모두 갖는다.
2. Root node 부터 leaf node 까지의 모든 경로 중 최소 경로와 최대 경로의 크기 비율은 2 보다 크지 않다. 이러한 상태를 balanced 상태라고 한다.
3. 노드의 child 가 없을 경우 child 를 가리키는 포인터는 NIL 값을 저장한다. 이러한 NIL 들을 leaf node 로 간주한다.

그리고 RBT의 삽입과 삭제 연산은 다음과 같이 수행된다.

* 삽입

    BST의 특성을 유지하면서 Node를 삽입한다. 그리고 삽인된 Node의 색은 Red로 지정한다.   
    삽입 결과가 RBT의 특성에 위반될 경우에는 Node의 색을 조정하고 Black-Height가 같아야 한다는 정의에 위반되면 rotation을 통해서 Height를 조정한다.
    이러한 과정을 통해서 RBT는 동일한 Height에 존재하는 internal node들의 Black-hieght가 같아지게 되고 최소 경로와 최대 경로의 크기 비율이 2 미만으로 유지된다. 

* 삭제

    삭제도 삽입과 마찬가지로 BST 의 특성을 유지하면서 해당 노드를 삭제한다.   
    삭제될 노드의 child 의 개수에 따라 rotation 방법이 달라지게 된다.   
    그리고 만약 지워진 노드의 색깔이 Black 이라면 Black-Height 가 1 감소한 경로에 black node 가 1 개 추가되도록 rotation 하고 노드의 색깔을 조정한다.   
    지워진 노드의 색깔이 red 라면 Violation 이 발생하지 않으므로 RBT 가 그대로 유지된다.

#### B Tree

B Tree는 이진트리에서 발전되어 모든 Leaf Node들이 같은 level을 가질 수 있도록 자동으로 balancing이 되는 Tree이다. 정렬 순서를 보장하고 Multi Level Indexing을 통한 빠른 검색이 가능하다.  

최대 $M$개의 자식을 가질 수 있는 B Tree의 정의는 다음과 같다.  

1. Node에 최대 $M$개부터 $\frac{M}{2}$개 까지의 자식을 가질 수 있다.
2. Node에 최대 $M-1$개부터 $[\frac{M}{2}] - 1$개의 Key가 포함될 수 있다.
3. Node의 Key가 $x$개라면 자식의 수는 $x+1$개이다.
4. 최소 차수는 자식수의 하한값을 의미하며 최소 차수가 $t$라면 $M = 2t-1$을 만족한다.
5. BST 처럼 각 Key들의 왼쪽 자식들은 항상 부모보다 작은 값을, 오른쪽 자식들은 부모보다 큰 값을 가진다.

#### B+ Tree

B+ Tree는 B Tree를 변형한 자료구조이다. 이는 실제 여러 DB에서 Index를 저장할때 사용되고 있다. 

B+ Tree는 B Tree와 다음이 다르다.

1. 모든 Key, Data가 Leaf Node에 모여 있다. B Tree는 Leaf Node가 아닌 각자 Key 마다 data를 가진다면 B+ Tree는 Leaf Node에 모든 data를 가진다. 
2. 모든 Leaf node가 Linked List의 형태를 띄고 있다. 따라서 B+ Tree에서는 Root Node에서부터 검색을 하는 것이 아닌, Leaf Node에서 선형 검사를 할 수 있어서 시간 복잡도가 굉장히 줄어든다.


### Heap

Heap은 특정한 규칙을 만족시키는 Complete Binary Tree의 일종이다. Heap에는 Max Heap과 Min Heap 2가지 종류가 있다. 

Max Heap이란 각 Node의 값이 해당 Children의 값보다 크거나 같은 Complete Binary Tree를 말한다.   
그리고 Min Heap은 그것의 반대이다.   

Max Heap에서는 Root node 에 있는 값이 제일 크므로, 최대값을 찾는데 소요되는 연산의 time complexity 이 $O(1)$이다. 그리고 complete binary tree이기 때문에 배열을 사용하여 효율적으로 관리할 수 있다. (즉, random access 가 가능하다.) 또한 Min Heap도 최소값이라는 점만 다르고 나머지는 동일하다.

### Hash Table 

Hash의 정의는 다음과 같다.

> 내부적으로 Array를 사용하며 특수한 알고리즘(Hash function)을 사용하여 데이터와 연관된 고유한 숫자를 만들어낸 후에 이를 Index로 사용하는 자료구조.

이러한 특수한 알고리즘 덕분에 collision이 발생하지 않는다면 $O(1)$의 시간 복잡도로 데이터를 찾을 수 있다. 

하지만 어설픈 Hash function을 사용하면 당연히 collision이 자주 발생하게 된다. 
따라서 좋은 Hash Function을 사용해서 Hash Table을 구성해야 한다. 

Hash function을 무조건 1:1로 만드는 것보다 Collision을 최소화하는 방향으로 설계하고 발생하는 Collision에 대비해서 어떻게 대응할 것인가가 훨씬 더 중요하다. 1:1 대응이 되도록 만드는 것이 거의 불가능하기도 하지만 그런 hash function를 만들어봤자 그건 array 와 다를바 없고 메모리를 너무 차지하게 된다.

이제는 Conflict를 해결하는 방법을 알아보도록 하겠다.

1. Open Address 방식

    Hash에서 Collision이 발생하면 다른 Hash Bucket에 해당 자료를 삽입하는 방식이다. 
    여기서 Hash Bucket은 Hash Table에 데이터를 저장하는 하나의 공간이다.   

    이렇게 되면 원하는 데이터를 $O(1)$에 못 찾을 수도 있다. 그럴 경우에 원하는 데이터를 찾아내는 여러가지 방법이 있다. 대표적인 예시로 다음과 같은 것들이 있다.

    * Linear Probing: 순차적으로 탐색하며 비어있는 버킷을 찾을 때까지 계속 진행된다.
    * Quadratic probing: 2 차 함수를 이용해 탐색할 위치를 찾는다.
    * Double hashing probing: 하나의 해쉬 함수에서 충돌이 발생하면 2 차 해쉬 함수를 이용해 새로운 주소를 할당한다. 위 두 가지 방법에 비해 많은 연산량을 요구하게 된다.

2. Separate Chaining 방식

    일반적으로 Open Addressing은 Separate Chaining 방식보다 느리다. Separate Chaining 방식은 Hash Collision이 발생하지 않도록 보조 함수를 통해 조정할 수 있다면 Worst Case에 가까워지는 빈도를 줄일 수 있다.

    이러한 Separate Chaining 방식은 2가지 구현 방식이 있다. 

    1. Linked List를 활용: 각각의 버킷(bucket)들을 연결리스트(Linked List)로 만들어 Collision 이 발생하면 해당 bucket 의 list 에 추가하는 방식이다. 
    2. Red Black Tree를 이용: 연결 리스트 대신 트리를 사용하는 방식이다. 연결 리스트를 사용할 것인가와 트리를 사용할 것인가에 대한 기준은 하나의 해시 버킷에 할당된 key-value 쌍의 개수이다. 데이터의 개수가 적다면 링크드 리스트를 사용하는 것이 맞다.

### Graph

Graph는 크게 Undirected Graph와 Directed Graph로 나뉜다. 말 그대로 간선 연결 관계에서 방향성이 없는 것을 전자, 방향성을 가진 것을 후자라고 한다.   
이때, Degree라는 개념이 등장한다. 이는 Undirected Graph 에서 각 정점(Vertex)에 연결된 Edge 의 개수를 Degree 라 한다.  
그리고 Directed Graph 에서는 간선에 방향성이 존재하기 때문에 Degree 가 두 개로 나뉘게 된다. 각 정점으로부터 나가는 간선의 개수를 Outdegree 라 하고, 들어오는 간선의 개수를 Indegree 라 한다.

이러한 Graph를 구현하는 방법은 2가지가 있다. 

* 인접 행렬을 사용하는 방법: 해당하는 위치의 value 값을 통해서 vertex 간의 연결 관계를 $O(1)$ 으로 파악할 수 있다. Edge 개수와는 무관하게 $V^2$ 의 Space Complexity 를 갖는다. Dense graph 를 표현할 때 적절할 방법이다.
* 인접 리스트를 사용하는 방법: vertex 의 adjacent list 를 확인해봐야 하므로 vertex 간 연결되어있는지 확인하는데 오래 걸린다. Space Complexity 는 $O(E + V)$이다. Sparse graph 를 표현하는데 적당한 방법이다.


#### Graph Travarsal 

그래프는 정점의 구성 뿐만 아니라 간선의 연결에도 규칙이 존재하지 않기 때문에 탐색이 복잡하다. 따라서 그래프의 모든 정점을 탐색하기 위한 방법은 다음의 두 가지 알고리즘을 기반으로 한다.

1. DFS (Depth First Search)

    그래프 상에서 존재하는 임의의 한 정점으로부터 연결되어 있는 한 정점으로만 나아가는 방법을 우선하여 탐색한다. Stack을 사용하여 구현할 수 있고 시간 복잡도는 $O(V + E)$ 이다.

2. BFS (Breath First Search)

    그래프 상에 존재하는 임의의 한 정점으로부터 연결되어 있는 모든 정점으로 나아간다. Tree 에서의 Level Order Traversal 형식으로 진행되는 것이다. BFS 에서는 자료구조로 Queue 를 사용한다. 시간 복잡도는 $O(V + E)$이다. 이때 간선이 가중치를 가지고 있지 않으면 (전부 동일한 가중치를 가지고 있다면) BFS로 찾은 경로가 최단 경로이다.

#### Minimum Spanning Tree

그래프 G 의 spanning tree 중 edge weight 의 합이 최소인 spanning tree를 말한다. 여기서 말하는 spanning tree란 그래프 G 의 모든 vertex 가 cycle 이 없이 연결된 형태를 말한다.

특정 Graph에서 MST를 찾아내려면 다음 2가지 알고리즘을 사용하면 된다.

1. Kruskal Algorithm

    초기화 작업으로 edge 없이 vertex 들만으로 그래프를 구성한다. 그리고 weight 가 제일 작은 edge 부터 검토한다. 그러기 위해선 Edge Set 을 non-decreasing 으로 sorting 해야 한다. 그리고 가장 작은 weight 에 해당하는 edge 를 추가하는데 추가할 때 그래프에 cycle 이 생기지 않는 경우에만 추가한다. spanning tree 가 완성되면 모든 vertex 들이 연결된 상태로 종료가 되고 완성될 수 없는 그래프에 대해서는 모든 edge 에 대해 판단이 이루어지면 종료된다.

    이때 Cycle의 여부는 Union find 알고리즘으로 판단한다. 

    시간 복잡도: $O(E \log E)$

2. Prim Algorithm

    초기화 과정에서 한 개의 vertex 로 이루어진 초기 그래프 A 를 구성한다. 그리고나서 그래프 A 내부에 있는 vertex 로부터 외부에 있는 vertex 사이의 edge 를 연결하는데 그 중 가장 작은 weight 의 edge 를 통해 연결되는 vertex 를 추가한다. 어떤 vertex 건 간에 상관없이 edge 의 weight 를 기준으로 연결하는 것이다. 이렇게 연결된 vertex 는 그래프 A 에 포함된다. 위 과정을 반복하고 모든 vertex 들이 연결되면 종료한다.

    시간 복잡도: $O(E \log V)$

## Operating System

### Process

> Process는 실행중인 프로그램으로 디스크로부터 메모리에 적재되어 CPU를 할당받을 수 있는 것을 의미한다. 

Process는 OS로 부터 Memory, File, Address Space등을 할당 받으며 이들을 총칭하여 process라고 부른다.   
OS는 Process를 관리하기 위해 특정 자료구조로 Process의 정보를 저장하고 있는데, 이것이 Process Control Block (PCB)이다. OS는 Process의 생성과 동시에 고유의 PCB를 생성하고 관리한다.  

(중요도 낮음)
PCB에는 다음과 같은 정보들이 저장된다.

* Process ID
* Process 상태
* Program Counter
* CPU Register
* CPU Scheduling 정보
* Memory 관리 정보
* I/O 상태 정보
* Accounting 정보

### Thread

> Thread는 Process의 내부에서 각자의 주소 공간이나 자원을 공유하는 하나의 실행 단위이다.

Thread는 Thread ID, Program Counter, Register 집합, 그리고 Stack으로 구성된다. 같은 Process에 속한 Thread들 끼리는 코드, 데이터 섹션, 그리고 열린 파일이나 신호와 같은 OS의 자원을 공유한다.   

### Multi-processing / Multi-Threading

* Multi-Threading의 장점
  
Multi-Process를 통해 동시에 처리하던 Task를 Multi-Threading으로 처리할 경우, 메모리 공간과 OS Resource를 절약할 수 있다.   
만약 병렬로 처리해야하는 Task가 각자의 통신을 요구하는 경우에도 Multi-Processing과는 다르게 별도의 자원을 요구하는 것이 아닌, 전역 변수 공간에 할당된 Heap 영역을 통해 데이터의 공유가 가능하다.  
심지어 Thread의 Context Switch는 Process의 그것과는 달리 Cache Memory를 비울 필요가 없기 때문에 더 빠르다.  
이러한 이유로 Multi-Threading이 Multi-Processing보다 자원 소모가 적고 더 빠른 응답 시간을 가진다. 

* Multi-Threading의 단점

Multi-Processing과는 달리, Multi-Threading은 공유 자원인 Heap 영역이 있기 때문에 동일한 자원을 동시에 접근할 수 있어서 이 부분을 신경써서 Programming을 해야한다.  
이 때문에 Multi-Threading은 동기화 작업이 필요하다.   
하지만 이를 위해서 과도한 lock을 걸어버리면 오히려 그것 때문에 병목현상이 발생하고 성능이 저하될 수 있다.

* Multi-Threading vs Multi-Processing

Multi-Threading은 Multi-Processing보다 적은 메모리를 차지하고 빠른 Context Switching이 가능하다는 장점이 있지만, 오류로 인해 하나의 Thread가 죽으면 전체 Thread가 죽을 수도 있다는 단점이 있다.   
반면 Multi-Processing은 하나의 Process가 죽더라도 다른 Process는 전혀 영향을 받지 않는다.

### Scheduling

Scheduler란 한정적인 메모리를 여러 프로세스가 효율적으로 사용할 수 있도록 다음 실행 시간에 실행할 수 있는 프로세스 중에 하나를 선택하는 역할을 일컫는다. 

이러한 Scheduling을 위해서 Scheduling Queue라는 것이 존재한다.   

* Job Queue : 현재 시스템 내에 있는 모든 프로세스의 집합
* Ready Queue : 현재 메모리 내에 있으면서 CPU 를 잡아서 실행되기를 기다리는 프로세스의 집합
* Device Queue : Device I/O 작업을 대기하고 있는 프로세스의 집합

각각의 Queue 에 프로세스들을 넣고 빼주는 스케줄러에도 크게 세 가지 종류가 존재한다.

1. Long Term Scheduler

    long-term스케줄러는 보조기억장치에 존재하는 프로세스 중 어떤 프로세스를 주 메모리로 load할 지 결정하는 역할을 해준다.   
    메모리는 한정되어 있는데 많은 프로세스들이 한꺼번에 메모리에 올라올 경우, 대용량 메모리(일반적으로 디스크)에 임시로 저장된다. 이 pool 에 저장되어 있는 프로세스 중 어떤 프로세스에 메모리를 할당하여 ready queue 로 보낼지 결정하는 역할을 한다.


2. Short Term Scheduler

    CPU 와 메모리 사이의 스케줄링을 담당한다. Ready Queue 에 존재하는 프로세스 중 어떤 프로세스를 running 시킬지 결정하는 Scheduler이다. 

3. Medium Term Scheduler

    여유 공간 마련을 위해 프로세스를 통째로 메모리에서 디스크로 쫓아내는 역할을 한다.   
    현 시스템에서 메모리에 너무 많은 프로그램이 동시에 올라가는 것을 조절하는 스케줄러.

그리고 CPU를 잘 사용하기 위해서 Process를 잘 배정할 필요가 있다. 이 과정을 CPU Scheduling이라고 부른다. 

이러한 CPU Scheduling은 선점과 비선점 Scheduling으로 나뉜다. 

1. 비선점 Scheduling

    프로세스 종료 or I/O 등의 이벤트가 있을 때까지 실행 보장 (처리시간 예측 용이함)

    이러한 비선점 Scheduling에도 종류가 여러가지가 있다.

    * FCFS (First Come First Served)
    * SJF (Shortest Job First)
    * HRN (Hightest Response-ratio Next)

2. 선점 Scheduling

    OS가 CPU의 사용권을 선점할 수 있는 경우, 강제 회수하는 경우 (처리시간 예측 어려움)

    이러한 선점 Scheduling에도 종류가 여러가지가 있다.

    * Priority Scheduling
    * Round Robin
    * Multilevel-Queue
    * Multilevel-Feedback-Queue

이렇게 여러가지 CPU Scheduling 알고리즘이 있는데 이를 평가할 수 있는 척도가 있어야 한다. 그래야 어떤 알고리즘이 더 좋은지 알 수 있기 때문이다.   
이러한 척도는 총 5가지가 있다.

1. CPU utilization : 계속 CPU를 사용할수 있도록 하기 위함. CPU가 컴퓨터가 켜져있는 동안 계속 동작한다면 100% 그리고 CPU가 한번도 동작하지 않는다면 0%이다.

2. Throughput : 특정 시간동안 실행될 수 있는 프로세스 몇개인가?

3. Turnaround time : 메모리에 들어가기 위해 기다리는 시간 + ready queue에서 기다리는 시간 + CPU 실행하는 시간 + I/O하는 시간

4. Waiting time : 프로세스가 ready queue에서 기다리는 시간의 총 합

5. Response time : 프로세스가 응답을 시작할 때까지 걸리는 시간

### System Call

fork( ), exec( ), wait( )와 같은 것들을 Process 생성과 제어를 위한 System call이라고 부름.

* fork, exec는 새로운 Process 생성과 관련이 되어 있다.
* wait는 Process (Parent)가 만든 다른 Process(child) 가 끝날 때까지 기다리는 명령어임.


1. Fork

    새로운 Process를 생성할 때 사용하는 System Call임.  
    

2. Exec

    child에서는 parent와 다른 동작을 하고 싶을 때는 exec를 사용할 수 있음.

3. Wait

    child 프로세스가 종료될 때까지 기다리는 Task이다. 

### Synchronization

동기화의 정의를 다루기전에 먼저 Race Condition과 Critical Section의 개념을 다룰 필요가 있다.

> Race Condition이란 여러 Process들이 동시에 데이터에 접근하는 상황에서 어떤 순서로 데이터에 접근하느냐에 따라서 결과값이 달라질 수 있는 상황을 말한다.

> Critical Section이란 동일한 자원을 접근하는 작업을 실행하는 코드 영역을 일컫는다. 또는 코드 상에서 Race Condition이 발생할 수 있는 특정 부분이라고도 말한다.

*Process의 동기화란이런 Critical Section에서 발생할 수 있는 Race Condition을 막고 데이터의 일관성을 유지하기 위해서 협력 Process간의 실행 순서를 정해주는 메커니즘이다.*

이러한 Critical Section상에서 발생할 수 있는 문제를 해결하기 위해서는 다음 조건들을 만족해야 한다.

1. Mutual Exculsion (상호 배제)

    이미 한 Process가 Critical Section에서 작업중이면 다른 모든 Process는 Critical Section에 진입하면 안된다.

2. Progress (진행)

    Critical Secion에서 작업중인 Process가 없다면 Critical Section에 진입하고자 하는 Process가 존재하는 경우에만 진입할 수 있어야 한다.

3. Bounded Waiting (한정 대기)

    Process가 Critical Section에 들어가기 위해 무한정 기다려서는 안된다.

이러한 Critical Section 문제를 해결하기 위해서 다음과 같은 해결책들이 존재한다.

* Mutex Lock (Mutual Exclusion Lock)

Mutex Lock은 이미 하나의 Process가 Critical Section에서 작업중이면 다른 Process들은 Critical Section에 진입할 수 없도록 한다.   
하지만 이러한 방식은 *Busy Waiting*의 단점이 있다. Critical Section에 Process가 존재할때 다른 Process들이 계속 Critical Section에 진입하려고 시도하면서 CPU를 낭비하는 것이다.

이렇게 lock이 반환될때까지 계속 확인하면서 Process가 대기하고 있는 것을 Spin Lock이라고도 한다. Spin lock은 Critical Section에 진입을 위한 대기 시간이 짧을때 즉, Context Switching을 하는 비용보다 기다리는게 더 효율적인 특수한 상황에서 사용된다.

* Semaphores

Semaphore는 Busy Waiting이 필요 없는 동기화 도구이며, 여러 Process/Thread가 Critical Section에 진입할 수 있게 하는 Locking 메커니즘이다. 

Semaphore는 Counter를 활용하여 동시에 자원에 접근할 수 있는 Process를 제한한다. 단순히 생각해서 *공유 자원의 개수*라고 생각하면 편하다. 이러한 Semaphore에는 2 종류가 있다.

1. Counting Semaphore: 자원의 개수만큼 존재하는 Semaphore
2. Binary Semaphore: 0과 1뿐인 Semaphore

### Memory

#### Memory Management Strategy

각각의 프로세스 는 독립된 메모리 공간을 갖고, 운영체제 혹은 다른 프로세스의 메모리 공간에 접근할 수 없는 제한이 걸려있다. 단지, 운영체제 만이 운영체제 메모리 영역과 사용자 메모리 영역의 접근에 제약을 받지 않는다.  


#### Locality of Cache



## Database

### SQL vs NoSQL

SQL과 NoSQL은 각자 다음과 같은 뜻이 있다.

SQL: Structured Query Language
NoSQL: Not Only SQL

각자 쓰이는 장소가 다르다. SQL은 관계형 데이터베이스(RDBMS)에서 사용되고 NoSQL은 비 관계형 데이터베이스에서 사용된다. 

이들은 각각 특성이 있을뿐, 장단으로 명확하게 나뉘지 않는다.

SQL의 특성:

* 명확하게 정의된 스키마가 있으며 데이터의 무결성을 보장한다.
* 관계는 각 데이터를 중복없이 한번만 저장하면 된다.
* 데이터의 스키마를 사전에 계획하고 알려야 한다. (나중에 수정하기 매우 힘들다)
* 관계를 맺고 있어서 Join문이 많은 복잡한 쿼리를 사용해야할 수도 있음
* 대체적으로 수직적 확장만 가능함. (DB 서버의 성능을 올리는 방향)

NoSQL의 특성:

* 명확하게 정의된 스키마가 없으며 유연한 데이터 구조를 지님
* 데이터를 Application이 필요로 하는 형태로 저장할 수 있음. 데이터의 I/O가 빨라짐.
* 수직 및 수평적 확장이 용이함. 
* 데이터의 중복을 계속 업데이트할 필요가 있음.

### Index

Index는 DB의 검색 속도를 높이기 위한 테크닉중 하나이다.   

Index를 통해서 DB는 Table을 Full Scan하지 않고도 원하는 데이터를 찾을 수 있다. 이러한 Index는 대개 B+ Tree 구조로 저장되어 있어서 Index를 통한 검색을 더욱 빠르게 진행할 수 있다.   

하지만 Index를 사용하게 되면 한 페이지를 동시에 수정할 수 있는 병행성이 줄어들고 Insert, Update, Delete 등의 Write 작업의 성능에 영향을 미칠 수 있다. 또한 Index를 관리하기 위해 저장 공간의 약 10%가 사용되므로 Write 연산이 빈번한 Table에 Index를 잘못 걸게 되면 오히려 성능이 저하될 수 있다.

### Transaction

Transaction이란 DB의 상태를 변화시키기 위한 수행 단위를 일컫는다.  
예를 들어서, 은행에서 송금을 처리하는 퀴리문들이 있다고 가정하자. 그렇다면 다음의 쿼리문들이 하나의 Transaction이다. 

> 사용자 A의 계좌에서 만원을 차감한다 : UPDATE 문을 사용해 사용자 A의 잔고를 변경  
> 사용자 B의 계좌에 만원을 추가한다 : UPDATE 문을 사용해 사용자 B의 잔고를 변경

위의 2개의 쿼리문이 하나의 Transaction이다. 2개의 쿼리문이 성공적으로 완료되어야만 하나의 Transaction이 성공하고 DB에 반영된다. (이를 Commit한다고 한다.)  
반대로 둘중 하나라도 실패하게 된다면 모든 쿼리문을 취소하고 DB는 원상태로 돌아간다. (이를 Rollback이라고 한다.)

RDB에서 Transaction은 다음을 엄격하게 만족시켜야 한다.  
ACID (Atomicity, Consistency, Isolation, Durability)

* Atomicity: 트랜잭션이 DB에 모두 반영되거나, 혹은 전혀 반영되지 않아야 된다.
* Consistency: 트랜잭션의 작업 처리 결과는 항상 일관성 있어야 한다.
* Isolation: 둘 이상의 트랜잭션이 동시에 병행 실행되고 있을 때, 어떤 트랜잭션도 다른 트랜잭션 연산에 끼어들 수 없다.
* Durability: 트랜잭션이 성공적으로 완료되었으면, 결과는 영구적으로 반영되어야 한다.

반대로 NoSQL DB에서는 완화된 ACID를 지원하는데, 이를 종종 BASE(Basically Avaliable, Soft state, Eventually consistent)라고 부르기도 한다.

* Basically Avaliable: 기본적으로 언제든지 사용할 수 있다는 의미를 가지고 있다.
* Soft state: 외부의 개입이 없어도 정보가 변경될 수 있다는 의미를 가지고 있다.
* Eventually consistent: 일시적으로 일관적이지 않은 상태가 되어도 일정 시간 후 일관적인 상태가 되어야한다는 의미를 가지고 있다.

BASE는 ACID와는 다르게, 데이터의 일관성보다는 가용성을 우선시하는 것이 특징이다. 어느 것이 좋다고 말을 할 수 없이 그냥 특성으로 받아들이는 것이 좋다.

### Lock

DB에서 Lock은 Transaction 처리의 순차성을 보장하기 위해서 나온 개념이다. Lock은 여러 connection에서 동시에 동일한 자원을 요청할 경우 순서대로 한 시점에는 하나의 connection만 변경할 수 있게 해주는 역할을 한다.  

Lock의 종류로는 Shared Lock과 Exclusive Lock이 있다. Shared Lock은 Read Lock이라고도 불리우며 Exclusive Lock은 Write Lock이라고도 불리운다. 

1. Shared Lock: 공유 Lock은 데이터를 읽을 때 사용되어지는 Lock이다. 이런 공유 Lock은 공유 Lock 끼리는 동시에 접근이 가능하다.
2. Exclusive Lock: 베타 Lock은 데이터를 변경하고자 할 때 사용되며, 트랜잭션이 완료될 때까지 유지된다. 베타락은 Lock이 해제될 때까지 다른 트랜잭션(읽기 포함)은 해당 리소스에 접근할 수 없다.

### deadlock

교착상태는 두 트랜잭션이 각각 Lock을 설정하고 다음 서로의 Lock에 접근하여 값을 얻어오려고 할 때 이미 각각의 트랜잭션에 의해 Lock이 설정되어 있기 때문에 양쪽 트랜잭션 모두 영원히 처리가 되지않게 되는 상태를 말한다. 교착상태가 발생하면 DBMS가 둘 중 한 트랜잭션에 에러를 발생시킴으로써 문제를 해결합니다. 교착상태가 발생할 가능성을 줄이기 위해서는 접근 순서를 동일하게 하는것이 중요하다.

교착 상태의 빈도를 낮추는 방법:

1. 트랜잭션을 자주 커밋한다.
2. 정해진 순서로 테이블에 접근한다. 트랜잭션들이 동일한 테이블 순으로 접근하게 한다.
3. 읽기 잠금 획득 (SELECT ~ FOR UPDATE)의 사용을 피한다.
4. 한 테이블의 복수 행을 복수의 연결에서 순서 없이 갱신하면 교착상태가 발생하기 쉽다, 이 경우에는 테이블 단위의 잠금을 획득해 갱신을 직렬화 하면 동시성을 떨어지지만 교착상태를 회피할 수 있다.

### Redis

> Redis는 In-Memory기반의 Key-Value 데이터 구조의 DB이다.

Redis는 메모리(RAM)에 저장해서 디스크 스캐닝이 필요없어 매우 빠른 장점이 존재한다. 캐싱도 가능해 실시간 채팅에 적합하며 세션 공유를 위해 세션 클러스터링에도 활용된다.

### High Avaliablity Strategy

#### High Avaliablity on SQL DB

MySQL 같은 SQL DB에서 HA를 구축하는 전략중에 가장 보편적인 방법은 Master-Slave Node를 구성하는 방법이다.   
Master Node는 Read/Write 연산을 둘 다 담당할 수 있고 Slave Node는 Read Only이다.  
Slave Node가 Master Node의 Binary log를 읽어서 Slave DB의 Relay log로 복사 후 반영하여 변경된 데이터를 동기화하는 방식이 잘 알려진 Replication이다.

Slave 대수에 따라 수 초의 Delay가 발생할 수 있어 실시간이 매우 중요한 데이터 읽기 트랜잭션은 Master DB에서 읽게 하고 Slave DB는 이외 읽기 트랜잭션을 적용하는 것이 일반적이다.

필자는 Proxy SQL을 사용하여 Application에서 들어오는 Transaction을 Master/Slave Node에 분류하여 처리하게 하였고 Read 연산의 경우 Slave node들에 Round Robin으로 Load Balancing을 진행하였다. 

그리고 여기서 또 중요한 점이 한가지 존재한다. Master가 죽었을 때 어떻게 Slave들이 데이터의 일관성을 유지할 것인지가 굉장히 중요한 논점이다.  

#### High Avaliablity on Redis

Redis에서 HA를 적용하는 방법은 크게 2가지가 있다.

* Master/Slave Node with Redis Sentinel

    이는 SQL에서와 마찬가지로 Read/Write 연산을 담당하는 Master Node와 Read Only 연산을 담당하는 Slave Node로 나눈 뒤에, 각 Node의 상태를 Redis Sentinel이 감시하게 하는 방법을 사용한다.  
    Redis Sentinel은 최소 3대로 Cluster를 이루어 Master의 생존 여부를 투표로 결정하고 Master가 죽은 경우에는 투표를 통해서 살아있는 Slave Node를 Master로 승격시킨다.

* Redis Cluster

    Redis Cluster의 경우, Redis Sentinel 없이 최소 3대의 Master/Slave Node가 서로를 모니터링 하면서 고 가용성을 만든다.  
    데이터세트을 여러 노드로 자동분할 하는 기능(샤딩) (최대 1000개)을 가지고 있으며 Gossip Protocol을 통해서 서로 통신을 진행한다.  

    Master Node에 장애가 발생할 경우, Master의 Slave는 Gossip Protocol을 통해 장애를 파악한다.
    이후 스스로 Master로 승격하여 Master를 대신한다. 장애가 발생한 Master가 복구될경우, 스스로 Slave로 강등

### ORM

ORM은 Object Relational Mapping의 약자로써, 풀어서 설명하자면 우리가 OOP(Object Oriented Programming)에서 쓰이는 객체라는 개념을 구현한 클래스와 RDB(Relational DataBase)에서 쓰이는 데이터인 테이블 자동으로 매핑(연결)하는 것을 의미한다.

쌩 SQL Query를 작성하는 것보다 ORM이 가지는 장단점은 다음과 같다.

장점)

* 완벽한 객체지향적 코드: ORM을 사용하면 SQL문이 아닌 Class의 Method를 통해서 DB를 조작할 수 있다. 이는 개발자가 Object Model만을 이용해서 Programming을 하는데 집중할 수 있게 해준다.
* 재사용/유지보수/리펙토링의 용이: ORM은 기존 객체와 독립적으로 작성되어있으며, Object로 작성되어 있기 때문에 재활용이 가능하다. 또한 Mapping하는 정보가 명확하기 때문에 ERD를 보는 의존도를 낮출 수 있다.
* DBMS의 종속성 하락: 객체 간의 관계를 바탕으로 SQL문을 자동으로 생성하고, 객체의 자료형 타입까지 사용할 수 있기 때문에 RDBMS의 데이터 구조와 객체지향 모델 사이의 간격을 좁힐 수 있다.

단점)

* ORM은 명확한 한계가 존재한다: Project가 커지면 그만큼 구현의 난이도가 올라가고, 이때 설계를 부실하게 하면 오히려 속도가 떨어지는 문제점이 생긴다. 또한 그 뿐만 아니라 일부 SQL의 경우에는 속도를 위해서 별도의 튜닝이 필요한데, 이때는 결국 SQL문을 직접 작성해야 한다.
* Object-Relation 간의 불일치   
  * 세분성: 경우에 따라서 데이터베이스에 있는 테이블 수보다 더 많은 클래스를 가진 모델이 생길 수 있다.
  * 상속성: RDB에서는 객체지향의 특징인 상속 기능이 존재하지 않는다.
  * 연관성: 객체지향 언어는 방향성이 있는 객체의 참조를 사용하여 연관성을 나타내지만, RDBMS는 방향성이 없는 foreign key를 통해서 이를 나타낸다.


## Web

### TCP / UDP

우선 TCP 통신과 UDP 통신부터 알아보자.

* TCP 통신

    TCP는 TCP/IP 모델의 3계층, Transport 계층의 프로토콜이다.   
    기본적으로 신뢰성 있는 데이터 전송을 위한 연결 지향형 프로토콜이다. 
    TCP는 우선 HandShake 방식을 통해서 데이터를 송/수신한다. 예를 들어서 TCP에서는 송신자가 데이터를 전송할 경우, 다음과 같은 검증 과정을 거친다.

    1. 데이터를 제대로 수신했는지
    2. 에러는 발생하지 않았는지
    3. 수신 속도에 비해 전송 속도가 너무 빠르지는 않았는지 (흐름 제어)
    4. 네트워크가 혼잡하지는 않은지 (혼잡 제어)

    이러한 기본적인 사항들을 지키는 Protocol이기 때문에 TCP를 높은 신뢰성을 보장한다고 말한다.

    여기서 흐름 제어와 혼잡 제어를 보다 더 정확하게 설명하면 다음과 같다.

    * 흐름 제어

        수신측이 송신측보다 데이터 처리 속도가 빠르면 문제없지만, 송신측의 속도가 빠를 경우 문제가 생긴다. 수신측에서 제한된 저장 용량을 초과한 이후에 도착하는 데이터는 손실 될 수 있으며, 만약 손실 된다면 불필요하게 응답과 데이터 전송이 송/수신 측 간에 빈번히 발생한다.  
        아래서 설명하겠지만, Receiver Buffer의 크기만큼 수신자는 데이터를 읽을 수 있다. 따라서 수신측의 데이터 처리 속도가 송신측보다 느리면 문제가 발생한다.

        해결 방법:
        
        * Stop and Wait : 매번 전송한 패킷에 대해 확인 응답을 받아야만 그 다음 패킷을 전송하는 방법
        * Sliding Window (Go Back N ARQ): 수신측에서 설정한 윈도우 크기만큼 송신측에서 확인응답없이 세그먼트를 전송할 수 있게 하여 데이터 흐름을 동적으로 조절하는 제어기법

    * 혼잡 제어

        송신측의 데이터는 지역망이나 인터넷으로 연결된 대형 네트워크를 통해 전달된다. 만약 한 라우터에 데이터가 몰릴 경우, 자신에게 온 데이터를 모두 처리할 수 없게 된다.   
        이런 경우 호스트들은 또 다시 재전송을 하게되고 결국 혼잡만 가중시켜 오버플로우나 데이터 손실을 발생시키게 된다. 따라서 이러한 네트워크의 혼잡을 피하기 위해 송신측에서 보내는 데이터의 전송속도를 강제로 줄이게 되는데, 이러한 작업을 혼잡제어라고 한다.


    이러한 TCP는 전체적으로 다음과 같은 전송 과정을 거쳐서 데이터를 전송한다.

    * Application Layer: Sender Application layer가 socket에 data를 전송함.
    * Transport layer: data를 segment에 감싼다. 그리고 network layer에 넘겨준다.
    * Network layer - physical layer에서 도착지에 데이터를 전송한다. 이때 Sender의 send buffer에 data를 저장하고 receiver는 receive buffer에 data를 저장한다.
    * Application에서 준비가 되면 buffer에 있는 데이터를 읽음 (흐름 제어의 핵심은 이 receiver buffer가 넘치지 않게 하는 것에 있다.)
    * Receiver는 Receive Window : receiver buffer의 남은 공간을 홍보함.

* UDP

    UDP의 경우에는 TCP 처럼 신뢰성을 보장하지 않는다. 하지만 그만큼 신뢰성을 보장하기 위한 흐름 제어/혼잡 제어같은 처리가 필요 없어서 overhead가 발생하지 않고 빠른 처리 및 경량화된 헤더를 가지게 된다. 이러한 이유 때문에 UDP는 다음과 같은 연속성과 성능이 더 중요한 분야에 사용되게 된다.

    * 인터넷 전화
    * 온라인 게임
    * 멀티미디어 스트리밍 

### TCP 3/4way handshake    

TCP 통신은 연결을 수립/종료할때 각각 3/4 way handshake를 통해서 진행하게 된다.   
먼저 연결을 수립하는 과정인 3 way handshake를 생각해 보자.   

3 way handshake는 총 3번의 통신 과정을 거쳐서 내가 누구와 통신중인지, 내가 받아야할 데이터의 sequence number는 몇번인지 등등의 정보를 주고 받으면서 연결의 상태를 생성하게 된다.

1. 먼저 수신자는 LISTEN 상태로 대기를 하게 된다. 이때, 수신자는 요청자가 요청을 하기 전까지는 대기를 하게 되는데 이 상태를 수동 개방이라고 하고 수신자를 Passive Opener라고 한다.
2. 요청자가 수신자에게 연결 요청을 하면서 랜덤한 숫자인 Sequence Number를 생성하여 SYN 패킷에 담아 보낸다. 이 Sequence Number를 활용하여 계속 새로운 값을 만들고 서로 확인하며 연결 상태와 패킷의 순서를 확인하게 된다.
3. 수신자는 요청자가 보낸 SYN 패킷을 잘 받았으면 SYN + ACK 패킷을 보낸다. 여기에는 제대로된 Sequence Number를 받았다는 의미인 승인 번호값이 들어있으며 이는 Sequence Number + 상대방이 보낸 데이터의 크기(byte)의 값을 가지고 있다. 하지만 맨 처음에는 1을 더해서 보낸다.
4. 요청자가 이러한 ACK의 값이 Sequence Number와 1차이가 나는 것을 확인하였으면 다시 확인차 ACK 패킷을 보내 연결이 수립된다.

1번이 초기 상태이고 2~4번부터 3 Way Handshake의 과정이다. 이렇게 3단계의 과정을 거쳐서 TCP의 연결을 수립한다.

이제 4 Way Handshake에 대해서 알아보도록 하겠다.   
이는 연결을 종료할때 사용하는 과정이다. 굳이 이러한 과정을 거쳐서 연결을 종료하는 이유는 연결을 그냥 끊었다가는 다른 한쪽이 연결이 끊겼는지 아닌지 알 방법이 없기 때문이다. 이러한 과정을 총 4번의 통신을 거치기에 4 way handshake라고 한다.

1. 먼저, 연결을 종료하고자 하는 요청자가 상대에게 FIN 패킷을 보내면서 FIN_WAIT 상태로 들어가게 된다. 
2. 수신자는 요청자가 보낸 Sequence Number + 1로 승인 번호를 만들어서 다시 요청자에게 ACK로 응답해주면서 CLOSE_WAIT 상태로 들어가게 된다. 이때 요청자는 자신이 받은 값이 Sequence Number + 1이 맞는지 다시 확인해 본다.
3. 수신자는 자신이 처리할 데이터가 더 없다면 FIN을 다시 한번 요청자에게 보내고 LAST_ACK 상태에 들어간다.
4. 요청자는 FIN을 수신자로부터 받고 다시 ACK 패킷을 수신자에게 보내준다. 그럼 수신자는 완벽하게 요청을 끊는 것이다.

### TLS/SSL  

TLS는 Transport Layer Security의 약자이고 SSL은 Secure Socket Layer의 약자이다.  
SSL은 암호화 기반의 인터넷 보안 프로토콜이다. TLS의 전신인 이 프로토콜은 전달되는 모든 데이터를 암호화하고 특정 유형의 사이버 공격 또한 차단한다.   
하지만 SSL은 3.0 이후로 업데이트 되지 않고 있으며 여러 취약점들이 알려져 사용이 중된되었다. 따가서 이렇게 개발된 최신 암호화 프로토콜이 TLS이다.

TLS 는 SSL의 업데이트 버전으로 SSL의 최종버전인 3.0과 TLS의 최초버전의 차이는 크지않으며, 이름이 바뀐것은 SSL을 개발한 Netscape가 업데이트에 참여하지 않게 되어 소유권 변경을 위해서였다고 한다.  
결과적으로 TLS는 SSL의 업데이트 버전이며 명칭만 다르다고 볼 수 있다.

SSL/TLS는 handshake를 통해서 실제 데이터를 주고 받기 전에 서로 연결 가능한 상태인지 파악을 하게 된다. 

1. Client Hello 

    Client는 Server에게 다음과 같은 정보를 넘겨주면서 Hello라는 패킷을 보낸다.

    * Client에서 생성한 Random 값
    * Client가 지원하는 암호화 방식
    * (이전에 HandShake가 일어났을 경우) Session ID

2. Server Hello

    Server는 Client에게 다음과 같은 정보를 전달한다.

    * Server에서 생성한 Random 값
    * Client가 제공한 암호화 방식중 Serve가 선택한 암호화 방식
    * 인증서

3. Certificate 

    Server가 자신의 TLS/SSL 인증서를 Client에게 전달한다. Client는 Server가 보낸 CA(Certifiacte Authority)의 개인키로 암호화된 SSL 인증서를 이미 모두에게 공개된 CA의 공개키를 사용해 복호화 한다. 즉, 인증서를 검증하는 과정을 거친다. 

4. Server Key Exchange / Hello Done

    Server Key Exchange는 Server의 공개키가 TLS/SSL 인증서 내부에 없을 경우 Server가 직접 이를 전달하는 것을 의미한다. 공개키가 인증서 내부에 있으면 이 과정은 생략된다.   

5. Client Key Exchange

    실제로 데이터를 암호화 하는 키(대칭키)를 Client가 생성해서 SSL 인증서 내부에서 추출한 Server의 공개키를 이용해 암호화한 뒤에, Server에 전달한다. 여기서 전달된 대칭키가 SSL Handshake의 목적이자, 실제 데이터를 암호화 하게될 비밀키이다. 

6. Change Cipher Spec / Finished

    Client, Server가 모두 서로에게 보내는 Packet으로 교환할 정보를 모두 교환환 뒤에, 통신을 할 준비가 되었음을 알리는 Packet이다. 이것으로 Handshake는 종료된다.

이를 요약하면 다음과 같다.

> ClientHello(암호화 알고리즘 나열 및 전달) - > Serverhello(암호화 알고리즘 선택) - > Server Certificate(인증서 전달) - > Client Key Exchange(데이터를 암호화할 대칭키 전달) - > Client / ServerHello done (정보 전달 완료) - > Fisnied (SSL Handshake 종료)

### HTTP 1.0/1.1/2.0

HTTP의 명확한 정의는 다음과 같다. 

> WWW 상에서 정보를 주고 받을 수 있는 프로토콜. 주로 HTML 문서를 주고 받는 데에 쓰인다.

HTTP는 TCP위에서 돌아가는 Application Level의 Protocol이며 기본적으로 Stateless Protocol이다.

HTTP는 1.0과 1.1, 2.0 그리고 최근데 3.0이 나왔는데, 1.0과 1.1의 차이를 비교하면서 1.0과 1.1을 논하도록 하고 1.1과 2.0의 차이를 논하도록 하겠다.

HTTP 1.0과 1.1의 차이점은 다음과 같다.

* Connection 유지

    1.0과 1.1의 차이는 TCP Session을 유지할 수 있느냐 없느냐이다. TCP는 Handshake를 통해서 연결을 수립하는데 HTTP/1.0를 통해서 통신을 할때마다 이를 끊고 다시 수립하는 과정이 있어야 했다. 이를 보완하려면 Persistence Connection을 지원하는 Client는 Server에게 요청을 보낼때 connection:keep-alive header를 통해서 Persistence Connection을 수립해야 했다. 그리고 Server는 응답에서 똑같은 Header를 넣어서 보낸다.

    하지만 HTTP/1.1은 이를 기본으로 지원한다. 즉, 굳이 Connection Header 를 사용하지 않아도 모든 요청과 응답은 기본적으로 Persistent Connection을 지원하도록 되어 있으며 필요 없을 경우(HTTP 응답 완료 후 TCP 연결을 끊어야 하는 경우) Connection Header를 사용한다.

    Keep-Alive 를 통해 얻을 수 있는 장점으로는 단일 시간 내의 TCP 연결을 최소화 함으로써 CPU와 메모리 자원을 절약할 수 있고 네트워크 혼잡이나 Latency 에 대한 경우의 수를 줄일 수 있다는 점이다.

    * Pipelining

        HTTP 1.0 에서는 HTTP 요청들의 연결을 반복적으로 끊고 맺음으로서 서버 리소스적으로 비용을 요구한다. 

        HTTP 1.1 에서는 다수의 HTTP Request 들이 각각의 서버 소켓에 Write 된 후, Browser는 각 Request들에 대한 Response를 순차적으로 기다리는 문제를 해결하기 위해 여러 요청들에 대한 응답 처리를 뒤로 미루는 방법을 사용한다.

        Pipelining이 적용되면 하나의 Connection으로 다수의 Request와 Response를 처리할 수 있고 Network Latency를 줄일 수 있게 된다.

* Host Header

    HTTP 1.0 환경에서는 하나의 IP에 여러 개의 도메인을 운영할 수 없었지만 HTTP 1.1 부터 가능하게 된다.

    HTTP 1.1 에서 Host 헤더의 추가를 통해 비로소 Virtual Hosting이 가능해졌다.

* 향상된 Authentication Prcedure

    HTTP 1.1 에서 다음 2개의 헤더가 추가된다.

    * proxy-authentication  
    * proxy-authorization  

    실제 Server에서 Client 인증을 요구하는 www-authentication 헤더는 HTTP 1.0 에서부터 지원되어 왔으나, Client와 Server 사이에 프록시가 위치하는 경우 프록시가 사용자의 인증을 요구할 수 있는 방법이 없었다.

그리고 다음과 같은 특성을 가진 HTTP/2.0이 나왔다.

* Multiplexed Streams (한 커넥션에 여러개의 메세지를 동시에 주고 받을 수 있음)
* 요청이 커넥션 상에서 다중화 되므로 HOL(Head Of Line) Blocking 이 발생하지 않음
* Stream Prioritization (요청 리소스간 의존관계를 설정)
* Header Compression (Header 정보를 HPACK 압축 방식을 이용하여 압축 전송)
* Server Push (HTML문서 상에 필요한 리소스를 클라이언트 요청없이 보내줄 수 있음)
* 프로토콜 협상 메커니즘 - 프로토콜 선택, 예. HTTP / 1.1, HTTP / 2 또는 기타.
* HTTP / 1.1과의 높은 수준의 호환성 - 메소드, 상태 코드, URI 및 헤더 필드
* 페이지 로딩 속도 향상

기존 HTTP는 Body가 문자열로 이루어져 있지만, HTTP 2.0 부터는 Binary framing layer 라고 하는 공간에 이진 데이터로 전송된다.

### HTTP Cache

HTTP Cache는 이미 가져온 데이터나 계산된 결과값의 복사본을 저장함으로써 처리 속도를 향상시키며, 이를 통해 향후 요청을 더 빠르게 처리할 수 있다. 

브라우저가 시도하는 모든 HTTP 요청은 먼저 브라우저 캐시로 라우팅되어, 요청을 수행하는데 사용할 수 있는 유효한 캐시가 있는지를 먼저 확인한다. 만약 유효한 캐시가 있으면, 이 캐시를 읽어서 불필요한 전송으로 인해 발생하는 네트워크 대기시간, 데이터 비용을 모두 상쇄한다.

요약하자면 Cache의 효과는 다음과 같다.

* 불필요한 데이터 전송을 줄여 네트워킹 비용을 줄여준다.
* 거리로 인한 지연시간을 줄여 웹페이지를 빨리 불러올 수 있게 된다.
* 서버에 대한 요청을 줄여 서버의 부하를 줄인다.


### RESTful API

Representational State Transfer의 약자로 REST이다. 이는 여러 특성을 가지고 있는데 이 특성에 부합하는 API를 RESTful API라고 한다.   
결국 REST는 다음과 같이 정의된다.

> 자원을 이름(자원의 표현)으로 구분하여 해당 자원의 상태(정보)를 주고 받는 모든 것을 의미한다.

자세한 정의를 말해보자면 자원의 표현은 HTTP URI로 하며 HTTP Method를 통해 해당 자원의 CRUD를 적용하는 것을 RESTful 하다고 표현한다.

Restful API는 흔히 Stateless라고 표현하고 이의 특징이 존재한다.  
Restful API를 구현하는 절대 조건은 서버가 Client의 정보를 어느 것도 저장하지 않는다는 것이다. 이렇게 구현하는 절대적인 이유는 서버의 Scalablity이다.  

### Sync / Async / Blocking / Non-Blocking 

제목의 4가지 개념은 각기 다른 것이다. Server에서 어떻게 요청을 처리할 것인지를 나타내고 있다. 각각 다음과 같은 의미를 지닌다.

* Synchronous: A함수가 B 함수를 호출할때, B 함수의 결과를 A함수에서 처리함.
* ASynchronous: A함수가 B 함수를 호출할때, B 함수의 결과를 B함수에서 처리함. (Callback)
* Blocking: A함수가 B함수를 호출할때, B함수가 자신의 작업이 종료되기 전까지 A함수에게 제어권을 넘겨주지 않음
* Non-Blocking: A 함수가 B 함수를 호출할때, B함수가 바로 제어권을 A 함수에게 넘겨주면서 A 함수가 다른 일을 할 수 있도록 함.

### Blocking IO / Non-Blocking IO

I/O 작업은 Kernel Level에서만 수행이 가능하다. 따라서 Process나 Thread는 Kernel에게 I/O를 요청해야만 한다.

이때, I/O를 Blocking 방식으로 처리하느냐 Non-Blocking 방식으로 처리하느냐에 따라서 효율성이 다르다.

1. Blocking I/O

    Blocking 방식으로 I/O를 구현하면 다음과 같이 I/O를 처리해야 한다.

    * Process나 Thread가 Kernel에게 I/O를 요청하는 함수를 호출함.  
    * Kernel이 작업을 완료하면 작업 결과를 반환받음.  

    이렇게 되면 I/O 작업이 진행되는 동안 Process/Thread는 자신의 작업을 중단한 채로 대기한다. 이는 Resource의 낭비가 심하다. (I/O 작업이 CPU를 거의 사용하지 않으므로)

    만약 여러 Client가 접속하는 서버를 Blocking 방식으로 구현하려는 경우 client별로 별도의 Thread를 생성해서 처리해야한다. 이로인해 많아진 Thread들의 Context Switching 비용이 늘어나서 효율성이 감소하게 된다. 

2. Non-Blocking I/O

    Non-Blocking 방식으로 I/O를 구현하면 I/O 작업이 진행되는 동안 User Process의 작업을 중단하지 않는다.

    * User Process가 recvfrom 함수 호출 (커널에게 해당 Socket으로부터 data를 받고 싶다고 요청함)
    * Kernel은 이 요청에 대해서, 곧바로 recvBuffer를 채워서 보내지 못하므로, "EWOULDBLOCK"을 return함.
    * Blocking 방식과 달리, User Process는 다른 작업을 진행할 수 있음.
    * recvBuffer에 user가 받을 수 있는 데이터가 있는 경우, Buffer로부터 데이터를 복사하여 받아옴. 이때, recvBuffer는 Kernel이 가지고 있는 메모리에 적재되어 있으므로, Memory간 복사로 인해, I/O보다 훨씬 빠른 속도로 data를 받아올 수 있음.
    * recvfrom 함수는 빠른 속도로 data를 복사한 후, 복사한 data의 길이와 함께 반환함.


## Javascript/Nodejs

### Javascript

* var, let, const 차이점

var: 함수 스코프를 갖는다. 변수를 한번 더 선언해도 에러가 발생하지 않는다. 그래서 es6 업데이트로 let과 const 도입되었다.

let : 블록 스코프를 갖는다. 예를들어 if문 for문 while문 중괄호{}안에 선언 되었을 경우, 변수가 그 안에서 생성되고 사라진다.

const : 블록 스코프, 상수를 선언 할 때 사용한다. 선언과 동시에 할당 되어야합니다. (재할당하면 오류발생)

* hoisting

hoisting은 코드가 실행하기 전 변수선언/함수선언이 해당 스코프의 최상단으로 끌어 올려진 것 같은 현상을 말한다.

* Temporal Dead Zone

const와 let은 범위가 지정된 변수를 갖는다.
이 둘의 호이스팅이 실행 되기전 까지 액세스 할수 없는 것을 TDZ라 한다. 

> 즉, let/const선언 변수는 호이스팅되지 않는 것이 아니다.  
> 스코프에 진입할 때 변수가 만들어지고 TDZ(Temporal Dead Zone)가 생성되지만, 코드 실행이 변수가 실제 있는 위치에 도달할 때까지 액세스할 수 없는 것이다.   
> let/const변수가 선언된 시점에서 제어흐름은 TDZ를 떠난 상태가 되며, 변수를 사용할 수 있게 된다.

### NodeJS

* NodeJS

NodeJS는 Single Threaded Event Driven Asynchronous Non-Blocking Javascript Engine이다. 하나의 스레드로 동작하지만, Asynchronous I/O 작업을 통해 요청들을 서로 Blocking하지 않는다.  
즉, 동시에 많은 요청을 비동기로 처리함으로써 Single Threaded이더라도 많은 요청들을 Asynchronous하게 처리할 수 있다.

어떻게 그것이 가능하느냐? Single Threaded로 동작하면서 Blocking 작업을 어떻게 Non Blocking으로 Asynchronous하게 처리할 수 있을까?  
그것은 NodeJS에서 일부 Blocking이 필요한 작업을 libuv의 Thread Pool에서 수행되기 때문이다.    


* Event Loop (JS Engine)

Event Loop라는 것을 설명하기 전에 Event Driven의 개념부터 짚고 넘어가야 한다.  

> Event Driven이라는 것은 Event가 발생할때 미리 지정해둔 작업을 수행하는 방식을 의미한다.

NodeJS는 EventListener에 등록해둔 Callback함수를 수행하는 방식으로 이를 처리한다.  

Event Loop는 Main Thread 겸 Single Thread로써 Busniess Logic을 수행한다. 이를 수행하는 도중에 Blocking I/O 작업을 만나면 Kernel Asynchronous 또는 자신의 Worker Thread Pool에게 넘겨주는 역할을 한다. GC 또한 Event Loop에서 돌아가고 있다.

위에서 Blocking Task(api call, DB Read/Write 등)들이 들어오면 Event Loop가 uv_io에게 내려준다고 하였다. libuv는 Kernel단(윈도우의 경우 IOCP, 리눅스는 AIO)에서 어떤 비동기 작업들을 지원해주는지 알고 있기때문에, 그런 종류의 작업들을 받으면, 커널의 Asynchronous 함수들을 호출한다. 작업이 완료되면 시스템콜을 libuv에게 던져준다. libuv 내에 있는 Event Loop에게 Callback으로서 등록된다.  

그러한 Event Loop는 다음과 같은 phase로 동작하고 있다.   
각 phase들은 Queue를 가지고 있으며 이 Queue에는 특정 이벤트의 Callback들을 넣고 Event Loop가 해당 Phase를 호출할때 실행된다. 

> *Timers*: setTimeout()과 setInterval과 같은 Timer 계열의 Callback들이 처리된다.  
> *I/O Callbacks*: Close Callback, Timer로 스케쥴링된 Callback. setImmediate()를 제외한 거의 모든 Callback들을 집핸한다.  
> *idle, prepare*: 내부용으로만 사용 모든 Queue가 비어있으면 idle이 되서 tick frequency가 떨어짐  
> *poll*: Event Loop가 uv__io_poll()을 호출했을때 poll Queue에 있는 Event, Callback들을 처리함.  
> *check*: setImmediate() Callback은 여기서 호출되고 집행된다.  
> *Close Callback*: .on('close', ...) 같은 것을이 처리됨. 

Event Loop는 round-robin 방식으로 Nodejs Process가 종료될때까지 일정 규칙에 따라 여러개의 Phase들을 계속 순회한다. Phase들은 각각의 Queue들을 관리하고, 해당 Queue들은 FIFO 순서로 Callback Function들을 처리한다.

* NodeJS module

> module.exports: nodejs에서 .js파일을 만들어서 객체를 따로 모아놓은 것들을 module이라고 부른다. 그것을 외부로 export하기 위해서 module.exports 객체로 만들어 놓은 것이다.   
> exports: 이는 module.exports를 call by reference로 가리키고 있는 변수이다. 따라서 이를 바로 쓸 수는 없고 다른 property로 지정해서 써야한다. 

* Async/Await, Promise

Promise는 JS Asynchoronous 처리에 사용되는 객체이다.  
그리고 이 Promise 객체를 반환하게 하는 decorator가 async이고 Async 함수 안에서 다른 Async 작업의 Blocking을 담당하는 것이 await이다.

## Object Oriented Programming Basic

### Properties of OOP

OOP에는 4가지 특성이 있다. 

1. 추상화: 필요로 하는 속성이나 행동을 추출하는 작업
2. 캡슐화: 낮은 결합도를 유지할 수 있도록 설계하는 것
3. 상속: 일반화 관계(Generalization)라고도 하며, 여러 개체들이 지닌 공통된 특성을 부각시켜 하나의 개념이나 법칙으로 성립하는 과정
4. 다형성: 서로 다른 클래스의 객체가 같은 메시지를 받았을 때 각자의 방식으로 동작하는 능력

객체 지향 설계 과정
* 제공해야 할 기능을 찾고 세분화한다. 그리고 그 기능을 알맞은 객체에 할당한다.
* 기능을 구현하는데 필요한 데이터를 객체에 추가한다.
* 그 데이터를 이용하는 기능을 넣는다.
* 기능은 최대한 캡슐화하여 구현한다.
* 객체 간에 어떻게 메소드 요청을 주고받을 지 결정한다.

### 5 Principle  

SOLID라고 부르는 5가지 설계 원칙이 존재한다.

* SRP(Single Responsibility) - 단일 책임 원칙

    클래스는 단 한 개의 책임을 가져야 한다.

    클래스를 변경하는 이유는 단 한개여야 한다.

    이를 지키지 않으면, 한 책임의 변경에 의해 다른 책임과 관련된 코드에 영향이 갈 수 있다.


* OCP(Open-Closed) - 개방-폐쇄 원칙

    확장에는 열려 있어야 하고, 변경에는 닫혀 있어야 한다.

    기능을 변경하거나 확장할 수 있으면서, 그 기능을 사용하는 코드는 수정하지 않는다.

    이를 지키지 않으면, instanceof와 같은 연산자를 사용하거나 다운 캐스팅이 일어난다.


* LSP(Liskov Substitution) - 리스코프 치환 원칙

    상위 타입의 객체를 하위 타입의 객체로 치환해도, 상위 타입을 사용하는 프로그램은 정상적으로 동작해야 한다.

    상속 관계가 아닌 클래스들을 상속 관계로 설정하면, 이 원칙이 위배된다.


* ISP(Interface Segregation) - 인터페이스 분리 원칙

    인터페이스는 그 인터페이스를 사용하는 클라이언트를 기준으로 분리해야 한다.

    각 클라이언트가 필요로 하는 인터페이스들을 분리함으로써, 각 클라이언트가 사용하지 않는 인터페이스에 변경이 발생하더라도 영향을 받지 않도록 만들어야 한다.


* DIP(Dependency Inversion) - 의존 역전 원칙

    고수준 모듈은 저수준 모듈의 구현에 의존해서는 안된다.

    저수준 모듈이 고수준 모듈에서 정의한 추상 타입에 의존해야 한다.

    즉, 저수준 모듈이 변경돼도 고수준 모듈은 변경할 필요가 없는 것이다.