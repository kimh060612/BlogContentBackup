---
title: Recurrent Neural Network 강의 내용 part.B
date: 2022-03-05 19:26:47
categories: 
    - Deep Learning
tags: 
    - Deep Learning
    - Tensorflow2
mathjax: true
---

*본 포스트는 Hands-on Machine learning 2nd Edition, CS231n, Tensorflow 공식 document를 참조하여 작성되었음을 알립니다.*

### Index  
  
1. ~~Definition of Recurrent Neural Network(RNN)~~
2. ~~Back Propagation of RNN~~
3. Partice

<br>

### Partice

-------------------------------------------------------

진짜 여기까지 과연 읽은 사람이 있을까 싶다. 있다면 압도적 감사의 의미로 그랜절을 박고 싶은 마음이다. ㅋㅋ 

장난은 그만하고 오늘도 시작하자. 오늘도 역시 Model Subclassing API를 활용하여 간단하게 RNN을 활용하는 예제를 구현해볼 것이다.

우선 RNN을 tensorflow에서 어떻게 사용할 수 있는지부터 보자.

```python
class RNNLayer(keras.Model):
    def __init__(self, num_hidden=128, num_class=39):
        super(RNNLayer, self).__init__()
        self.RNN1 = keras.layers.SimpleRNN(num_hidden, activation='tanh', return_sequences=True)
        self.out = keras.layers.Dense(num_class, activation="softmax")
        self.out = keras.layers.TimeDistributed(self.out)
        
    def call(self, x):
        x = self.RNN1(x)
        out = self.out(x)
        return out
```

보면 알다시피 아주 간단하다. 그래서 이번 포스트에서는 간단하게 2개의 관전 포인트만을 다루려고 한다.

1. SimpleRNN layer의 특징
2. TimeDistributed layer의 특징

1번부터 차례로 시작하자.

우선 SimpleRNN을 이해하려면 입력으로는 어떤 tensor를 받아먹어서 출력으로는 어떤 tensor를 뱉어내는지를 알아야 한다.
지난 포스트에서 RNN의 구조를 보았다면 당연히 입력은 시간순서대로 벡터가 들어가니 적어도 3차원(배치까지 포함해서) 일 것이고 출력도 시간 순서대로 나와야 하니 같은 3차원이라는 것 쯤은 유추가 가능할 것이다. 하지만 실제로는 이것 또한 조절이 가능하다.

```python
keras.layers.SimpleRNN(
    units, activation='tanh', use_bias=True,
    kernel_initializer='glorot_uniform',
    recurrent_initializer='orthogonal',
    bias_initializer='zeros', kernel_regularizer=None,
    recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False,
    go_backwards=False, stateful=False, unroll=False, **kwargs
)
```

이는 tensorflow 공식 문서에 적혀 있는 SimpleRNN에 관한 내용이다. 여기서 우리가 관심 있게 봐야 할 것은 return_sequences와 return_state이다.

만약, return_sequences가 False면, SimpleRNN layer는 맨 마지막의 출력만을 뱉어낸다. 즉, 2차원 출력이 되는 것이다. ([Batch_size, output_dim]) 하지만 이 파라미터가 True이면 모든 시간의 출력을 출력으로 내뱉는다. 즉, 3차원 출력이 되는 것이다. 

그리고 return_state의 경우, 이것이 True이면 출력의 Hidden State를 출력으로 같이 내뱉는다. 즉, 출력이 tuple이 되는 것이다. (출력 벡터, hidden 벡터)

이렇게 되면 우리는 여러가지 경우의 수를 고려해서 모델을 만드는 것이 가능해진다.

1. 맨 마지막 출력만을 고려하여 예측하는 모델
2. 입력마다 다음에 나올 출력을 예측하는 모델

전자를 Many to One, 후자를 Many to Many라고 한다.

그렇다면 Many to Many를 해주기 위해서는 시간 순서에 맞춰서 *같은* Dense layer를 적용해줄 필요가 있다. 이를 위해서 필요한 것이 바로 TimeDistributed layer이다. 이는 각 시간 step에 대해서 같은 Dense layer의 weight로 연산을 진행해 준다.

하지만 기본적으로 tensorflow에서 Dense layer는 broadcast 기능을 제공한다. 따라서 굳이 TimeDistributed layer 없이도 3차원 텐서가 들어오면 맨 마지막 차원에 대해서 Dense 연산을 진행하게 된다. 

따라서, Dense만을 사용할거면 굳이 TimeDistributed layer가 필요 없다.

RNN의 포스팅은 이걸로 마치도록 하겠다. 딱히 크게 어려운 점도 없고 나중에 나올 Attention이 훨씬 더 어그로가 끌려야할 주제이기 때문이다.