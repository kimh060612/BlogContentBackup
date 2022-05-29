---
title: Deep learning Model Serving 강의 part.A
date: 2022-03-05 19:29:08
categories: 
    - Deep Learning
tags: 
    - Deep Learning
    - Tensorflow2
mathjax: true
---

### Index  
  
1. Tensorflow Serving Server
2. Tensorflow Serving Client
3. Tensorflow Serving Application

여기서는 1~2은 Part. A이고 3는 Part. B에서 다루도록 하겠다.
<br>

### Tensorflow Serving Server

-------------------------------------------------------

Tensorflow Serving은 tensorflow로 학습시킨 model을 보다 효율적으로 배포하기 위해 나온 것이다. 이번 포스트에서는 docker를 이용하여 학습시킨 ViT model을 배포하고 HTTP를 통해서 입력과 출력을 주고 받는 예제를 만들어 보겠다.

우선 그러려면 Tensorflow Serving이 뭔지부터 알아야 한다. Tensorflow serving은 tensorflow로 만들어진 모델을 서비스에 배포하기 위해서 만들어진 하나의 application이라고 생각하면 된다. 

<p align="center"><img src="https://kimh060612.github.io/img/Serving.png" width="100%"></p>

위의 그림과 같이 Docker와 함께 사용하여 우리가 만든 모델을 Tensorflow Serving에 넣어주면 HTTP/HTTPS/gRPC를 통해서 model의 inference 결과를 받아볼 수 있다.

종래의 Tensorflow model serving의 경우, 대체적으로 flask를 사용해서 REST API 형식으로 Serving을 했었다. 하지만 그것보다는 tensorflow serving의 성능이 좋다. 이유는 대체적으로 여러 요인이 있지만 대표적으로는 2가지가 있다.

1. python의 overhead 해소
2. HTTP/2.0을 사용하는 gRPC를 지원하기 때문

즉, tensorflow serving을 제일 완벽하게 활용하려면 gRPC를 사용해야 하지만 현재로써는 필자의 지식이 부족하기 때문에 일단 HTTP 통신을 사용해 보도록 하겠다. 하지만 그렇더라도 tensorflow serving이 일반 flask를 활용한 serving 보다는 빠르다.

그렇다면 어떻게 Docker를 사용하여 우리가 만든 모델을 Tensorflow Serving으로 배포할 수 있을까?
우선 예제를 보기 전에 자신만의 모델을 만들자. 필자는 ViT를 예시로 들겠다.
모델을 만들었다면, 다음과 같은 폴더 구조에 .pb와 함께 다양한 파일들이 모델로써 저장될 것이다.

```
ViT/1/
├── assets
├── variables
│   ├── variables.data-00000-of-00001
│   └── variables.index
├── keras_metadata.pb
└── saved_model.pb
```

그렇다면 우리는 이 폴더를 Docker Container에 mount해주고 8501번 포트를 열어주면 된다. 다음과 같은 명령어로 말이다.

```
sudo docker run -p 8501:8501 --name tf_serving -v /path/to/your/model/ViT/:/models/ViT -e MODEL_NAME=ViT -t tensorflow/serving &
```

그렇다면 준비 완료이다. 이제 Tensorflow Serving으로 ViT 모델을 Serving한 것이다.
물론, 실제 production 할때는 이따구로 하면 큰일난다. 어디까지 예제로 만들기 위해서 간단하게 만든 것이다.


### Tensorflow Serving Client

-------------------------------------------------------

이제 띄워놓은 ViT model에게 HTTP를 통해서 Shiba견의 이미지를 보내고 결과를 받아와보자.

<p align="center"><img src="https://kimh060612.github.io/img/Shiba.jpg" width="100%"></p>

~~(머스크형 제발 닥쳐줘)~~

한번 예제 소스코드를 다음과 같이 짜 보았다.

```python
image = cv2.imread("./image/Shiba.jpg")
image = cv2.resize(image, dsize=(32, 32))
image = np.expand_dims(image, axis=0)

data = json.dumps({"signature_name": "serving_default", "instances": image.tolist()})

headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/ViT:predict', data=data, headers=headers)
predictions = np.array(json.loads(json_response.text)["predictions"])
print(np.argmax(predictions[0]))
```

우선 데이터는 위와 같은 형식을 유지해주면 된다. 그리고 POST method로 HTTP request를 보내면 결과 tensor가 text 형식으로 온다.

우리는 그걸 json으로 바꾸고 안에 있는 list를 꺼내서 inference 결과를 받으면 되는 것이다.

다음 포스트에서는 이걸 이용해서 간단한 application을 flask로 만들어보는 시간을 가져보자.