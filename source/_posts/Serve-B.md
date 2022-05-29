---
title: Deep learning Model Serving 강의 part.B
date: 2022-03-05 19:29:16
categories: 
    - Deep Learning
tags: 
    - Deep Learning
    - Tensorflow2
mathjax: true
---

### Index  
  
1. ~~Tensorflow Serving Server~~
2. ~~Tensorflow Serving Client~~
3. Tensorflow Serving Application

<br>

### Tensorflow Serving Application

-------------------------------------------------------

이번 파트에서는 flask를 이용해서 간단한 웹 페이지를 만들어서 배포해보고 tensorflow serving과 통신하는 것까지 구현해 보도록 하겠다.

일단 간단하게 HTML 페이지를 작성해 보도록 하겠다.

```html
<body>
    <p>Upload Your Image</p>
    
    <form id="upload" action="/predict" method="POST" enctype="multipart/form-data">
        <button class="btn">Upload</button>
        <input type="file" value="Upload" name="image">

        <br>

        <p>Result</p>
        {% if label %}
            <span class="result">
                {{ label }}
            </span>
        {% endif %}
    </form>

</body>
```

귀찮아서 body만 가져왔다.  
대충 이렇게 페이지를 짜주자. 왜냐면 메인은 이게 아니니깐.

그리고 다음과 같이 flask server를 구성해 보겠다.

```python
classes = ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "boat", "truck"]
headers = {"content-type": "application/json"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    data = request.files["image"].read()
    image = Image.open(io.BytesIO(data))
    
    image = image.convert("RGB")
    image = image.resize((32, 32))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = np.asarray(image, dtype=np.uint8)
    
    data = json.dumps({"signature_name": "serving_default", "instances": image.tolist()})
    json_response = requests.post('http://localhost:8501/v1/models/ViT:predict', data=data, headers=headers)
    print(json_response.text)
    predictions = np.array(json.loads(json_response.text)["predictions"])
    index = np.argmax(predictions[0])
    
    return render_template("index.html", label=classes[index])
```

굉장히 심플하게 구성했다. 이제 서버를 구동해 주면 이미지를 띄울 수 있는 창이 뜨고 사진을 업로드 하면 inference 결과를 보여준다.

<p align="center"><img src="https://kimh060612.github.io/img/page1.png" width="100%"></p>