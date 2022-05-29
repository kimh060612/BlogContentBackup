---
title: Vision Transformer 강의 내용 part.B
date: 2022-03-05 19:28:46
categories: 
    - Deep Learning
tags: 
    - Deep Learning
    - Tensorflow2
mathjax: true
---

*본 포스트는 [블로그](https://medium.com/towards-data-science/is-this-the-end-for-convolutional-neural-networks-6f944dccc2e9)와논문 "An Image is Worth 16x16 Words: Transformers for iamge recognition at scale"을 참조하여 작성되었음을 알립니다.*

### Index  
  
1. ~~CNN의 종말~~
2. ~~Vision Transformer의 구조~~
3. ~~ViT Inspection~~
   * ~~Embedding Filters~~
   * ~~Positional Encoding~~
   * ~~Attention Distance~~
4. practice


<br>

### practice

-------------------------------------------------------

이제 필자의 Deep learning 포스트도 꽤나 현대적인 수준의 모델을 다루게 되었다. 그리고 이번에는 2019년도에 나온 ViT의 구현이다. ViT의 구현은 생각보다 어렵지 않다. 왜냐면 Transformer의 연장선이기 때문이다. 따라서 기존 Transformer의 부분인 Multi-Head Attention과 Encoder 부분이 아닌 다른 부분만 이 포스트에서 다루도록 하겠다.

먼저, ViT 모델부터 보고 시작하자.

*file: model/model.py*

```python
class ViT(keras.Model):
    def __init__(self, patch_size, out_dim, mlp_dim, 
                 num_layer, d_model, num_haed, d_ff, drop_out_prob = 0.2):
        super(ViT, self).__init__()
        
        self.d_model = d_model
        self.patch_size = patch_size
        
        self.patch_gen = GeneratePatch(patch_size=patch_size)
        self.linear_proj = keras.layers.Dense(d_model, activation="relu")
        
        self.encoders = [ Encoder(d_model=d_model, num_head=num_haed, d_ff=d_ff, drop_out_prob=drop_out_prob, name = "Encoder_" + str(i + 1)) for i in range(num_layer) ]
        
        self.dense_mlp = keras.layers.Dense(mlp_dim, activation="relu")
        self.dropout = keras.layers.Dropout(drop_out_prob)
        self.dense_out = keras.layers.Dense(out_dim, activation="softmax")
    
    def build(self, x_shape):
        num_patch = (x_shape[1] * x_shape[2]) // (self.patch_size * self.patch_size) 
        
        self.pos_emb = self.add_weight("pos_emb", shape=(1, num_patch + 1, self.d_model))
        self.class_emb = self.add_weight("class_emb", shape=(1, 1, self.d_model)) 

    def call(self, x, training):
        
        patches = self.patch_gen(x)
        x = self.linear_proj(patches)
        
        batch_size = tf.shape(x)[0]
        class_emb = tf.broadcast_to(self.class_emb, [batch_size, 1, self.d_model])    
        x = tf.concat([class_emb, x], axis=1)
        
        x = x + self.pos_emb
        
        for layer in self.encoders:
            x = layer(x, training)
        
        x = self.dense_mlp(x[:, 0, :])
        x = self.dropout(x)
        x = self.dense_out(x)
        
        return x
```

먼저, 시작 전에 이번에는 다른 모델들과는 사뭇 다르게 build 함수를 overriding해서 사용하였다. 그 이유는 뭐냐? 이전에 ViT의 정의를 보면 명확해 진다. 

1. $x_{class}$는 BERT에서 사용되는 것과 똑같은 token이다. 이의 구현 방법은 part.B에서 더욱 자세히 다루어 보도록 하겠다.
2. 각 Image Patch들은 1 개의 FCNN layer를 거치게 된다. 이때, Bias는 없다.
3. 위 표기 중에 $LN$은 Layer Normalization의 약자이다. 
4. Encoder를 $L$번 반복하고 나온 결과를 MLP에 넣는다.
5. 최종 결과에 다시 한번 Layer Normalization을 진행한다. 이때, sequence의 마지막 Component만을 가져와서 진행한다.

여기서 논문을 보면 learnable class token $x_{class}$라고 나온다. 한마디로 맨 앞에 꼽사리 끼는 $x_{class}$ token 또한 학습이 가능한 weight여야 한다는 것이다. 그리고 여기서는 positional encoding 또한 학습이 가능한 weight 여야 한다. 그래서 입력 dimension을 사전에 받아서 모델을 빌드할때 이들을 정의해 줘야한다.

그리고 Image를 Flatten하고 Dense에 넣어줘야 하는데, 이 과정은 다른 layer를 통해서 진행할 것이다. 다음을 보자.

*file: model/layer.py*

```python
class GeneratePatch(keras.layers.Layer):
    def __init__(self, patch_size):
        super(GeneratePatch, self).__init__()
        self.patch_size = patch_size
    
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(images=images, 
                                        sizes=[1, self.patch_size, self.patch_size, 1], 
                                        strides=[1, self.patch_size, self.patch_size, 1], rates=[1, 1, 1, 1], padding="VALID")
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims]) #here shape is (batch_size, num_patches, patch_h*patch_w*c) 
        return patches
```

이 layer를 이용해서 image를 patch로 나눈 뒤에 flatten하고 ViT에서 Dense를 거쳐서 Transformer의 입력으로써 사용한다. 
나머지는 Transformer와 같다. 딱히 다를 것이 없다.

