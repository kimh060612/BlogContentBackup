---
title: Optimization 강의 내용 part.B
date: 2022-03-05 19:27:39
categories: 
    - Deep Learning
tags: 
    - Deep Learning
    - Tensorflow2
mathjax: true
---

*본 포스트는 Hands-on Machine learning 2nd Edition, CS231n, Tensorflow 공식 document, Pattern Recognition and Machine Learning, Deep learning(Ian Goodfellow 저)을 참조하여 작성되었음을 알립니다.*

### Index  
  
1. ~~Essential Mathematics~~  
   * ~~Basic of Bayesian Statistics~~
   * ~~Information Theory~~
   * ~~Gradient~~
2. ~~Loss Function Examples~~
3. ~~What is Optimizer?~~
4. ~~Optimizer examples~~
5. Partice

<br>

### Partice

-------------------------------------------------------

자, 이제 구현의 난이도 측면에서는 가장 어렵다고 말할 수 있는 파트가 왔다. 이번에도 Model Subclassing API를 활용하여 Custom Optimizer를 만들어 보도록 하겠다. Custom이라고 해서 뭔가 새로운건 아니고, 간단하게 SGD를 구현해 볼 것이다. 근데 솔직히 이건 별로 쓸데가 없다. 이것까지 건드려야하는 사람은 아마도 직접 짜는게 빠르지 않을까 싶다.
여기서 가장 중요한 것은 Custom Training loop와 Custom loss이다. SGD는 간단하게 소스코드만 보고 Training loop와 Custom loss를 자세히 설명하도록 하겠다.

```python
class CustomSGDOptimizer(keras.optimizers.Optimizer):
    def __init__(self, learning_rate = 0.001, name = "CustomSGDOptimizer", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._is_first = True

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "pv") # previous variable
        for vat in var_list:
            self.add_slot(var, "pg") # previous gradient
    
    @tf.function
    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)

        new_var_m = var - lr_t * grad

        pv_var = self.get_slot(var, "pv")
        pg_var = self.get_slot(var, "pg")

        if self._is_first :
            self._is_first = False
            new_var = new_var_m
        else:
            cond = grad * pg_var >= 0
            avg_weight = (pv_var + var) / 2.0
            new_var = tf.where(cond, new_var_m, avg_weight)
        
        pv_var.assign(var)
        pg_var.assign(grad)

        var.assign(new_var)

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError
    
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate" : self._serialize_hyperparameter("lr")
        }
```

> 그만 알아보자.

장난 안치고 이건 나중에 하나의 포스트를 다 써서 설명할 것이다. 그 정도로 다른 중요한 topic과 같이 다루기에는 무겁다.

일단, Custom Training loop를 어떻게 만드는지를 알아보도록 하겠다.

```python

train_dataset = tf.data.Dataset.from_tensor_slices((train_img, train_labels))
                            # 섞어.                        # 배치 사이즈 만큼 나눠.
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BatchSize)

# Optimizer & Loss Function 정의
optimizer = keras.optimizers.Adam(learning_rate=LR)
loss_function = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

train_accuracy = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

for epoch in range(EPOCHS):
    print("Epoch %d start"%epoch)
    # step, 1개의 batch ==> 의사 코드에서 batch 뽑는 역할
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        # **********************************************************************************************
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss_val = loss_function(y_batch, logits)
            # 여기서 신경망의 Feed Forward & Expectation of Loss 계산을 진행함.
        # tape.gradient를 호출하면 ==> gradient 계산이 진행됨.
        grad = tape.gradient(loss_val, model.trainable_weights)
        # 정의된 Optimizer를 이용해서 Update를 진행함.
        optimizer.apply_gradients(zip(grad, model.trainable_weights))

        # train에서의 정확도를 계산함.
        train_accuracy.update_state(y_batch, logits)
        if step % 500 == 0 :
            print("Training loss at step %d: %.4f"%(step, loss_val))
    
    # 정확도 뽑아 보겠다.
    train_acc = train_accuracy.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))
    train_accuracy.reset_states()

    for x_batch_val, y_batch_val in validation_dataset:
        val_logits = model(x_batch_val, training = False)
        val_acc_metric.update_state(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    print("Validation acc: %.4f" % (float(val_acc),))
    val_acc_metric.reset_states()

```

기본적으로 Custom Training Loop는 위와 같이 구현된다. 이걸 하나하나 뜯어서 설명해 보도록 하겠다.

1. tf.GradientTape()
   
    tensorflow 2의 핵심인 AutoGrad를 구동시켜주는 친구이다. 이 tape scope 안에서 실행된 tensorflow operation들은 back propagation을 위한 AutoGrad 미분 그래프의 구축이 시작된다.  

2. tape.gradient
   
    이 함수를 실행하면 구축된 AutoGrad 미분 그래프를 따라서 미분이 시작된다. 

3. apply_gradients

    이는 optimizer(ex. Adam)의 method이고 tape.gradient에서 구한 gradient를 사용자가 정한 optimizing algorithm을 통해서 Weight를 업데이트 해준다.

4. update_state
   
   이는 학습 과정중에서 측정할 수 있는 Metric들을 구하는데 사용된다. (ex. 정확도) keras에서 제공하거나 직접 만든 Metric 객체를 새로 Nerual Network에서 계산된 batch에 적용하고 싶을때 사용한다.

이렇게 Custom Training loop는 크게 4가지 요소로 구성된다.
필자는 이것을 보통 템플릿으로 가지고 개발할때마다 조금씩 바꿔서 사용하는 편이다. 독자들도 조금 복잡한 트릭이 필요한 Neural Network를 구현할때 본인만의 Training loop를 구성하고 조금씩 바꿔가면서 사용하는 것을 추천한다. 

다음으로 다룰 것은 Custom loss이다. 

```python
# Custom Loss
class CustomMSE(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, y_true, y_pred):
        # tf.math.square : 성분별 제곱
        # [1,2,3] ==> [1,4,9]
        L = tf.math.square(y_true - y_pred)
        # tf.math.reduce_sum: 벡터 합.
        # [1,2,3] ==> 6
        L = tf.math.reduce_mean(L)
        return L

# Custom Regularizer(규제 ==> MAP)
class CustomRegularizer(keras.regularizers.Regularizer):
    def __init__(self, _factor):
        super().__init__()
        self.factor = _factor
    
    def call(self, weights):
        return tf.math.reduce_sum(tf.math.abs(self.factor * weights))
    
    def get_config(self):
        return {"factor" : self.factor} # 모델을 저장할때 custom layer, loss, 등등을 저장할때 같이 저장해 주는 역할.
```

tensorflow는 사용자가 Loss나 Regularizer까지도 자유롭게 구성할 수 있도록 허락해준다. 사용 방법은 기존 loss들과 똑같다. 그저 자신이 원하는 연산들을 생각해서 구현하면 된다.

이걸로 Optimization part.B를 마치도록 하겠다. 이 정도까지 Custom해서 사용할 사람들은 많이 없겠지만 혹시라도 필요한 사람들이 있을까 싶어서 적어 보았다.