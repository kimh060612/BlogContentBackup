---
title: Transformer 강의 내용 part.C
date: 2022-03-05 19:28:31
categories: 
    - Deep Learning
tags: 
    - Deep Learning
    - Tensorflow2
mathjax: true
---

*본 포스트는 해당 링크의 [블로그](https://wikidocs.net/22893)와 논문 "Attention is all you need"를 참조하여 작성되었음을 알립니다.*

### Index  
  
1. ~~Attention!~~
    * ~~Seq2Seq~~
    * ~~Attention in Seq2Seq~~
2. ~~Transformer Encoder~~
    * ~~Attention in Transformer~~
    * ~~Multi-Head Attention~~
    * ~~Masking~~
    * ~~Feed Forward~~
3. ~~Transformer Decoder~~
4. ~~Positional Encoding~~
5. Partice
    * Seq2Seq Attention
    * Transformer

<br>

### Partice

-------------------------------------------------------

이 파트에서는 지난 시간부터 쭉 다뤄온 Attention과 Transformer를 구현해 보는 시간을 가질 것이다. 본격적으로 Model Subclassing API를 제대로 활용하는 시간이 될 것이다.

#### Seq2Seq Attention

Transformer 구조를 보기 전에 Attention을 Seq2Seq 모델에 적용시켜 보도록 하겠다. 먼저 소스코드부터 보고 가자.

```python
class Attention(keras.layers.Layer):
    def __init__(self):
        super(Attention, self).__init__()
        
    def call(self, q, k, v, mask=None):
        # q: (batch_size, seq_len, d_model)
        # k: (batch_size, seq_len, d_model)
        # v: (batch_size, seq_len, d_model)

        matmul_qk = tf.matmul(q, k, transpose_b=True)
        # qk^T : (batch_size, seq_len, seq_len)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        # output: (batch_size, seq_len, d_model)
        return output, attention_weights
```

우선 우리가 사용할 Attention 구조이다. 지난 Transformer part.A에서 정의한 연산(masking 포함)을 고대로 구현한 것이다.
이를 어떻게 Seq2Seq 모델에 적용하는지는 밑의 그림과 소스코드를 보자.

<p align="center"><img src="https://kimh060612.github.io/img/Attention.png" width="100%"></p>

위 그림에 따르면 일단 Seq2Seq는 Encoder와 Decoder가 각기 다른 모델로 있다. 그리고 Encoder에서 나온 모든 time step의 출력과 Decoder의 출력을 같이 사용해서 연산을 해야할 필요가 있다.

필자는 일단 Decoder에 Encoder의 모든 Time step의 출력을 넣는 형식으로 구현을 하였다.

```python
class Encoder(keras.Model):
    def __init__(self, WORDS_NUM, emb_dim, hidden_unit, batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.hidden_unit = hidden_unit
        self.embedding = keras.layers.Embedding(WORDS_NUM, emb_dim)
        self.LSTM = keras.layers.LSTM(hidden_unit, return_state=True, return_sequences=True)

    def call(self, inputs, enc_hidden = None):
        x = self.embedding(inputs)
        y, h, _ = self.LSTM(x, initial_state=enc_hidden)
        return y, h

class Decoder(keras.Model):
    def __init__(self, WORDS_NUM, emb_dim, hidden_unit, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = keras.layers.Embedding(WORDS_NUM, emb_dim)
        self.lstm = tf.keras.layers.LSTM(hidden_unit, return_sequences=True, return_state=True)
        self.attention = Attention()
        self.dense = tf.keras.layers.Dense(WORDS_NUM, activation='softmax')

    def call(self, x, hidden, mask=None):
        x = self.embedding(x)
        _, h, c = self.lstm(x)
        context_vec, attention_weight = self.attention(h, hidden, hidden, mask=mask)
        
        x_ = tf.concat([context_vec, c], axis=-1)
        out = self.dense(x_)
        return out, h, attention_weight
```

소스코드를 보기 전에 Attention의 $Q$, $K$, $V$의 정의를 먼저 상기하고 가자.

$Q$ : 특정 시점의 디코더 셀에서의 은닉 상태  
$K$ : 모든 시점의 인코더 셀의 은닉 상태들  
$V$ : 모든 시점의 인코더 셀의 은닉 상태들  

Encoder에서는 Decoder에서 활용하기 위해서 2개의 출력을 내놓는다. 본인의 출력값과 hidden state의 값이다.

Decoder에서는 Encoder에서의 Hidden state를 입력으로 받아서 자신의 hidden state와 같이 attention! 을 진행한다.

이렇게 만들어진 vector와 마지막 출력 값을 concatenation한 값을 dense에 넣어서 최종적인 출력을 내 놓는다.

이의 training loop를 보면 더 이해가 잘 될 것이다. 

```python
@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0
    
    with tf.GradientTape() as tape:
        _, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)
        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden)
            loss += loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)
        
        batch_loss = (loss / int(targ.shape[1]))
        
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss
```

#### Transformer

이제 대망의 Transformer이다. 먼저 Transformer를 구현하기 위해서 어떤 Component들이 필요했는지 알아보자.

1. Multi-head Attention
2. Positional Encoding
3. Encoder Blocks
4. Decoder Blocks

우리는 이 4가지 Component들을 각각 순차적으로 구현하고 이를 통해서 최종 모델을 구현할 것이다.

*file: model/layer.py*

```python
class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model, num_head):
        super(MultiHeadAttention, self).__init__()        
        if not d_model % num_head  == 0:
            raise ValueError("Invalid head and d_model!")
        self.d_k = d_model // num_head
        self.num_head = num_head
        self.d_model = d_model
        
        self.Wq = keras.layers.Dense(self.d_model)
        self.Wk = keras.layers.Dense(self.d_model)
        self.Wv = keras.layers.Dense(self.d_model)
        
        self.dense = keras.layers.Dense(self.d_model, activation="relu")

    def split_head(self, batch_size, x):
        x = tf.reshape(x, (batch_size, -1, self.num_head, self.d_k))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask = None):
        batch_size = q.shape[0]
        
        q, k, v = self.Wq(q), self.Wk(k), self.Wv(v)
        # q: (batch_size, seq_len, d_model)
        # k: (batch_size, seq_len, d_model)
        # v: (batch_size, seq_len, d_model)
        
        q, k, v = self.split_head(batch_size, q), self.split_head(batch_size, k), self.split_head(batch_size, v)
        # q: (batch_size, num_head, seq_len, d_k)
        # k: (batch_size, num_head, seq_len, d_k)
        # v: (batch_size, num_head, seq_len, d_v)
        
        qkT = tf.matmul(q, k, transpose_b=True) # (batch_size, num_head, seq_len, seq_len)
        d_k = tf.cast(self.d_k, dtype=tf.float32)
        scaled_qkT = qkT / tf.math.sqrt(d_k)
        if not mask == None:
            scaled_qkT += (mask * -1e9)
        
        attention_dist = tf.nn.softmax(scaled_qkT, axis=-1)
        attention = tf.matmul(attention_dist, v) # (batch_size, num_head, seq_len, d_k)
        
        attention = tf.transpose(attention, perm=[0, 2, 1, 3]) # (batch_size, seq_len, num_head, d_k)
        concat_attention = tf.reshape(attention, (batch_size, -1, self.d_model)) # (batch_size, seq_len, d_model)
        output = self.dense(concat_attention)
        
        return output
```

각 연산 뒤에 차원을 적어 두었으니 이해하기는 어렵지 않을 것이다. 구체적인 정의가 떠오르지 않을 독자들을 위해서 Multi-Head Attention의 수식을 적어두고 가겠다.

$$
\text{Attention}(Q, K, V) = \text{Softmax}(\frac{Q_iK_i^T}{\sqrt{d_k}})V_i
\tag{definition 3}
$$

$$
M_{out} = \text{Concat}(O_1, O_2, ..., O_{H_n}) W^O
\tag{equation 1}
$$


위 수식을 고대~로 구현한 것 밖에 되지 않는다.

그리고 positional Encoding은 다음과 같이 구현했다.

*file: model/layer.py*

```python
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates
    
def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)
```

이것 또한 정의를 고대로 구현한 것이다. ~~(참 쉽죠?)~~

$$
\begin{aligned}
    PE_{(pos, 2i)} = \sin(pos/10000^{2i/d_{model}}) \\
    PE_{(pos, 2i + 1)} = \cos(pos/10000^{2i/d_{model}})
\end{aligned}
$$

이제 Encoder와 Decoder Block을 구현할 차례이다. 귀찮으니 한번에 소스를 적어 두도록 하겠다.

*file: model/layer.py*

```python
class Encoder(keras.layers.Layer):
    def __init__(self, d_model, num_head, d_ff, drop_out_prob = 0.2):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_head = num_head
        
        self.MultiHeadAttention = MultiHeadAttention(d_model=d_model, num_head=num_head)
        
        self.dense_1 = keras.layers.Dense(d_ff, activation="relu")
        self.dense_2 = keras.layers.Dense(d_model, activation="relu")
        
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(drop_out_prob)
        self.dropout2 = keras.layers.Dropout(drop_out_prob)

    def call(self, x, training=True, mask=None):
        
        out_atten = self.MultiHeadAttention(x, x, x, mask)
        out_atten = self.dropout1(out_atten, training=training)
        x = self.layernorm1(out_atten + x)
        
        out_dense = self.dense_1(x)
        out_dense = self.dense_2(out_dense)
        out_dense = self.dropout2(out_dense, training=training)
        x = self.layernorm2(out_dense + x)
        
        return x
        
    
class Decoder(keras.layers.Layer):
    def __init__(self, d_model, num_head, d_ff, drop_out_prob = 0.2):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.MultiHeadAttention1 = MultiHeadAttention(d_model, num_head)
        self.MultiHeadAttention2 = MultiHeadAttention(d_model, num_head)
        
        self.dense_1 = keras.layers.Dense(d_ff, activation="relu")
        self.dense_2 = keras.layers.Dense(d_model, activation="relu")
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_out_prob)
        self.dropout2 = tf.keras.layers.Dropout(drop_out_prob)
        self.dropout3 = tf.keras.layers.Dropout(drop_out_prob)
        
        pass

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        
        out_atten1 = self.MultiHeadAttention1(x, x, x, look_ahead_mask)
        out_atten1 = self.dropout1(out_atten1, training=training)
        x = self.layernorm1(out_atten1 + x)
        
        out_atten2 = self.MultiHeadAttention2(enc_output, enc_output, x, padding_mask)
        out_atten2 = self.dropout2(out_atten2, training=training)
        x = self.layernorm2(out_atten2 + x)
        
        out_dense = self.dense_1(x)
        out_dense = self.dense_2(out_dense)
        out_dense = self.dropout3(out_dense, training=training)
        x = self.layernorm3(out_dense + x)
        
        return x
```

Encoder는 별로 볼게 없고, Decoder를 보자면 지난번 정의에서 다뤘듯이, 첫번째 Multi-Head Attention과 두번째 Mutli-Head Attention은 다른 입력이 주어지고, 다른 masking이 주어진다.

따라서 Decoder의 입력으로 출력 sequence, encoder 출력, look ahead masking, padding masking이 들어가야 한다.

자세한 정의는 part.B를 다시 참고하자.

참고로, masking은 다음 함수로 만들었다.

*file: model/model.py*

```python
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)
```

그리고 최종적으로 이렇게 만든 Encoder, Decoder Block을 쌓아서 Transformer 모델을 만들어야 한다.

편의상, Encoder모델, Decoder 모델을 만들고 이를 하나로 합쳐서 Transformer 모델을 만들었다.

*file: model/model.py*

```python
class EncoderModel(keras.Model):
    def __init__(self, input_voc_size, num_layers, max_seq_len, d_model, num_head, d_ff, drop_out_prob = 0.2):
        super(EncoderModel, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_voc_size, d_model)
        self.pos_encoding = positional_encoding(max_seq_len, self.d_model)
        self.enc_layers = [ Encoder(d_model, num_head, d_ff, drop_out_prob) for _ in range(num_layers) ]
        self.dropout = tf.keras.layers.Dropout(drop_out_prob)
        
    def call(self, x, training, mask):
        
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
            
        return x
        
class DecoderModel(keras.Model):
    def __init__(self, output_voc_size, num_layers, max_seq_len, d_model, num_head, d_ff, drop_out_prob = 0.2):
        super(DecoderModel, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(output_voc_size, d_model)
        self.pos_encoding = positional_encoding(max_seq_len, self.d_model)
        
        self.dec_layers = [ Decoder(d_model, num_head, d_ff, drop_out_prob) for _ in range(num_layers) ]
        self.dropout = tf.keras.layers.Dropout(drop_out_prob)
        
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            
        return x
    
class Transformer(keras.Model):
    def __init__(self, input_voc_size, output_voc_size, num_layers, max_seq_len_in, max_seq_len_out, d_model, num_head, d_ff, drop_out_prob = 0.2):
        super().__init__()
        self.Encoder = EncoderModel(input_voc_size, num_layers, max_seq_len_in, d_model, num_head, d_ff, drop_out_prob)
        self.Decoder = DecoderModel(output_voc_size, num_layers, max_seq_len_out, d_model, num_head, d_ff, drop_out_prob)
        
        self.final_layer = tf.keras.layers.Dense(output_voc_size)
        
    def call(self, inputs, training):
        inp, tar = inputs
        
        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)
        enc_output = self.Encoder(inp, training, enc_padding_mask)
        dec_output = self.Decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        
        final_output = self.final_layer(dec_output)
        return final_output
        
    def create_masks(self, inp, tar):
        enc_padding_mask = create_padding_mask(inp)
        dec_padding_mask = create_padding_mask(inp)

        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, look_ahead_mask, dec_padding_mask
```

그리고 Transformer에서 보면 Embedding layer를 사용하는데, 이는 Dense와 결과적인 측면에서는 다를 것이 없디. 하지만 token을 vector로 Embedding하는데 있어서 훨씬 효율적이니 되도록 token을 vector로 임베딩할때 이걸 사용하도록 하자.

각 Encoder와 Decoder는 num_layer(hyper parameter)만큼 Encoder, Decoder 블록을 반복해서 쌓고, 그것들을 차례로 출력으로 내놓는다. 이제 2개를 합쳐서 간단하게 Transformer 모델이 완성되는 것이다.

이쯤되면 필자들도 느낄 것이다. 이럴때 Model Subclassing API가 빛을 발휘한다. 이 각각의 Component들을 함수로만 구현하는 것 보다는 class로 묶어서 관리하는 것이 가독성, 유지보수 측면에서 훨씬 이득이다.

그리고 논문에 의하면, Transformer는 독자적인 learning rate scheduler를 가진다. 이 수식은 다음과 같다.

<p align="center"><img src="https://kimh060612.github.io/img/lr_schedule.png" width="50%"></p>

이를 구현하기 위해서 다음과 같이 Custom Scheduler를 만들 수 있다.

*file: model/model.py*

```python
class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model_save = d_model
        self.d_model = tf.cast(d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
    def get_config(self):
        config = {
            'd_model': self.d_model_save,
            'warmup_steps': self.warmup_steps,
        }
        return config
```

여기까지 구현하면 Transformer의 구현은 끝이 난다. 굳이 여기서는 Training loop까지는 다루지 않겠다. 한번 필자가 직접 완성해 보도록 하자. 정 모르겠다면 필자의 github에 완성판이 있으니 직접 가서 찾아 보기를 바란다.
