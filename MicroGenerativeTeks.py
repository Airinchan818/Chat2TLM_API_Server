import tensorflow as tf
from tensorflow import keras

@keras.utils.register_keras_serializable()
class BlockDecoders (keras.Layer) :
  def __init__ (self,d_model,ffn_dim,num_heads,dropout_rate=0.1,**kwargs) :
    super(BlockDecoders,self).__init__(**kwargs)
    self.mha = keras.layers.MultiHeadAttention(num_heads=num_heads,key_dim=d_model,dropout=dropout_rate)
    self.normal1 = keras.layers.LayerNormalization(epsilon=1e-6)
    self.ffn = keras.Sequential([
        keras.layers.Dense(ffn_dim,activation=keras.activations.gelu),
        keras.layers.Dense(d_model)
    ])
    self.dropout = keras.layers.Dropout(rate=dropout_rate)
    self.normal2 = keras.layers.LayerNormalization(epsilon=1e-6)
    self.d_model = d_model
    self.ffn_dim = ffn_dim
    self.num_head = num_heads
    self.dropout_rate = dropout_rate

  def call(self,x,training=False) :
    attn = self.mha(x,x,x,training=training,use_causal_mask=True)
    attn = self.normal1(attn + x)

    ffn = self.ffn(attn)
    ffn = self.dropout(ffn)
    ffn = self.normal2(ffn + attn)
    return ffn

  def get_config(self) :
    config = super(BlockDecoders,self).get_config()
    config.update ({
        'd_model' : self.d_model,
        'ffn_dim' : self.ffn_dim,
        'num_head' : self.num_head,
        'dropout_rate' : self.dropout_rate
    })
    return config

  @classmethod
  def from_config(cls,config) :
    return cls(**config)

@keras.utils.register_keras_serializable()
class Micro_Gen_Teks (keras.Model) :
  def __init__ (self,vocab_size,d_model,ffn_dim,num_heads,num_blocks,maxpos,dropout_rate=0.1,**kwargs) :
    super(Micro_Gen_Teks,self).__init__(**kwargs)
    self.Embedding = keras.layers.Embedding(vocab_size,d_model)
    self.pos_embedding = keras.layers.Embedding(maxpos,d_model)
    self.BlockDecoders = [BlockDecoders(
        d_model=d_model,ffn_dim=ffn_dim,num_heads=num_heads,dropout_rate=dropout_rate
    ) for _ in range(num_blocks)]
    self.final_layer = keras.layers.Dense(vocab_size)

    self.vocab_size = vocab_size
    self.d_model = d_model
    self.ffn_dim = ffn_dim
    self.num_heads = num_heads
    self.num_blocks = num_blocks
    self.dropout_rate = dropout_rate
    self.maxpos = maxpos

  def call(self,x,training = True) :
    batch, seq = tf.shape(x)[0], tf.shape(x)[1]
    pos = tf.range(start=0,limit=seq,delta=1)
    pos = self.pos_embedding(pos)
    pos = tf.expand_dims(pos,axis=0)
    x = self.Embedding(x)
    x *= tf.sqrt(tf.cast(self.d_model,dtype=tf.float16))
    x = x + pos
    for block in self.BlockDecoders :
      x = block(x,training=training)

    x = self.final_layer(x)
    return x


  def get_config(self) :
    config = super(Micro_Gen_Teks,self).get_config()
    config.update({
        'vocab_size' : self.vocab_size,
        'd_model' : self.d_model,
        'ffn_dim' : self.ffn_dim,
        'num_heads' : self.num_heads,
        'num_blocks' : self.num_blocks,
        'dropout_rate' : self.dropout_rate,
        'maxpos' : self.maxpos
    })
    return config

  @classmethod
  def from_config(cls,config) :
    return cls(**config)