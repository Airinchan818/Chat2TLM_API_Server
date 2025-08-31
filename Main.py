from tensorflow import keras 
import tensorflow as tf 
from MicroGenerativeTeks import Micro_Gen_Teks
from fastapi import FastAPI
from tokenizers import Tokenizer
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


keras.mixed_precision.set_global_policy('mixed_float16')


tokenizer = Tokenizer.from_file("tokenizer.json")
model = keras.models.load_model("Pretrained.keras")


def sampling_top_k(logits, temperature=1.0, top_k=10):
    temperature = tf.cast(temperature, dtype=tf.float16)
    logits = logits / temperature
    if len(logits.shape) > 2:
        logits = logits[:, -1, :]
    if top_k > 0: 
        values, _ = tf.math.top_k(logits, k=top_k)
        min_values = tf.expand_dims(values[:, -1], axis=-1)
        logits = tf.where(logits < min_values, 1e-9, logits)
    return tf.random.categorical(logits, num_samples=1)


def generate_teks(seq, max_len=150):
    seq = tokenizer.encode(seq).ids
    sos = tokenizer.encode("s/").ids
    token = sos + seq
    start_idx = len(token)
    token = keras.preprocessing.sequence.pad_sequences([token], padding="post", maxlen=300)
    token = np.array(token, dtype=np.int16)
    values = []
    zeros_toleran = 0

    for i in range(start_idx, max_len):
        if zeros_toleran >= 5:
            break
        outputs = model(token, training=False)
        logits = tf.nn.softmax(outputs, axis=-1)
        result = sampling_top_k(logits, top_k=70)
        t = result.numpy()[0][0]
        values.append(t)
        if t == 0:
            zeros_toleran += 1
        token[0, i] = t 
        print(f"Token Status {t} : Token position {i}")
    return tokenizer.decode(values)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class PredictInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(data: PredictInput):
    seq = data.text
    print(f"input request {seq} status : Model Running ")
    sentence = generate_teks(seq,max_len=100)
    return {"response": sentence}

@app.post("/test")
def test():
    return {"Message": "API HAS CONNECTED"}
