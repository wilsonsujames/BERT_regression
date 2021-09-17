import os
import shutil
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import matplotlib.pyplot as plt
from tensorflow.keras import losses


# reloaded_model = tf.saved_model.load('reg_Bert')
reloaded_model = tf.keras.models.load_model('reg_Bert')

examples = [
    'this is such an amazing movie!',
    'The movie was great!',
    'The movie was meh.',
    'The movie was okish.',
    'The movie was terrible...',
    'hello hello hello'
]


result = reloaded_model.predict([examples[0]])
print(result)

result = reloaded_model.predict([examples[1]])
print(result)
result = reloaded_model.predict([examples[4]])
print(result)

# result = reloaded_model(tf.constant(examples))
# print(result)



