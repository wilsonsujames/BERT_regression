import os
import shutil
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import matplotlib.pyplot as plt
from tensorflow.keras import losses


AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32
seed = 42


for dirPath, dirNames, fileNames in os.walk("train"):
    pass

train_label =[]

for file_name in fileNames:
    # print(file_name.split(".")[0].split("_")[1])
    label = int(file_name.split(".")[0].split("_")[1])
    train_label.append(label)


raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    labels = train_label,
    seed=seed
    )

val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    labels = train_label,
    seed=seed)


for dirPath, dirNames, test_fileNames in os.walk("test"):
    pass

test_label =[]

for file_name in test_fileNames:
    # print(file_name.split(".")[0].split("_")[1])
    label = int(file_name.split(".")[0].split("_")[1])
    test_label.append(label)


test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'test',
     labels = test_label,
    batch_size=batch_size)


class_names = raw_train_ds.class_names
print(class_names)

train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# for text_batch, label_batch in train_ds.take(1):
#     for i in range(3):
#         print(f'Review: {text_batch.numpy()[i]}')
#         label = label_batch.numpy()
#         print(f'Label : {label} ')


tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'
tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'


def build_regression_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dense(units=1)(net)
    return tf.keras.Model(text_input, net)


regression_model = build_regression_model()

regression_model = tf.keras.Sequential([
    regression_model,
    tf.keras.layers.Dense(units=1)
])


loss='mean_absolute_error'
optimizer=tf.optimizers.Adam(learning_rate=0.1)


regression_model.compile(optimizer=optimizer,
                         loss=loss,
                        )


history = regression_model.fit(x=train_ds,
                               validation_data=val_ds,
                               epochs=5)
regression_model.save('reg_Bert')

history_dict = history.history
print(history_dict.keys())

loss = history_dict['loss']
val_loss = history_dict['val_loss']

fig = plt.figure(figsize=(10, 6))
fig.tight_layout()
# "bo" is for "blue dot"
plt.plot([1,2,3,4,5],loss, 'r', label='Training loss')
plt.plot( [1,2,3,4,5],val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
# plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()



examples = [
    'this is such an amazing movie!',
    'The movie was great!',
    'The movie was meh.',
    'The movie was okish.',
    'The movie was terrible...'
]


result = regression_model.predict(examples)
print(result)
