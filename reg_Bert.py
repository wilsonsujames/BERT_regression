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
    label = int(file_name.split(".")[0].split("_")[1]) / 10
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
    label = int(file_name.split(".")[0].split("_")[1]) / 10
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

# bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)

text_test = ['this is such an amazing movie!']
# text_preprocessed = bert_preprocess_model(text_test)

# print(f'Keys       : {list(text_preprocessed.keys())}')
# print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
# print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
# print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
# print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')


# bert_model = hub.KerasLayer(tfhub_handle_encoder)

# bert_results = bert_model(text_preprocessed)

# print(f'Loaded BERT: {tfhub_handle_encoder}')
# print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
# print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
# print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
# print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')


def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(1 )(net)
  return tf.keras.Model(text_input, net)

classifier_model = build_classifier_model()
bert_raw_result = classifier_model(tf.constant(text_test))
print(bert_raw_result)


loss='mean_absolute_error'
optimizer=tf.optimizers.Adam(learning_rate=0.1)


classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                        )


history = classifier_model.fit(x=train_ds,
                               validation_data=val_ds,
                               epochs=1)
classifier_model.save('reg_Bert2')

# history_dict = history.history
# print(history_dict.keys())

# loss = history_dict['loss']
# val_loss = history_dict['val_loss']

# fig = plt.figure(figsize=(10, 6))
# fig.tight_layout()
# # "bo" is for "blue dot"
# plt.plot([1,2,3],loss, 'r', label='Training loss')
# plt.plot( [1,2,3],val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# # plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.show()



examples = [
    'this is such an amazing movie!',
    'The movie was great!',
    'The movie was meh.',
    'The movie was okish.',
    'The movie was terrible...'
]


result = classifier_model.predict(examples)
print(result)
