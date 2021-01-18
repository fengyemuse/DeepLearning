from tensorflow.keras.models import model_from_json
from sklearn import datasets
from tensorflow.keras.utils import to_categorical

dataset = datasets.load_iris()
x = dataset.data
y = dataset.target
Y_labels = to_categorical(y, num_classes=3)  # one hot编码

with open('model.json','r') as file:
    model_json = file.read()

new_model = model_from_json(model_json)
new_model.load_weights('model.json.h5')
new_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
scores = new_model.evaluate(x, Y_labels, verbose=0)
print('%s: %f%%' % (new_model.metrics_names[1], scores[1] * 100))