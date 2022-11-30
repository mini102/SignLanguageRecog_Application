import tensorflow as tf
import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load('best.onnx')
tf_rep = prepare(onnx_model)
tf_rep.export_graph('./content/model.pb')

converter = tf.lite.TFLiteConverter.from_saved_model('./content/model.pb')
tflite_model = converter.convert()

with open('./content/model.tflite', 'wb') as f:
  f.write(tflite_model)