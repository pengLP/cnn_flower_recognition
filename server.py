from flask import Flask, render_template, request
import json
from skimage import io,transform
import tensorflow as tf
import numpy as np
app = Flask(__name__)

modle_path = "D:/python/workspace/flower/model/model.ckpt.meta"
flower_dict = {0:'dasiy',1:'dandelion',2:'roses',3:'sunflowers',4:'tulips'}
flower_dict1 = {0:'flower',1:'other'}
@app.route('/')
def hello_world():
    return render_template("phone.html", data=None)



def getType(path):
    w = 100
    h = 100
    c = 3
    img = io.imread(path)
    data = []
    data.append(transform.resize(img,(w,h,3)))
    with tf.Session() as sess:

        saver = tf.train.import_meta_graph('D:/python/workspace/flower/model1/model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('D:/python/workspace/flower/model1/'))

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        feed_dict = {x: data}

        logits = graph.get_tensor_by_name("logits_eval:0")

        classification_result = sess.run(logits, feed_dict)
        # 打印出预测矩阵每一行最大值的索引
        print(classification_result)

        output = tf.argmax(classification_result, 1).eval()
        if (output[0] == 1):
            return "不是花"
    tf.reset_default_graph()

    with tf.Session() as sess1:
        saver1 = tf.train.import_meta_graph('D:/python/workspace/flower/model/model.ckpt.meta')
        saver1.restore(sess1, tf.train.latest_checkpoint('D:/python/workspace/flower/model/'))
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        feed_dict = {x: data}

        logits = graph.get_tensor_by_name("logits_eval:0")

        classification_result = sess1.run(logits, feed_dict)
        # 打印出预测矩阵每一行最大值的索引

        output = tf.argmax(classification_result, 1).eval()
    return flower_dict[output[0]]

@app.route('/upload',methods=['POST'])
def upload():
    file = request.files.get('file')

    type = getType(file)

    res = file.filename +    "，类型是：" + type
    return json.dumps(res, ensure_ascii=False)

@app.route('/phone',methods=['POST'])
def phone():
    file = request.get_data()

    res = file.filename +    "，类型是：" + type
    return json.dumps(res, ensure_ascii=False)

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080)