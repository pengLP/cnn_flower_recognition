from skimage import io,transform
import tensorflow as tf
import numpy as np


path1 = "D:/python/workspace/flower/data/flowers/dandelion/8223968_6b51555d2f_n.jpg"
path2 = "D:/python/workspace/flower/data/other/1582514704.jpg"


flower_dict = {0:'flower',1:'other'}

w=100
h=100
c=3

def read_one_image(path):
    img = io.imread(path)
    img = transform.resize(img,(w,h))
    return np.asarray(img)

with tf.Session() as sess:
    data = []
    data1 = read_one_image(path1)
    data2 = read_one_image(path2)
    data.append(data1)
    data.append(data2)
    print(data1.shape)
    saver = tf.train.import_meta_graph('D:/python/workspace/flower/model1/model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('D:/python/workspace/flower/model1/'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x:data}

    logits = graph.get_tensor_by_name("logits_eval:0")

    classification_result = sess.run(logits,feed_dict)

    #打印出预测矩阵
    print(classification_result)
    #打印出预测矩阵每一行最大值的索引
    print(tf.argmax(classification_result,1).eval())
    #根据索引通过字典对应花的分类
    output = []
    output = tf.argmax(classification_result,1).eval()
    for i in range(len(output)):
        print("第",i+1,"朵花预测:"+flower_dict[output[i]])
