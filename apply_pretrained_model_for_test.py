

import tensorflow as tf

input_data=[[1,5,3,7,8,10,12],
           [3,2,4,5,7,9,10]]
label_data=[[0,0,0,1,0],
           [1,0,0,0,0]]




INPUT_SIZE =7
HIDDEN1_SIZE =10
HIDDEN2_SIZE =8
CLASSES =5

lr=0.05

x= tf.placeholder(tf.float32, shape=[None, INPUT_SIZE], name='x')
y_= tf.placeholder(tf.float32, shape=[None, CLASSES], name='y_')


tensor_map = {x:input_data, y_:label_data}



#with tf.device('/gpu:0'):
with tf.name_scope('test1') as test1:
    W_h1 =tf.Variable(tf.truncated_normal(shape=[INPUT_SIZE,HIDDEN1_SIZE]),dtype=tf.float32, name='W_h1')
    b_h1 =tf.Variable(tf.zeros(shape=[HIDDEN1_SIZE]), name='b_h1',dtype=tf.float32)

with tf.name_scope('test2') as test2:
    W_h2 =tf.Variable(tf.truncated_normal(shape=[HIDDEN1_SIZE,HIDDEN2_SIZE]),dtype=tf.float32,name = 'W_h2')
    b_h2 =tf.Variable(tf.zeros(shape=[HIDDEN2_SIZE]),dtype=tf.float32, name='b_h2')

with tf.name_scope('test4') as test4:
    W_o =tf.Variable(tf.truncated_normal(shape=[HIDDEN2_SIZE,CLASSES]),dtype=tf.float32)
    b_o =tf.Variable(tf.zeros(shape=[CLASSES]),dtype=tf.float32)

param_list =[W_h1, b_h1, W_h2, b_h2]


#param_list =[W_h1, b_h1, W_h2, b_h2, W_o, b_o]
saver = tf.train.Saver(param_list)



with tf.name_scope('hidden_layer_1') as h1scope:
    hidden1=tf.sigmoid(tf.matmul(x,W_h1) + b_h1,name ='hidden1')
with tf.name_scope('hidden_layer_2') as h2scope:
    hidden2=tf.sigmoid(tf.matmul(hidden1,W_h2) + b_h2, name= 'hidden2')
with tf.name_scope('output_layer') as oscope:
    y= tf.sigmoid(tf.matmul(hidden2,W_o) + b_o, name ='y')


sess =tf.Session()

sess.run(tf.global_variables_initializer())

saver.restore(sess,'./tensorflow_live.ckpt')


result = sess.run(y,tensor_map)
print result

sess.close()
