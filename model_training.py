

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

with tf.name_scope('test3') as test3:   
    W_o =tf.Variable(tf.truncated_normal(shape=[HIDDEN2_SIZE,CLASSES]),dtype=tf.float32, name='W_o')
    b_o =tf.Variable(tf.zeros(shape=[CLASSES]),dtype=tf.float32, name='b_o')

# parameter saver
param_list =[W_h1, b_h1, W_h2, b_h2, W_o, b_o]
saver = tf.train.Saver(param_list)



with tf.name_scope('hidden_layer_1') as h1scope:
    hidden1=tf.sigmoid(tf.matmul(x,W_h1) + b_h1,name ='hidden1')
with tf.name_scope('hidden_layer_2') as h2scope:
    hidden2=tf.sigmoid(tf.matmul(hidden1,W_h2) + b_h2, name= 'hidden2')
with tf.name_scope('output_layer') as oscope:    
    y= tf.sigmoid(tf.matmul(hidden2,W_o) + b_o, name ='y')
    
    
with tf.name_scope('calcuclate_costs'):
    cost = tf.reduce_sum((-y_*tf.log(y)-(1-y_)*tf.log(1-y)),reduction_indices=1)
    cost = tf.reduce_mean(cost)
    #tf.scalar_summary('cost',cost)
    
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))

with tf.name_scope('training'):
    train=tf.train.GradientDescentOptimizer(lr).minimize(cost)

with tf.name_scope('evaluation'):
    comp_pred = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(comp_pred, tf.float32))



#with tf.Session() as sess:
sess= tf.Session()


#saver.restore(sess,'./tensorflow_live.ckpt') 
sess.run(tf.global_variables_initializer())

#merge= tf.merge_all_summaries()
#train_writer= tf.train.SummaryWriter('./summaries',sess.graph)
        
for i in range(4001):
    #_, loss=sess.run([train, cost],feed_dict=tensor_map)
    _,loss , acc =sess.run([train, cost, accuracy],feed_dict=tensor_map)
    #train_writer.add_summary(summary,i)
    if i %1000 ==0:
        saver.save(sess,'./tensorflow_live.ckpt')
        print "------------------------------"
        print "step: ", i 
        print "loss: ", loss
        print "acc: ", acc
        
sess.close()    
    
    
    