import tensorflow as tf
#from ipdb import set_trace as debug
from IPython import embed as debug

input_dim = 2

with tf.name_scope('input') as scope:
    I = tf.placeholder(tf.float32, [None,input_dim])#, None])

j = 0
with tf.name_scope('layer'+str(j)) as s1:
    for i in range(10):
        with tf.name_scope('hidden'+ str(i)) as scope:
            a = tf.constant(5, dtype=tf.float32, name='alpha')
            W = tf.Variable(tf.random_uniform([1,input_dim], -1.0, 1.0), name='weights')
            b = tf.Variable(tf.zeros([1]), name='biases')
        with tf.name_scope('activation') as scope:
            pre_act = W*I + b
            act = a*tf.nn.relu(pre_act)

init = tf.global_variables_initializer()

s = []
for i in range(10):
    for j in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='layer0/hidden'+str(i)):
        if type(s) == list:
            s = j
        else:
            s +=j
#s = tf.abs(s)
s = tf.Print(s, [s], message="This is a: ")

y_g = tf.constant([[1,0]], dtype=tf.float32)
cost = tf.reduce_sum(tf.abs(y_g - s))#y_g*tf.log(s))
cost = tf.Print(cost, [cost], message="cost: ")
tf.summary.scalar('cost', cost)

with tf.name_scope("train") as _:
    opti = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
merged_summary_op = tf.summary.merge_all()

dirpath = 'logs'
if tf.gfile.Exists(dirpath):
       tf.gfile.DeleteRecursively(dirpath) 
tf.gfile.MkDir(dirpath)
_input = [[1,2],[4,5],[2,3]]

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(dirpath, graph=sess.graph)
    sess.run(init)
    for k in range(100):
        avg_cost = 0
        sess.run(opti, feed_dict={I: _input})
        summary_str = sess.run(merged_summary_op, feed_dict={I:_input})
        summary_writer.add_summary(summary_str, k)





  debug()


