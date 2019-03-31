import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
rng = np.random
#参数
learning_rate = 0.01
training_epochs = 1000
display_step = 50
#训练数据
train_x = np.asarray([1.2,3.3,4.5,6.93,4.45,7.56,6.0,5.4,7.89,8.0])
train_y = np.asarray([1.4,2.3,4.6,6.4,7.0,7.89,6.7,6.4,8.5,6.7])
n_samples = train_x.shape[0]

X = tf.placeholder('float32')
Y = tf.placeholder('float32')
#设置模型权重和偏置
W = tf.Variable(rng.randn(),name='weight')
b = tf.Variable(rng.randn(),name='bias')
#建立线性模型
pred = tf.add(tf.multiply(X,W),b)
#损失函数、
cost = tf.reduce_sum(tf.pow(pred - Y,2))/(2*n_samples)
#梯度下降
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#初始化变量
init = tf.global_variables_initializer()

#开始训练                        ]
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
       for (x,y) in zip(train_x,train_y):
           sess.run(optimizer,feed_dict={X:x,Y:y})
       if(epoch+1)%display_step ==0:
            c = sess.run(cost,feed_dict={X:train_x,Y:train_y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))
    print('optimization is finisshed')
    training_cost = sess.run(cost,feed_dict={X:train_x,Y:train_y})
    print('Training cost=',training_cost,'W=',sess.run(W),'b=',sess.run(b))

    # Graphic display
    plt.plot(train_x, train_y, 'ro', label='Original data')
    plt.plot(train_x, sess.run(W) * train_x + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

