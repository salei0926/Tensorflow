import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
''''Xaiver初始化器，让权重的初始化大小正好合适'''
def xaiver_init(fan_in,fan_out,constant = 1):
    '''
    :param fan_in: 输入节点数量
    :param fan_out: 输出节点数量
    :param constant:
    :return:
    '''
    low = -constant *np.sqrt(6.0/(fan_in + fan_out))
    high = constant *np.sqrt(6.0/(fan_in + fan_out))

    return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype=tf.float32)

'''去噪编码的类'''
class AdditiveGaussianNoiseAutoencoder():
    def __init__(self,n_input,n_hidden,transfer_function=tf.nn.softplus,
                 optimizer = tf.train.AdamOptimizer(),scale=0.1):
        '''
        :param n_input: 输入变量数
        :param n_hidden: 隐含层节点数
        :param transfer_function: 隐含层激活函数
        :param optimizer: 优化器
        :param scale: 高斯噪声系数
        '''
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights() #用于参数初始化
        self.weights = network_weights

        self.x = tf.placeholder(tf.float32,[None,self.n_input])
        #建立隐含层，，，将x加上噪音
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),
                                                     self.weights['w1']),self.weights['b1']))
        #输出层，，，，，进行数据复原重建操作
        self.reconstruction = tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])

        self.cost = 0.5*tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction,self.x),2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    '''去噪编码的类               end'''

    def _initialize_weights(self):
        all_weights = dict()         #存放所有参数
        all_weights['w1'] = tf.Variable(xaiver_init(self.n_input,self.n_hidden))#初始化w1
        all_weights['b1'] = tf.Variable(tf.zeros(self.n_hidden),dtype=tf.float32)
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,self.n_input],dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],dtype=tf.float32))
        return all_weights
    #在节点上操作，计算损失函数，用于训练集
    def partial_fit(self,X):
        cost,opt = self.sess.run((self.cost,self.optimizer),feed_dict={self.x:X,self.scale:self.training_scale})
        return cost
    #在节点上操作，计算损失函数，用于测试集
    def calc_total_cost(self,X):
        return self.sess.run(self.cost,feed_dict={self.x:X,self.scale:self.training_scale})

    #返回自编码器隐含层的输出结果
    def transform(self,X):
        return self.sess.run(self.hidden,feed_dict={self.x:X,self.scale:self.training_scale})

    #将隐含层的输出结果作为输入，对重建层节点进行操作
    def generate(self,hidden = None):
        if hidden is None:
            hidden = np.random.normal(size = self.weights['b1'])
        return self.sess.run(self.reconstruction,feed_dict={self.hidden:hidden})

    #输入原数据，输出复原数据，输出层
    def reconstruct(self,X):
        return self.sess.run(self.reconstruction,feed_dict={self.x:X,self.scale:self.training_scale})
    #获取隐含层权重
    def getWeights(self):
        return self.sess.run(self.weights['w1'])
    #获取隐含层偏置
    def getBiases(self):
        return self.sess.run(self.weights['b'])

#对训练测试数据进行标准化处理
def standard_scale(X_train,X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train,X_test

def get_random_block_from_data(data,batch_size):
    start_index = np.random.randint(0,len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]
#对训练集，测试集进行标准化变换
X_train,X_test = standard_scale(mnist.train.images,mnist.test.images)

n_samples = int(mnist.train.num_examples)
training_epochs = 20      #最大训练轮数
batch_size= 128
display_step = 1          #每隔一轮输出一次cost

#创建自编码器实例
autoencoder = AdditiveGaussianNoiseAutoencoder(n_input = 784,
                                               n_hidden=200,
                                               transfer_function=tf.nn.softplus,
                                               optimizer=tf.train.AdamOptimizer(0.001),
                                               scale = 0.01)

for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples/batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train,batch_size)

        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost / n_samples * batch_size
    if epoch % display_step ==0:
        print('Epoch:','%04d' %(epoch+1),'cost=','{:.9f}'.format(avg_cost))
print('Total cost:'+ str(autoencoder.calc_total_cost(X_test)))

