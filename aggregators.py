import tensorflow as tf
from abc import abstractmethod


LAYER_IDS = {}


def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class Aggregator(object):
    def __init__(self, batch_size, seq_len, dim, dropout, act, name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dim = dim

    def __call__(self, self_vectors, neighbor_vectors, question_embeddings):
        outputs = self._call(self_vectors, neighbor_vectors, question_embeddings)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, question_embeddings):
        # dimension:
        # self_vectors: [batch_size, -1, dim]
        # neighbor_vectors: [batch_size, -1, n_neighbor, dim]
        # neighbor_relations: [batch_size, -1, n_neighbor, dim]
        # user_embeddings: [batch_size, dim]
        pass




class SumAggregator(Aggregator):
    """
    累加聚合，直接累加
    """
    def __init__(self, batch_size, seq_len, dim, dropout=0., act=tf.nn.relu, name=None):
        super(SumAggregator, self).__init__(batch_size,seq_len,dim, dropout, act, name)
        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')



    def _call(self,self_vectors, neighbor_vectors, question_embeddings):
        # [batch_size,seq_len, -1, dim]
        neighbors_agg = tf.reduce_mean(neighbor_vectors, axis=-2) #以倒数第二维为轴作平均值，也就消除了倒数第二维代表邻居轴的
        #也就是把周围的向量求平均值 然后在下一行相加 再Reshape产生每个人每个习题向量 一次卷积后的结果
        output = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        #neighbors_agg = tf.concat([tf.reshape(self_vectors,[self.batch_size,self.seq_len,-1,1,self.dim]),neighbor_vectors],-2)

        # [-1, dim]
        #output = tf.reshape(tf.reduce_mean(neighbors_agg,-2), [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=self.dropout)
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size,seq_len, -1, dim]
        output = tf.reshape(output, [self.batch_size,self.seq_len, -1, self.dim])#似乎多了一维

        return self.act(output)



class ConcatAggregator(Aggregator):
    """
    连接聚合，两个向量连接
    """
    def __init__(self, batch_size,seq_len, dim, dropout=0., act=tf.nn.relu, name=None):
        super(ConcatAggregator, self).__init__(batch_size, seq_len,dim, dropout, act, name)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim * 2, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, question_embeddings):
        # [batch_size,seq_len, -1, dim]
        neighbors_agg = tf.reduce_mean(neighbor_vectors, axis=-2)

        # [batch_size,seq_len, -1, dim * 2]
        output = tf.concat([self_vectors, neighbors_agg], axis=-1)

        # [-1, dim * 2]
        output = tf.reshape(output, [-1, self.dim * 2])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)

        # [-1, dim]
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size,self.seq_len, -1, self.dim])

        return self.act(output)


