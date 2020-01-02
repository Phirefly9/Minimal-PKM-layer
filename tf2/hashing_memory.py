import math
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, BatchNormalization, Dropout, Softmax

def get_uniform_keys(n_keys, dim, seed):
    """
    Generate random uniform keys (same initialization as nn.Linear).
    """
    bound = 1 / math.sqrt(dim)
    keys = tf.random.uniform(shape=(n_keys, dim), minval=-bound, maxval=bound, seed=seed)
    return keys

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class HashingMemory(Model):

    def __init__(self, input_dim, output_dim, params):

        super().__init__()

        # global parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k_dim = params.k_dim
        self.v_dim = output_dim
        self.n_keys = params.n_keys
        self.size = self.n_keys ** 2
        self.heads = params.heads
        self.knn = params.knn
        assert self.k_dim >= 2 and self.k_dim % 2 == 0

        # dropout
        self.input_dropout = params.input_dropout
        self.input_dropout_layer = Dropout(self.input_dropout)
        self.query_dropout = params.query_dropout
        self.query_dropout_layer = Dropout(self.query_dropout)
        self.value_dropout = params.value_dropout
        self.value_dropout_layer = Dropout(self.value_dropout)

        self.softmax = Softmax()

        # initialize keys / values
        self.initialize_keys()
        # torch uses embedding bag, which is unified embedding grab and then sum
        # we need to follow up with our own implementation
        self.values = Embedding(self.size, self.v_dim)
        # TODO: check to see if initializer is correct
        # nn.init.normal_(self.values.weight, mean=0, std=self.v_dim ** -0.5)

        self.query_proj = tf.keras.Sequential()
        self.query_proj.add(Dense(self.heads * self.k_dim, input_dim=self.input_dim))
        if params.query_batchnorm:
            self.query_proj.add(BatchNormalization())

        if params.query_batchnorm:
            tf.print("WARNING: Applying batch normalization to queries improves the performance "
                  "and memory usage. But if you use it, be sure that you use batches of "
                  "sentences with the same size at training time (i.e. without padding). "
                  "Otherwise, the padding token will result in incorrect mean/variance "
                  "estimations in the BatchNorm layer.\n")

    
    def initialize_keys(self):
        """
        Create two subkey sets per head.
        `self.keys` is of shape (heads, 2, n_keys, k_dim // 2)
        """
        half = self.k_dim // 2
        self.keys = tf.reshape(tf.convert_to_tensor([get_uniform_keys(self.n_keys, half, seed=(2 * i + j))
            for i in range(self.heads)
            for j in range(2)]), (self.heads, 2, self.n_keys, half))
    
    def _get_indices(self, query, subkeys):
        """
        Generate scores and indices for a specific head.
        """
        shape = tf.shape(query)
        tf.debugging.assert_equal(len(shape), 2)
        tf.debugging.assert_equal(shape[1], self.k_dim)
        bs = shape[0]
        knn = self.knn
        half = self.k_dim // 2
        n_keys = len(subkeys[0])

        # split query for product quantization
        q1 = query[:, :half]                                          # (bs,half)
        q2 = query[:, half:]                                          # (bs,half)

        # compute indices with associated scores
        scores1 = tf.matmul(q1, tf.transpose(subkeys[0]))                # (bs,n_keys)
        scores2 = tf.matmul(q2, tf.transpose(subkeys[1]))                # (bs,n_keys)
               
        scores1, indices1 = tf.math.top_k(scores1, k=knn)             # (bs,knn)
        scores2, indices2 = tf.math.top_k(scores2, k=knn)             # (bs,knn)

        # cartesian product on best candidate keys
        all_scores = tf.reshape(
            tf.broadcast_to(tf.reshape(scores1, [bs, knn, 1]), [bs, knn, knn]) +
            tf.broadcast_to(tf.reshape(scores2, [bs, 1, knn]), [bs, knn, knn])
        , [bs, -1])                                                   # (bs,knn**2)

        all_indices = tf.reshape(
            tf.broadcast_to(tf.reshape(indices1, [bs, knn, 1]), [bs, knn, knn]) * n_keys + 
            tf.broadcast_to(tf.reshape(indices2, [bs, 1, knn]), [bs, knn, knn])
        , [bs, -1])                                                # (bs,knn**2)

        # select best scores with associated indices
        scores, best_indices = tf.math.top_k(all_scores, k=knn)         # (bs,knn)
        
        indices = tf.gather(all_indices, best_indices, batch_dims=1)    # (bs,knn)
        
        tf.debugging.assert_equal(scores.shape, indices.shape)
        tf.debugging.assert_equal(indices.shape, [bs, knn])
        return scores, indices

    def get_indices(self, query):
        """
        Generate scores and indices.
        """
        shape = tf.shape(query)
        
        tf.debugging.assert_equal(len(query.shape), 2)
        tf.debugging.assert_equal(query.shape[1], self.k_dim)

        query = tf.reshape(query, [-1, self.heads, self.k_dim])
        bs = len(query)
        outputs = [self._get_indices(query[:, i], self.keys[i]) for i in range(self.heads)]
        s = tf.concat([tf.reshape(s, [bs, 1, self.knn]) for s, _ in outputs], axis=1)
        i = tf.concat([tf.reshape(i, [bs, 1, self.knn]) for _, i in outputs], axis=1)
        return tf.reshape(s, [-1, self.knn]), tf.reshape(i, [-1, self.knn])
    
    def call(self, x):
        """
        Read from the memory.
        """
        # input dimensions
        shape = tf.shape(x)
        tf.debugging.assert_equal(x.shape[-1], self.input_dim)
        prefix_shape = shape[:-1]
        bs = tf.math.reduce_prod(prefix_shape)

        # compute query 
        x = self.input_dropout_layer(x)                                         # (...,i_dim)
        query = self.query_proj(tf.reshape(x, [-1, self.input_dim]))            # (bs,heads*k_dim)  
        query = tf.reshape(query, [bs*self.heads, self.k_dim])                  # (bs*heads,k_dim)
        query = self.query_dropout_layer(query)                                 # (bs*heads,k_dim)
        tf.debugging.assert_equal(query.shape, [bs * self.heads, self.k_dim])

        # retrieve indices and scores
        scores, indices = self.get_indices(query)                               # (bs*heads,knn)
        scores = self.softmax(scores, )                                         # (bs*heads,knn)

        # merge heads / knn (since we sum heads)
        indices = tf.reshape(indices, [bs, self.heads * self.knn])              # (bs,heads*knn)                      
        scores = tf.reshape(scores, [bs, self.heads*self.knn])                  # (bs,heads*knn)

        # weighted sum of values, original implementation used an embedding bag, to reduce memory usage
        # unfortunately keras doesn't have anything like that
        output = self.values(indices)
        output = tf.reduce_sum(output, axis=1)
        output = output * scores                                                 # (bs,v_dim)
        
        output = self.value_dropout_layer(output)

        # reshape output
        if len(prefix_shape) >= 2:
            output = tf.reshape(output, [prefix_shape, self.v_dim])              # (...,v_dim)

        return output