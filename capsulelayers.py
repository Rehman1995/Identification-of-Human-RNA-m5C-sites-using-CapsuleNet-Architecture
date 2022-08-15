import keras.backend as K
import tensorflow as tf
from keras import initializers, layers,regularizers
from keras.layers import Dropout

class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))
    
    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


class Mask(layers.Layer):
    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            x = K.sqrt(K.sum(K.square(inputs), -1))
            # generate the mask which is a one-hot code.
            # mask.shape=[None, n_classes]=[None, num_capsule]
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])
        
        # inputs.shape=[None, num_capsule, dim_capsule]
        # mask.shape=[None, num_capsule]
        # masked.shape=[None, num_capsule * dim_capsule]
        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked
    
    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])


def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

class CapsuleLayer(layers.Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the 
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
    [None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.
    
    :param num_capsule: number of capsules in this layer 10 (output layer)
    :param dim_capsule: dimension of the output vectors of the capsules in this layer 16 (output layer)
    :param num_routing: number of iterations for the routing algorithm
    """
    def __init__(self, num_capsule, dim_capsule, num_routing=3,
                 kernel_initializer='glorot_uniform',kernel_regularizer=None,
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule 
        self.dim_capsule = dim_capsule
        self.num_routing = num_routing
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        
    
    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1] #1152
        self.input_dim_capsule = input_shape[2] #8
        
        # Transform matrix
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 name='W')
        
        self.built = True
    
    
    def call(self, inputs, training=None):
      # Expand the input in axis=1, tile in that axis to num_capsule, and 
      # expands another axis at the end to prepare the multiplication with W.
      #  inputs.shape=[None, input_num_capsule, input_dim_capsule]
      #  inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule]
      #  inputs_tiled.shape=[None, num_capsule, input_num_capsule, 
      #                            input_dim_capsule, 1]
      inputs_expand = tf.expand_dims(inputs, 1)
      inputs_tiled  = tf.tile(inputs_expand, [1, self.num_capsule, 1, 1])
      inputs_tiled  = tf.expand_dims(inputs_tiled, 4)
    
      # Compute `W * inputs` by scanning inputs_tiled on dimension 0 (map_fn).
      # - Use matmul (without transposing any element). Note the order!
      # Thus:
      #  x.shape=[num_capsule, input_num_capsule, input_dim_capsule, 1]
      #  W.shape=[num_capsule, input_num_capsule, dim_capsule,input_dim_capsule]
      # Regard the first two dimensions as `batch` dimension,
      # then matmul: [dim_capsule, input_dim_capsule] x [input_dim_capsule, 1]-> 
      #              [dim_capsule, 1].
      #  inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule, 1]
      
      inputs_hat = tf.map_fn(lambda x: tf.matmul(self.W, x), elems=inputs_tiled)     
    
      # Begin: Routing algorithm ----------------------------------------------#
      # The prior for coupling coefficient, initialized as zeros.
      #  b.shape = [None, self.num_capsule, self.input_num_capsule, 1, 1].
      b = tf.zeros(shape=[tf.shape(inputs_hat)[0], self.num_capsule, 
                          self.input_num_capsule, 1, 1])
    
      assert self.num_routing > 0, 'The routings should be > 0.'
      for i in range(self.num_routing):
        # Apply softmax to the axis with `num_capsule`
        #  c.shape=[batch_size, num_capsule, input_num_capsule, 1, 1]
        c = layers.Softmax(axis=1)(b)
    
        # Compute the weighted sum of all the predicted output vectors.
        #  c.shape =  [batch_size, num_capsule, input_num_capsule, 1, 1]
        #  inputs_hat.shape=[None, num_capsule, input_num_capsule,dim_capsule,1]
        # The function `multiply` will broadcast axis=3 in c to dim_capsule.
        #  outputs.shape=[None, num_capsule, input_num_capsule, dim_capsule, 1]
        # Then sum along the input_num_capsule
        #  outputs.shape=[None, num_capsule, 1, dim_capsule, 1]
        # Then apply squash along the dim_capsule
        outputs = tf.multiply(c, inputs_hat)
        outputs = tf.reduce_sum(outputs, axis=2, keepdims=True)
        outputs = squash(outputs, axis=-2)  # [None, 10, 1, 16, 1]
    
        if i < self.num_routing - 1:
          # Update the prior b.
          #  outputs.shape =  [None, num_capsule, 1, dim_capsule, 1]
          #  inputs_hat.shape=[None,num_capsule,input_num_capsule,dim_capsule,1]
          # Multiply the outputs with the weighted_inputs (inputs_hat) and add  
          # it to the prior b.  
          outputs_tiled = tf.tile(outputs, [1, 1, self.input_num_capsule, 1, 1])
          agreement = tf.matmul(inputs_hat, outputs_tiled, transpose_a=True)
          b = tf.add(b, agreement)
    
      # End: Routing algorithm ------------------------------------------------#
      # Squeeze the outputs to remove useless axis:
      #  From  --> outputs.shape=[None, num_capsule, 1, dim_capsule, 1]
      #  To    --> outputs.shape=[None, num_capsule,    dim_capsule]
      outputs = tf.squeeze(outputs, [2, 4])
      return outputs

    def compute_output_shape(self, input_shape):
        #return tuple([None, self.num_capsule, self.dim_capsule])
        return tuple([None, self.num_capsule, self.dim_capsule+self.input_num_capsule])

class CapsuleLayer_nogradient_stop(layers.Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the 
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer_nogradient_stop just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
    [None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.
    
    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param num_routing: number of iterations for the routing algorithm
    """
    def __init__(self, num_capsule, dim_capsule, num_routing=3,dropout=0,
                 kernel_initializer='glorot_uniform',kernel_regularizer=None,
                 **kwargs):
        super(CapsuleLayer_nogradient_stop, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.num_routing = num_routing
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.dropout = dropout
    
    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        
        # Transform matrix
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 name='W')
        
        self.built = True
    
    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_capsule]
        # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule]
        inputs_expand = K.expand_dims(inputs, 1)
        
        # Replicate num_capsule dimension to prepare being multiplied by W
        # inputs_tiled.shape=[None, num_capsule, input_num_capsule, input_dim_capsule]
        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])
        ####dropout on inputs_tiles#############################
        #K.in_train_phase(K.dropout(inputs_tiled,0.2,noise_shape=None),inputs_tiled,training=training)
        # Compute `inputs * W` by scanning inputs_tiled on dimension 0.
        # x.shape=[num_capsule, input_num_capsule, input_dim_capsule]
        # W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # Regard the first two dimensions as `batch` dimension,
        # then matmul: [input_dim_capsule] x [dim_capsule, input_dim_capsule]^T -> [dim_capsule].
        # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)
        ##########dropout on inputs_hat
        K.in_train_phase(K.dropout(inputs_hat,self.dropout,noise_shape=None),inputs_hat,training=training)
        # Begin: Routing algorithm ---------------------------------------------------------------------#
        # The prior for coupling coefficient, initialized as zeros.
        # b.shape = [None, self.num_capsule, self.input_num_capsule].
        b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])
        if self.num_routing ==0:
            c = tf.nn.softmax(b, dim=1)
            outputs=squash(K.batch_dot(c, inputs_hat, [2, 2]))
        #assert self.num_routing > 0, 'The num_routing should be > 0.'
        for i in range(self.num_routing):
            # c.shape=[batch_size, num_capsule, input_num_capsule]
            c = tf.nn.softmax(b)
            
            # c.shape =  [batch_size, num_capsule, input_num_capsule]
            # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
            # The first two dimensions as `batch` dimension,
            # then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
            # outputs.shape=[None, num_capsule, dim_capsule]
            outputs = tf.multiply(c, inputs_hat)
            outputs = tf.reduce_sum(outputs, axis=2, keepdims=True)
            outputs = squash(outputs, axis=-2)  # [None, 10, 1, 16, 1]            
            if i < self.num_routing - 1:
                # outputs.shape =  [None, num_capsule, dim_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # The first two dimensions as `batch` dimension,
                # then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
                # b.shape=[batch_size, num_capsule, input_num_capsule]
                b += K.batch_dot(outputs, inputs_hat, [2, 3])
        # End: Routing algorithm -----------------------------------------------------------------------#
        
        all=K.concatenate([outputs,c])
        return all
    
    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])


def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding,dropout,kernel_initializer='glorot_uniform'):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_capsule: the dim of the output vector of capsule 8
    :param n_channels: the number of types of capsules 32
    :param kernel_size: 6
    :return: output tensor, shape=[None, num_capsule, dim_capsule]
    """
    output = layers.Conv1D(filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding,kernel_initializer=kernel_initializer,
                           name='primarycap_conv2d')(inputs)
    output = Dropout(dropout)(output)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)