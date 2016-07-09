import tensorflow as tf
import math

dim_output = 3


def weight_init(shape, std=0.1):
    initial = tf.truncated_normal(shape, stddev=std)
    return tf.Variable(initial, trainable=True)


class QAgent(object):
    def __init__(self, opt):
        self.w1 = weight_init([8, 8, 4, 16], std=math.sqrt(2.0 / 8 / 8 / 4))
        self.w2 = weight_init([4, 4, 16, 32], std=math.sqrt(2.0 / 4 / 4 / 16))
        self.w3 = weight_init([3200, 256], std=math.sqrt(2.0 / 256))
        self.w4 = weight_init([256, dim_output], std=math.sqrt(1.0 / dim_output))
        self.s = tf.placeholder(tf.float32, [None, 80, 80, 4])
        self.y = tf.placeholder(tf.float32, [None])
        self.opt = opt

        def conv_relu(x, kernel, stride, padding):
            h = tf.nn.conv2d(x, kernel, stride, padding)
            return tf.nn.relu(h)

        def forward(self):
            l1 = conv_relu(self.s, self.w1, [1, 4, 4, 1], "SAME")
            l2 = conv_relu(l1, self.w2, [1, 2, 2, 1], "SAME")
            flat_dim = 3200
            l2_flat = tf.reshape(l2, [-1, flat_dim])
            fc1 = tf.nn.relu(tf.matmul(l2_flat, self.w3))
            q_value = tf.matmul(fc1, self.w4)
            return q_value

        self.q_value = forward(self)

        def get_weights(self):
            return [self.w1, self.w2, self.w3, self.w4]

        self.trainable_weights = get_weights(self)
        self.actions = tf.placeholder(tf.float32, [None, dim_output])

        def grad_update_op(self):
            action_q_values = tf.reduce_sum(tf.mul(self.q_value, self.actions), reduction_indices=1)
            loss = tf.reduce_mean(tf.square(self.y - action_q_values))
            grad_op = self.opt.compute_gradients(loss, self.trainable_weights)
            return grad_op

        self.grad_op = grad_update_op(self)


class A3CAgent(object):
    def __init__(self, opt):
        self.w1 = weight_init([8, 8, 4, 16], std=math.sqrt(2.0 / 8 / 8 / 4))
        self.w2 = weight_init([4, 4, 16, 32], std=math.sqrt(2.0 / 4 / 4 / 16))
        self.w3 = weight_init([3200, 256], std=math.sqrt(2.0 / 256))
        self.value_w4 = weight_init([256, 1], std=1.0)
        self.pi_w4 = weight_init([256, dim_output], std=math.sqrt(1.0/dim_output))
        self.s = tf.placeholder(tf.float32, [None, 80, 80, 4])
        self.R = tf.placeholder(tf.float32, [None])
        self.opt = opt

        def conv_relu(x, kernel, stride, padding):
            h = tf.nn.conv2d(x, kernel, stride, padding)
            return tf.nn.relu(h)

        l1 = conv_relu(self.s, self.w1, [1, 4, 4, 1], "SAME")
        l2 = conv_relu(l1, self.w2, [1, 2, 2, 1], "SAME")
        flat_dim = 3200
        l2_flat = tf.reshape(l2, [-1, flat_dim])
        self.fc1 = tf.nn.relu(tf.matmul(l2_flat, self.w3))

        def value(self):
            q_value = tf.matmul(self.fc1, self.value_w4)
            q_value = tf.reshape(q_value, [-1])
            return q_value

        self.q_value = value(self)

        def pi(self):
            policy = tf.matmul(self.fc1, self.pi_w4)
            policy = tf.nn.softmax(policy)
            return policy

        self.pi_value = pi(self)

        def get_weights(self):
            return [self.w1, self.w2, self.w3, self.value_w4, self.pi_w4]

        self.trainable_weights = get_weights(self)
        # compute loss and gradients
        # some placeholder
        self.beta = tf.placeholder(tf.float32)
        self.factor = tf.placeholder(tf.float32, [None])
        self.actions = tf.placeholder(tf.float32, [None, dim_output])

        def grad_update_op(self):
            entropy = -tf.reduce_sum(tf.mul(self.pi_value, tf.log(self.pi_value)), reduction_indices=1)
            pi_action = tf.reduce_sum(tf.mul(self.pi_value, self.actions), reduction_indices=1)
            policy_loss = -self.beta * entropy - tf.mul(tf.log(pi_action), self.factor)
            value_loss = 0.5*tf.square(self.R - self.q_value)
            total_loss = tf.reduce_sum(policy_loss + value_loss)
            grad_op = self.opt.compute_gradients(total_loss, self.trainable_weights)
            return grad_op

        self.grad_op = grad_update_op(self)
