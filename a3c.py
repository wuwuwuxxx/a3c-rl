import gym
import tensorflow as tf
import numpy as np
import time

from agents import A3CAgent
from atari import MyEnv

cp = 'Pong-v0'
dim_output = 3
# valid action are [1, 2, 3]

T = 0
Tmax = 8e7
num_thread = 16
init_lr = 1e-4


def get_action(x):
    return np.random.choice(np.size(x), p=x.reshape(dim_output))


def build_copy_op(from_, to_):
    f = from_.trainable_weights
    t = to_.trainable_weights
    op = [t[k].assign(f[k]) for k in xrange(len(f))]
    return op


def train_loop(env, graph, id, sess, thread_net, saver, thread_copy, apply_grad_op, grad_placeholder):
    global Tmax, T
    t = 1
    tmax = 100000000
    gamma = 0.99
    beta = 0.01
    mean_reward = None
    env = MyEnv(env, 4)

    observation = env.reset()
    done = None
    total_r = 0
    grads = [thread_net.grad_op[k][0] for k in xrange(len(grad_placeholder))]
    while T < Tmax:

        sess.run(thread_copy)

        t_start = t
        obs = []
        rs = []
        actions = []
        while True:
            obs.append(observation)
            pi = thread_net.pi_value.eval(session=sess, feed_dict={thread_net.s: [observation]})
            action = get_action(pi)
            at = np.zeros([dim_output])
            at[action] = 1
            actions.append(at)

            observation, reward, done, info = env.step(action)
            rs.append(reward)

            t += 1
            T += 1

            if t % 10000 == 0 and id == 0:
                saver.save(sess, 'checkpoint/model.ckpt', global_step=T)
                print "model saved!"

            total_r += reward

            if t - t_start == tmax or reward != 0:
                break

        if reward != 0:
            R = 0
        else:
            R = thread_net.q_value.eval(session=sess, feed_dict={thread_net.s: [observation]})

        Rs = np.zeros_like(rs)
        for j in reversed(xrange(0, np.size(rs))):
            Rs[j] = rs[j] + gamma * R
            R = Rs[j]


        with graph.device('/gpu:0'):
            factor = Rs - thread_net.q_value.eval(session=sess, feed_dict={thread_net.s: obs})

            fd = {}
            grads_data = sess.run(grads, feed_dict={
                thread_net.s: obs,
                thread_net.R: Rs,
                thread_net.beta: beta,
                thread_net.factor: factor,
                thread_net.actions: actions,
            })
        for i in xrange(len(grad_placeholder)):
            fd[grad_placeholder[i][0]] = grads_data[i]
        sess.run(apply_grad_op, feed_dict=fd)

        # do some log work
        if T > Tmax or done:
            observation = env.reset()
            mean_reward = (0.9 * mean_reward + 0.1 * total_r) if mean_reward is not None else total_r
            total_r = 0
            print "total step: {}, thread {:2d}: {}, mean reward: {:2f}, learning_rate: {}".format(T, id, t, mean_reward, sess.run(lr))
    print "thread id: %d ended with mean_reward: %.2f" % (id, mean_reward)


if __name__ == "__main__":
    g = tf.Graph()
    with g.as_default():
        with g.device('/cpu:0'):
            global_step = tf.Variable(0, trainable=False)
            lr = tf.train.exponential_decay(init_lr, global_step, 50000, 0.999)
            sess = tf.Session()
            opt = tf.train.RMSPropOptimizer(lr, decay=0.99)

            # init shared and target network
            shared_net = A3CAgent(opt)
            # init thread-specific network
            thread_networks = [A3CAgent(opt) for _ in xrange(num_thread)]

            grad_placeholder = [(tf.placeholder(tf.float32, var.get_shape()), var) for _, var in shared_net.grad_op]
            apply_grad_op = opt.apply_gradients(grad_placeholder, global_step=global_step)

            saver = tf.train.Saver(tf.all_variables())
            # sess.run(tf.initialize_all_variables())
            ckpt = tf.train.get_checkpoint_state('checkpoint/')
            saver.restore(sess, ckpt.model_checkpoint_path)

            # build copy op
            copy_to_threads = [build_copy_op(shared_net, thread_networks[k]) for k in xrange(num_thread)]

            threads = []
            envs = [gym.make(cp) for _ in xrange(num_thread)]

            coord = tf.train.Coordinator()

            for k in xrange(num_thread):
                threads.append(tf.train.LooperThread(coord, None, target=train_loop, args=(envs[k],
                                                                                           g,
                                                                                           k,
                                                                                           sess,
                                                                                           thread_networks[k],
                                                                                           saver,
                                                                                           copy_to_threads[k],
                                                                                           apply_grad_op,
                                                                                           grad_placeholder)))

            start_time = time.time()
            for t in threads:
                t.start()

            for t in threads:
                t.join()

            print "duration:", time.time() - start_time
            saver.save(sess, 'checkpoint/model.ckpt', global_step=T)
