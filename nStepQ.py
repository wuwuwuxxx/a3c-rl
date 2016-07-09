import gym
import tensorflow as tf
import numpy as np
import time

from agents import QAgent
from atari import MyEnv

cp = 'Pong-v0'
dim_output = 3
# valid action are [1, 2, 3]

T = 0
Tmax = 8e7
num_thread = 8
anneal_step = 4000000 / num_thread
init_lr = 7e-4


def get_action(x, eps=0.00001):
    if np.random.random() < eps:
        return np.random.randint(0, np.size(x))
    return np.argmax(x)


def build_copy_op(from_, to_):
    f = from_.trainable_weights
    t = to_.trainable_weights
    op = [t[k].assign(f[k]) for k in xrange(len(f))]
    return op


def sample_final_epsilon():
    """
    Sample a final epsilon value to anneal towards from a distribution.
    These values are specified in section 5.1 of http://arxiv.org/pdf/1602.01783v1.pdf
    """
    final_epsilons = np.array([.1, .01, .5])
    probabilities = np.array([0.4, 0.3, 0.3])
    return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]


def train_loop(grad_placeholder, apply_grad_op, env, id, sess, target_net, thread_net, saver, target_copy, thread_copy):
    time.sleep(id)
    global Tmax, T
    t = 1
    tmax = 5
    sess.run(target_copy)
    gamma = 0.99
    mean_reward = None
    env = MyEnv(env, 4)

    init_eps = 1
    final_eps = sample_final_epsilon()
    epsilon = init_eps

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
            tq_value = thread_net.q_value.eval(session=sess, feed_dict={thread_net.s: [observation]})
            action = get_action(tq_value, eps=epsilon)
            at = np.zeros([dim_output])
            at[action] = 1
            actions.append(at)

            # Scale down epsilon
            if epsilon > final_eps:
                epsilon -= (init_eps - final_eps) / anneal_step

            new_observation, reward, done, info = env.step(action)
            rs.append(reward)

            observation = new_observation
            t += 1
            T += 1

            if T % 40000 == 0:
                sess.run(target_copy)
                print "updating target network parameters!!!!"

            if t % 10000 == 0 and id == 0:
                saver.save(sess, 'checkpoint/model.ckpt', global_step=T)
                print "model saved!!!!"

            total_r += reward

            if t - t_start == tmax or done:
                break

        if done:
            R = 0
            observation = env.reset()
        else:
            R = np.amax(target_net.q_value.eval(session=sess, feed_dict={target_net.s: [new_observation]}))

        Rs = np.zeros_like(rs)
        Rs[-1] = R
        for j in reversed(xrange(1, len(rs))):
            Rs[j - 1] = rs[j] + gamma * Rs[j]

        fd = {}
        grads_data = sess.run(grads, feed_dict={
            thread_net.s: obs,
            thread_net.y: Rs,
            thread_net.actions: actions,
        })
        for i in xrange(len(grad_placeholder)):
            fd[grad_placeholder[i][0]] = grads_data[i]

        sess.run(apply_grad_op, feed_dict=fd)

        if T > Tmax or done:
            mean_reward = (0.9 * mean_reward + 0.1 * total_r) if mean_reward is not None else total_r
            total_r = 0
            print "step: %d, thread id: %2d, lr:%f, mean reward: %.4f, eps: %.7f -> %f" % (
                T, id, sess.run(lr), mean_reward, epsilon, final_eps)
    print "thread id: %d ended with mean_reward: %.2f" % (id, mean_reward)


if __name__ == "__main__":
    g = tf.Graph()
    with g.as_default():
        with g.device('/cpu:0'):
            global_step = tf.Variable(0, trainable=False)
            lr = tf.train.exponential_decay(init_lr, global_step, 10000, 0.999)
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

            opt = tf.train.RMSPropOptimizer(lr, epsilon=0.1, decay=0.99)

            # init shared and target network
            shared_net = QAgent(opt)
            target_net = QAgent(opt)
            # init thread-specific network
            thread_networks = [QAgent(opt) for _ in xrange(num_thread)]

            grad_placeholder = [(tf.placeholder(tf.float32, var.get_shape()), var) for _, var in shared_net.grad_op]
            apply_grad_op = opt.apply_gradients(grad_placeholder, global_step=global_step)

            saver = tf.train.Saver(tf.all_variables())
            sess.run(tf.initialize_all_variables())

            # build copy op
            copy_to_target = build_copy_op(shared_net, target_net)
            copy_to_threads = [build_copy_op(shared_net, thread_networks[k]) for k in xrange(num_thread)]

            threads = []
            envs = [gym.make(cp) for _ in xrange(num_thread)]

            coord = tf.train.Coordinator()

            for k in xrange(num_thread):
                threads.append(tf.train.LooperThread(coord, None, target=train_loop, args=(
                    grad_placeholder,
                    apply_grad_op,
                    envs[k],
                    k,
                    sess,
                    target_net,
                    thread_networks[k],
                    saver,
                    copy_to_target,
                    copy_to_threads[k])))

            start_time = time.time()
            for t in threads:
                t.start()

            for t in threads:
                t.join()

            sess.run(copy_to_target)

            print "duration:", time.time() - start_time
            saver.save(sess, 'checkpoint/model.ckpt', global_step=T)
