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


def train_loop(grad_placeholder, apply_grad_op, env, id, sess, shared_net, target_net, saver, target_copy):
    time.sleep(id)
    global Tmax, T
    t = 0
    sess.run(target_copy)
    gamma = 0.99
    mean_reward = None
    env = MyEnv(env, 4)

    init_eps = 1
    final_eps = sample_final_epsilon()
    epsilon = init_eps

    grads = [shared_net.grad_op[k][0] for k in xrange(len(grad_placeholder))]
    grads_data = None
    while T < Tmax:
        total_r = 0
        observation = env.reset()
        while True:
            tq_value = shared_net.q_value.eval(session=sess, feed_dict={shared_net.s: [observation]})
            action = get_action(tq_value, eps=epsilon)
            at = np.zeros([dim_output])
            at[action] = 1

            # Scale down epsilon
            if epsilon > final_eps:
                epsilon -= (init_eps - final_eps) / anneal_step

            new_observation, reward, done, info = env.step(action)

            total_r += reward

            if done:
                y = reward
            else:
                y = (reward + gamma * np.amax(
                    target_net.q_value.eval(session=sess, feed_dict={target_net.s: [new_observation]})))

            # accumulate gradients
            if grads_data is None:
                grads_data = np.array(sess.run(grads, feed_dict={
                    shared_net.s: [observation],
                    shared_net.y: [y],
                    shared_net.actions: [at],
                }))
            else:
                grads_data += np.array(sess.run(grads, feed_dict={
                    shared_net.s: [observation],
                    shared_net.y: [y],
                    shared_net.actions: [at],
                }))

            observation = new_observation
            t += 1
            T += 1

            if T % 40000 == 0:
                sess.run(target_copy)
                print "updating target network parameters!!!!"
            if t % 5 == 0 or done:
                if grads_data is not None:
                    fd = {}
                    for i, _ in enumerate(grad_placeholder):
                        fd[grad_placeholder[i][0]] = grads_data[i]
                    sess.run(apply_grad_op, feed_dict=fd)
                    grads_data = None
            if T > Tmax or done:
                mean_reward = (0.9 * mean_reward + 0.1 * total_r) if mean_reward is not None else total_r
                print "step: %d, thread id: %2d, mean reward: %.4f, eps: %.7f -> %f" % (
                    T, id, mean_reward, epsilon, final_eps)
                break
            if t % 10000 == 0 and id == 0:
                saver.save(sess, 'checkpoint/model.ckpt', global_step=T)
                print "model saved!!!"
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

            # apply gradients to shared net
            grad_placeholder = [(tf.placeholder(tf.float32, var.get_shape()), var) for _, var in shared_net.grad_op]
            apply_grad_op = opt.apply_gradients(grad_placeholder, global_step=global_step)

            saver = tf.train.Saver(tf.all_variables())
            sess.run(tf.initialize_all_variables())

            # build copy op
            copy_to_target = build_copy_op(shared_net, target_net)

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
                    shared_net,
                    target_net,
                    saver,
                    copy_to_target,)))

            start_time = time.time()
            for t in threads:
                t.start()

            for t in threads:
                t.join()

            sess.run(copy_to_target)

            print "duration:", time.time() - start_time
            saver.save(sess, 'checkpoint/model.ckpt', global_step=T)
