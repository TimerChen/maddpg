import os
import os.path as osp
import tensorflow as tf



class Net(object):
    def __init__(self, state_space, act_space, name, sess, trainable_vars=None):
        pass

    def save(self, model_dir, step):
        """Save model"""
        dir_name = osp.join(model_dir, self._name)

        if not osp.exists(dir_name):
            os.makedirs(dir_name)

        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self._global_scope)
        saver = tf.train.Saver(model_vars)
        save_path = saver.save(self.sess, "{}/{}".format(dir_name, self._name), global_step=step)
        print("[*] Model saved in file: {}".format(save_path))

    def load(self, model_dir, step):
        dir_name = os.path.join(model_dir, self._name)
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self._global_scope)
        saver = tf.train.Saver(model_vars)
        file_path = os.path.join(dir_name, "{}/{}-{}".format(dir_name, self._name, step))
        saver.restore(self.sess, file_path)
