import numpy as np

from collections import namedtuple


Batch = namedtuple('Batch', 'states, actions')


class Net(object):
    def __init__(self, state_space, act_space, name, sess, trainable_vars=None):
        self._state_space = state_space
        self._act_space = act_space
        self._name = name
        self._trainable_vars = trainable_vars
        self._global_scope = None

        self.sess = sess

    @property
    def trainable_vars(self):
        return self._trainable_vars

    # @property
    # def variable_placeholders(self):
    #     return self._var_phs

    def _construct(self, **kwarg):
        raise NotImplementedError

    # def _setup_optimization(self, grads, optimizer):
    #     new_vars = [var + var_ph for var, var_ph in zip(self.trainable_vars, self.variable_placeholders)]
    #     return optimizer.apply_gradients(zip(grads, new_vars))

    def train_step(self, **kwargs):
        pass

    def train(self, **kwargs):
        raise NotImplementedError

    def build_tilde(self, **kwargs):
        raise NotImplementedError

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


class LoopBuffer(object):
    def __init__(self, capacity):
        self._capacity = capacity
        self._flag = -1
        self._size = 0

    def __len__(self):
        return self._size

    @property
    def index(self):
        return (self._flag + 1) % self._capacity

    def append(self, **kwargs):
        self._flag = (self._flag + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size):
        raise NotImplementedError

    def clear(self):
        self._flag = -1
        self._size = 0


class Memory(LoopBuffer):
    def __init__(self, capacity, state_space):
        super(Memory, self).__init__(capacity)

        self._state = np.zeros((capacity,) + state_space, dtype=np.float32)
        self._act = np.zeros((capacity,), dtype=np.int32)

    def append(self, state, action):
        cur = self.index

        sn = self._state.shape[1] - state.shape[0]

        self._state[cur] = np.pad(state, (0, sn), mode='constant')
        self._act[cur] = action

        super(Memory, self).append()

    def sample(self, batch_size):
        if batch_size > len(self):
            return None

        idx = np.random.choice(len(self), batch_size)

        return Batch(self._state[idx], self._act[idx])

    def priority_sample(self, batch_size):
        """Linear sample

        :param batch_size: int, batch size
        :return: an instance of Batch
        """
        if batch_size > len(self):
            return None

        priority = np.zeros(self._capacity)
        priority[self.index:] = np.arange(1, len(self) - self.index)
        priority[:self.index] = np.arange(len(self) - self.index, len(self))

        priority /= np.sum(priority)

        idx = np.random.choice(len(self), batch_size, p=priority)

        return Batch(self._state[idx], self._act[idx])
