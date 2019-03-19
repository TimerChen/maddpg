import tensorflow as tf
import numpy as np
from maddpg.trainer.utils import construct_fc_weights, forward_fc
from maddpg.trainer.buffer import Memory
from maddpg.trainer.base import Net

class MAML(Net):
    def __init__(self, sess, name, obs_shape, num_action,
                 num_agents,
                 batch_size = 64, memory = None,
                 theta=0.5, max_inner_epoch=50,
                 memory_size=100,
                 useL2L=False,
                 lamb=1e-2,
                 dim_hidden = 32, num_layers = 2, lr = 0.01):
        self.theta = theta
        self.dim_hidden = [dim_hidden]*num_layers
        self.dim_input = obs_shape[0]
        self.obs_shape = obs_shape
        self.num_action = num_action
        self.num_agents = num_agents
        self.final_optimizer = None
        self.final_loss = None
        self.lr = lr
        self.useL2L = False

        self._lamb = lamb

        self._batch_size = batch_size
        self.max_inner_epoch = max_inner_epoch
        if memory is None:
            self._memory = [Memory(memory_size, obs_shape) for _ in range(num_agents)]
            self._new_memory = [Memory(memory_size, obs_shape) for _ in range(num_agents)]
        else:
            if not isinstance(memory, list) and num_agents == 1:
                self._memory = [memory]
            else:
                self._memory = memory
        self._name = name
        #print("My name is:", self._name)

        self._build()

        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    def _build(self):

        with tf.variable_scope("maml_{}".format(self._name)):
            self.scope = self._global_scope = tf.get_variable_scope().name # ?

            #place holders
            self.x_placeholder = tf.placeholder(tf.float32, shape=(self.num_agents, None,) + self.obs_shape)
            self.y_placeholder = tf.placeholder(tf.float32, shape=(self.num_agents, None, self.num_action))
            self.x2_placeholder = tf.placeholder(tf.float32, shape=(self.num_agents, None,) + self.obs_shape)
            self.y2_placeholder = tf.placeholder(tf.float32, shape=(self.num_agents, None, self.num_action))
            self.real_x_ph = [tf.placeholder(tf.float32, shape=(None,) + self.obs_shape) for _ in range(self.num_agents)]
            self.real_y_ph = [tf.placeholder(tf.float32, shape=(None, self.num_action)) for _ in range(self.num_agents)]

            print("ph_shape:", (self.num_agents, None,) + self.obs_shape, (self.num_agents, None, self.num_action))

            #Origin Model
            self.model_weights = construct_fc_weights(self.dim_input, self.dim_hidden, self.num_action)
            self.output = []
            self.loss = []
            self.final_loss = None
            self.loss2 = []
            self.future_weights = []
            self.sub_model_weights = []
            self.update_real_future = []
            self.train_real_model = []
            self.real_loss = []
            self.real_output = []
            self.tfb = []

            opt = tf.train.GradientDescentOptimizer(learning_rate = self.lr)

            self.tmp = None

            for i in range(self.num_agents):

                input = self.x_placeholder[i]
                self.tmp = input
                with tf.name_scope("model{}".format(i)):
                    output = forward_fc(input, self.model_weights, len(self.dim_hidden))

                    label = self.y_placeholder[i]
                    loss = self._build_loss(output, label)
                    self.output.append(output)
                    self.loss.append(loss)


                #Real Future Models
                with tf.name_scope("real_future{}".format(i)):
                    real_future_weights = construct_fc_weights(self.dim_input, self.dim_hidden, self.num_action)
                    for k in self.model_weights:
                        self.update_real_future.append(tf.assign(self.model_weights[k], real_future_weights[k]))

                    #Future model
                    input2 = tf.reshape(self.real_x_ph[i], shape = (-1,) + self.obs_shape)
                    output2 = forward_fc(input2, real_future_weights, len(self.dim_hidden))
                    label2 = tf.reshape(self.real_y_ph[i], shape = (-1, self.num_action))
                    loss2 = self._build_loss(output2, label2)

                    self.train_real_model.append(opt.minimize(loss2))
                    self.real_output.append(output2)
                    self.real_loss.append(loss2)

                    self.tfb.append(tf.summary.scalar('opponent_loss{}'.format(i), loss2))


                #MAML Part
                self._trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._global_scope)
                gs = opt.compute_gradients(loss, self._trainable_vars)
                future_weights = {}
                with tf.name_scope("future{}".format(i)):
                    for g, ref in gs:
                        #print(g, ref)
                        if g is None:
                            continue
                        delta = +self.theta*g
                        old_name = ref.name.split('/')[-1].split(':')[0]
                        #print("{} -> {}".format(ref.name, old_name))
                        f_ref = tf.add(ref, delta, name=old_name)
                        future_weights[old_name] = f_ref

                    #Future model
                    input2 = tf.reshape(self.x2_placeholder[i], shape = (-1,) + self.obs_shape)
                    output2 = forward_fc(input2, future_weights, len(self.dim_hidden))
                    label2 = tf.reshape(self.y2_placeholder[i], shape = (-1, self.num_action))
                    loss2 = self._build_loss(output2, label2)



                self.future_weights.append(future_weights)
                self.loss2.append(loss2)

            self.mid_loss = tf.reduce_mean(self.loss)
            self.final_loss = tf.reduce_mean(self.loss2)
            self._grads, self._train_op = self._optimize(self.final_loss)
            _, self._mid_train_op = self._optimize(self.mid_loss)


    def _build_loss(self, output, y):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output,labels=y))
        output = tf.nn.softmax(output)
        reg = self._lamb * tf.reduce_mean(tf.reduce_sum(output * tf.log(output+1e-6), axis=1))
        return loss + reg

    def _init_model(self):
        for i in self.model_weights:
            self.model_weights[i].initializer.run()

    def clear_new(self):
        for i in range(self.num_agents):
            self._new_memory[i].clear()

    def store_data(self, states, actions):
        for i, data in enumerate(zip(states, actions)):
            self._memory[i].append(data[0], data[1])
            self._new_memory[i].append(data[0], data[1])

    def pred_i(self, input, index):
        return np.argmax(self.sess.run(self.real_output[index] + self.real_loss[index],
                                feed_dict={self.real_x_ph[index]: input}))

    def prediction(self, inputs):
        return self.sess.run(self.output, feed_dict = {self.x_placeholder: inputs})

    def act(self, state, eps):
        inputs = [state] * self.num_agents
        return self.prediction(inputs)

    def evaluate(self, epoch):
        if self._batch_size >= len(self._memory[0]):
            #print(self._batch_size, len(self._memory[0]))
            return 0.0

        batch_num = (len(self._memory[0]) * 2) // self._batch_size
        loss_record, grads_record = [], []

        loss, loss_margin = float("inf"), 0.01

        # init theta
        #init_var_list = self.sess.run(self._trainable_vars)
        inner_epoch = 0
        max_inner_epoch = self.max_inner_epoch*10

        while loss > loss_margin and inner_epoch < max_inner_epoch:
            for b in range(batch_num):
                x, y = [], []
                #x2, y2 = [], []
                for i in range(self.num_agents):
                    batch = self._memory[i].sample(self._batch_size)
                    x.append(batch.states)
                    a = np.zeros((batch.actions.shape[0], self.num_action))
                    a[np.arange(batch.actions.shape[0]), batch.actions] = 1
                    y.append(a)

                _, loss = self.sess.run([self._mid_train_op, self.mid_loss], feed_dict={
                    self.x_placeholder: x, self.y_placeholder: y,
                    #self.x2_placeholder: x2, self.y2_placeholder: y2
                })
                #print("--- [opponent evaluate] batch #{:<4d} loss {:<4.3f}".format(b, loss), end="\r")
                loss_record.append(loss)

            loss = np.mean(loss_record[-batch_num:])
            print("+++ [opponent evaluate] epoch #{:<4d} inner-epoch #{:<4d} loss: {:.3f}".format(epoch, inner_epoch, loss), end="\r")

            inner_epoch += 1

        print("+++ [opponent evaluate] epoch #{:<4d} inner-epochs #{:<4d} final-loss: {:.3f}".format(epoch, inner_epoch, loss))
        mean_loss = np.mean(loss_record)
        return mean_loss

    def _optimize(self, loss, optimizer=None):
        if optimizer is None:
            #Final-Optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)

        _trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._global_scope)
        _grad_vars = optimizer.compute_gradients(loss, _trainable_vars)

        grads = [var for _, var in _grad_vars]
        train_op = optimizer.apply_gradients(_grad_vars)
        return grads, train_op

    def train_simple(self, data):
        x, y, x2, y2 = data
        return self.sess.run(self.train_op, feed_dict =
        {self.x_placeholder: x, self.y_placeholder: y,
         self.x2_placeholder: x2, self.y2_placeholder: y2})

    def update_real(self):
        self.sess.run( self.update_real_future)
        # self.train_real_model = []
        # self.real_loss)

    def train_real(self, epoch):
        _batch_size = min(self._batch_size, len(self._new_memory[0]))
        if _batch_size == 0:
            return None
        batch_num = (len(self._new_memory[0]) * 2) // _batch_size
        loss_record, grads_record = [], []
        mid_loss_record = [[] for _ in range(self.num_agents)]

        loss, loss_margin = float("inf"), 0.01

        # init theta
        init_var_list = self.sess.run(self._trainable_vars)
        inner_epoch = 0

        while np.max(loss) > loss_margin and inner_epoch < self.max_inner_epoch:
            for b in range(batch_num):
                x2, y2 = [], []
                fdict = {}
                for i in range(self.num_agents):
                    batch = self._new_memory[i].sample(_batch_size)
                    a = np.zeros((batch.actions.shape[0], self.num_action))
                    a[np.arange(batch.actions.shape[0]), batch.actions] = 1
                    fdict[self.real_x_ph[i]] = batch.states
                    fdict[self.real_y_ph[i]] = a

                res = self.sess.run(self.real_loss + self.train_real_model, feed_dict=fdict)
                loss = res[:len(self.real_loss)]
                # print("--- [opponent training] batch #{:<4d} loss {:<4.3f}".format(b, loss), end="\r")
                for i in range(self.num_agents):
                    mid_loss_record[i].append(loss[i])


            mid_loss = []
            for i in range(self.num_agents):
                mid_loss.append(np.mean(mid_loss_record[i][-batch_num:]))
            #print("--- [real training] epoch #{:<4d} inner-epoch #{:<4d} loss: {}".format(epoch, inner_epoch, mid_loss), end="\r")

            inner_epoch += 1
        #print("--- [real training] epoch #{:<4d} inner-epochs #{:<4d} final-loss: {}".format(epoch, inner_epoch, mid_loss))

        # grads_label = list(map(sum, zip(*grads_record)))

        mean_loss = [np.mean(i) for i in mid_loss]

        return mid_loss


    def train(self, epoch):
        if self._batch_size >= len(self._memory[0]):
            #print(self._batch_size, len(self._memory[0]))
            return None

        batch_num = (len(self._memory[0]) * 2) // self._batch_size
        loss_record, grads_record = [], []
        mid_loss_record = [[] for _ in range(self.num_agents)]

        loss, loss_margin = float("inf"), 0.01

        # init theta
        init_var_list = self.sess.run(self._trainable_vars)
        inner_epoch = 0

        while loss > loss_margin and inner_epoch < self.max_inner_epoch:
            for b in range(batch_num):
                x, y = [], []
                x2, y2 = [], []
                for i in range(self.num_agents):
                    batch = self._memory[i].sample(self._batch_size)
                    x.append(batch.states)
                    a = np.zeros((batch.actions.shape[0], self.num_action))
                    a[np.arange(batch.actions.shape[0]), batch.actions] = 1
                    y.append(a)

                    batch = self._memory[i].sample(self._batch_size)
                    x2.append(batch.states)
                    a = np.zeros((batch.actions.shape[0], self.num_action))
                    a[np.arange(batch.actions.shape[0]), batch.actions] = 1
                    y2.append(a)

                res = self.sess.run([self._train_op, self.final_loss, self._grads] + self.loss, feed_dict={
                    self.x_placeholder: x, self.y_placeholder: y,
                    self.x2_placeholder: x2, self.y2_placeholder: y2
                })
                _, loss, grads = res[:3]
                # print("--- [opponent training] batch #{:<4d} loss {:<4.3f}".format(b, loss), end="\r")
                for i in range(self.num_agents):
                    mid_loss_record[i].append(res[i+3])
                loss_record.append(loss)
                grads_record.append(grads)

            loss = np.mean(loss_record[-batch_num:])

            mid_loss = []
            for i in range(self.num_agents):
                mid_loss.append(np.mean(mid_loss_record[i][-batch_num:]))
            print("--- [opponent training] epoch #{:<4d} inner-epoch #{:<4d} loss: {:.3f}".format(epoch, inner_epoch, loss), end="\r")

            inner_epoch += 1


        print("--- [opponent training] epoch #{:<4d} inner-epochs #{:<4d} mid-loss: {} final-loss: {:.3f}".format(epoch, inner_epoch, mid_loss, loss))

        # grads_label = list(map(sum, zip(*grads_record)))
        new_var_list = self.sess.run(self._trainable_vars)
        grads_label = list(map(lambda v: v[1] - v[0], zip(init_var_list, new_var_list)))

        mean_loss = np.mean(loss_record)

        return mean_loss, grads_label
