from keras.src.utils import set_random_seed
set_random_seed(1)

from keras.src.backend import set_floatx
set_floatx("float64")

from keras.src.models import Model as keras_model

from tensorflow import GradientTape, constant, reduce_mean, stop_gradient, expand_dims
from tensorflow.random import set_seed as tf_set_seed
tf_set_seed(1)


class Q_approximator(keras_model):

    def __init__(self, **kwargs):
        super(Q_approximator, self).__init__(**kwargs)

    def train_step(self, data):
        x_data, y_data = data

        x_now, u_prev, observed_taken_actions, explored_taken_actions,  explored_next_states, explored_termination, V_min = x_data
        v_observed, r_explored = y_data

        gamma = constant(1.0, shape = (1,), dtype = v_observed.dtype)

        loss_list = []
        with GradientTape() as tape:

            with GradientTape() as taylor_grad_tape:
                taylor_grad_tape.watch(observed_taken_actions)
                v_obs_pred = self([x_now, u_prev, observed_taken_actions], training=True)

            advantage_gradients = taylor_grad_tape.batch_jacobian(v_obs_pred, observed_taken_actions)
            advantage_gradients = advantage_gradients[:, 1:2, :]


            v_next_explored_pred = stop_gradient(self([explored_next_states, explored_taken_actions, explored_taken_actions], training=True)[:, 0:1])
            q_explored_pred = self([x_now, u_prev, explored_taken_actions], training=True)

            v_loss =  (v_obs_pred[:, 0:1] - v_observed)  ** 2
            loss_list.append(v_loss)

            a_loss = v_obs_pred[:, 1:2] ** 2
            loss_list.append(a_loss)

            advantage = r_explored + gamma * v_next_explored_pred * (1.0 - explored_termination) + explored_termination * V_min - v_observed
            q_loss = (advantage - q_explored_pred[:, 1:2]) ** 2
            loss_list.append(q_loss)

            exploration = expand_dims(explored_taken_actions - observed_taken_actions, axis = -1)
            taylor_loss = advantage - (advantage_gradients @ exploration)[:, :, 0]
            taylor_loss = taylor_loss ** 2
            loss_list.append(taylor_loss)
            
            weighting = constant(1.0, shape = (1,), dtype = v_loss.dtype)
            v_loss *= weighting

            loss = reduce_mean(loss_list, axis = 0, keepdims=False)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        for m in self.metrics:
            if m.name == "loss":
                m.update_state(loss)
            else:
                m.update_state(v_observed, v_obs_pred[:, 0:1])
                m.update_state(0, v_obs_pred[:, 1:2])


        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x_data, y_data = data

        x_now, u_prev, observed_taken_actions, explored_taken_actions,  explored_next_states, explored_termination, V_min = x_data
        v_observed, r_explored = y_data

        gamma = constant(1.0, shape = (1,), dtype = v_observed.dtype)
        
        loss_list = []

        with GradientTape() as taylor_grad_tape:
            taylor_grad_tape.watch(observed_taken_actions)
            v_obs_pred = self([x_now, u_prev, observed_taken_actions], training=True)

        advantage_gradients = taylor_grad_tape.batch_jacobian(v_obs_pred, observed_taken_actions)
        advantage_gradients = advantage_gradients[:, 1:2, :]


        v_next_explored_pred = self([explored_next_states, explored_taken_actions, explored_taken_actions], training=True)[:, 0:1]
        q_explored_pred = self([x_now, u_prev, explored_taken_actions], training=True)

        v_loss =  (v_obs_pred[:, 0:1] - v_observed)  ** 2
        loss_list.append(v_loss)

        a_loss = v_obs_pred[:, 1:2] ** 2
        loss_list.append(a_loss)

        advantage = r_explored + gamma * v_next_explored_pred * (1.0 - explored_termination) + explored_termination * V_min - v_observed
        q_loss = (advantage - q_explored_pred[:, 1:2]) ** 2
        loss_list.append(q_loss)

        exploration = expand_dims(explored_taken_actions - observed_taken_actions, axis = -1)
        taylor_loss = advantage - (advantage_gradients @ exploration)[:, :, 0]
        taylor_loss = taylor_loss ** 2
        loss_list.append(taylor_loss)
        
        weighting = constant(1.0, shape = (1,), dtype = v_loss.dtype)
        v_loss *= weighting

        loss = reduce_mean(loss_list, axis = 0, keepdims=False)

        for m in self.metrics:
            if m.name == "loss":
                m.update_state(loss)
            else:
                m.update_state(v_observed, v_obs_pred[:, 0:1])
                m.update_state(0, v_obs_pred[:, 1:2])
        return {m.name: m.result() for m in self.metrics}