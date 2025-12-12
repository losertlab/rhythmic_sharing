import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from keras.layers import RepeatVector
from keras.callbacks import EarlyStopping
import keras
from keras import layers
from filterpy.kalman import unscented_transform, JulierSigmaPoints
from .utils import nearestPD
import tempfile
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


class BDM:
    '''
    Bidirectional dynamic model
    :param u_length: the length of input sequence for covariate encoder
    :param x_length: the number of time points for stacking sensor measurements
    '''

    def __init__(self, x_length, u_length):
        self.xl = x_length
        self.ul = u_length

    def prepare_data(self, df_data, signal_variables, control_variables, dict_continuous_variables, dict_discrete_variables, label=None, freq=1, normalization='standard'):
        """
        :param df: data frame containing signals and controls
        :param signal_variables:  list of columns corresponding to signals
        :param control_variables:  list of columns corresponding to control variables
        :param dict_discrete_variables:  dictionary of discrete variables; {name: unique_values}
        :param dict_continuous_variables:  dictionary of continuous variables; {name: {'min': min_value, 'max': max_value, 'mean': mean_value, 'std': std_value}}}
        :param freq: sampling frequency
        :param normalization: how continuous variables should be normalized
        :param label: column name of the labels; note there should be only one label column, with 0 indicating normal samples, 1 indicating abnormal samples
        :return: previous, current, and next signal sequences, and current controls sequence, and labels
        """
        
        continuous_variables = sorted(list(dict_continuous_variables.keys()))
        discrete_variables = sorted(list(dict_discrete_variables.keys()))
        df = df_data.copy()
        assert set(signal_variables + control_variables + discrete_variables + continuous_variables).issubset(set(df.columns)), "variables should correspond to column names!"
        assert set(signal_variables).intersection(set(discrete_variables)) == set(), "signals variables should all be continuous!"

        # normalization #
        if normalization is not None:
            assert normalization in ['min-max', 'standard'], 'only supports min-max and standarization transformation!'
            for v in sorted(list(set(continuous_variables))):
                if normalization == 'min-max':
                    v_min = dict_continuous_variables[v]['min']
                    v_max = dict_continuous_variables[v]['max']
                    df[v] = (df[v] - v_min)/float(v_max - v_min + 1e-8)
                if normalization == 'standard':
                    v_mean = dict_continuous_variables[v]['mean']
                    v_std = dict_continuous_variables[v]['std']
                    df[v] = (df[v] - v_mean)/float(v_std + 1e-8)

        df_signals = pd.DataFrame()
        df_controls = pd.DataFrame()

        for variable in sorted(list(set(signal_variables + control_variables))):
            if variable in signal_variables:
                df_signals[variable] = df[variable].copy()
            if variable in control_variables:
                if variable not in discrete_variables:
                    df_controls[variable] = df[variable].copy()
                else:
                    for value in dict_discrete_variables[variable]:
                        df_controls[str(variable) + ' = ' + str(value)] = np.multiply(df[variable] == value, 1)

        # current sequence of signals #
        _x_prev, _x_curr, _x_next, _u_curr = [], [], [], []
        for i in range(self.xl):
            _df = df_signals.shift(-i)
            _df.columns = ['xc_' + name + ' + ' + str(i) for name in df_signals.columns]
            _x_curr.append(_df)

            if label is not None:    # if any of the sequence observations is anomaly, declare as anomaly #
                df[label] = np.maximum(df_data[label], df_data[label].shift(-i).fillna(0))
        df_xcurr = pd.concat(_x_curr, axis=1)

        # next sequence of signals #
        for j in range(self.xl, (2 * self.xl)):
            _df = df_signals.shift(-j)
            _df.columns = ['xn_' + name + ' + ' + str(j) for name in df_signals.columns]
            _x_next.append(_df)
        df_xnext = pd.concat(_x_next, axis=1)

        # previous sequence of signals #
        for k in reversed(range(1, self.xl + 1)):
            _df = df_signals.shift(k)
            _df.columns = ['xp_' + name + ' - ' + str(k) for name in df_signals.columns]
            _x_prev.append(_df)
        df_xprev = pd.concat(_x_prev, axis=1)

        # current controls #
        for i in range((self.xl - self.ul), self.xl):
            # print(i)
            _df = df_controls.shift(-i)
            _df.columns = ['uf_' + name + (' - ' if i < 0 else ' + ') + str(abs(i)) for name in df_controls.columns]
            _u_curr.append(_df)
        df_ucurr = pd.concat(_u_curr, axis=1)

        df_full = pd.concat([df_xprev, df_xcurr, df_xnext, df_ucurr], axis=1)
        if label is not None:
            df_full[label] = df[label]
        if freq > 1:
            df_full = df_full.iloc[::freq, :]
        df_full = df_full.dropna()
        # df_full = df_full.reset_index(drop=True)

        x_prev = df_full.loc[:, df_xprev.columns].values
        x_curr = df_full.loc[:, df_xcurr.columns].values
        x_next = df_full.loc[:, df_xnext.columns].values
        u_curr = df_full.loc[:, df_ucurr.columns].values
        u_curr = np.reshape(u_curr, (len(u_curr), self.ul, -1))

        if label is None:
            return x_prev, x_curr, x_next, u_curr, None
        else:
            return x_prev, x_curr, x_next, u_curr, df_full[label].values


    def _make_network(self, x_dim, u_dim, s_dim, hid_dim, e_nlayers, d_nlayerss, f_nlayers, s_activation):

        s = keras.Input(shape=(s_dim,))
        xseq = keras.Input(shape=(self.xl, x_dim))
        xseq_prev = keras.Input(shape=(self.xl, x_dim))
        xseq_curr = keras.Input(shape=(self.xl, x_dim))
        xseq_next = keras.Input(shape=(self.xl, x_dim))
        u = keras.Input(shape=(self.ul, u_dim))

        # encoder #
        if e_nlayers == 1:
            e_out = layers.LSTM(s_dim, return_sequences=False, activation=s_activation)(xseq)
        else:
            e_out = layers.LSTM(hid_dim, return_sequences=True)(xseq)
            for _ in range(e_nlayers - 1):
                e_out = layers.LSTM(hid_dim, return_sequences=True)(e_out)
            e_out = layers.LSTM(s_dim, return_sequences=False, activation=s_activation)(e_out)

        e_net = keras.Model(inputs=xseq, outputs=e_out)
        s_prev = e_net(xseq_prev)
        s_curr = e_net(xseq_curr)
        s_next = e_net(xseq_next)

        # decoder #
        d_out = RepeatVector(self.xl)(s)
        for _ in range(d_nlayerss - 1):
            d_out = layers.LSTM(hid_dim, return_sequences=True)(d_out)
        d_out = layers.LSTM(hid_dim, return_sequences=True)(d_out)

        d_out = layers.TimeDistributed(layers.Dense(x_dim))(d_out)
        d_out = layers.Flatten()(d_out)
        d_net = keras.Model(inputs=s, outputs=d_out)

        # forward/backward transition #
        u_bidirection = layers.Bidirectional(layers.LSTM(hid_dim, return_sequences=True))(u)
        u_forward, u_backward = layers.Bidirectional(layers.LSTM(s_dim, return_sequences=False, activation=s_activation), merge_mode=None)(u_bidirection)
        f = layers.Dense(hid_dim, activation='relu')(s)
        for _ in range(f_nlayers-1):
            f = layers.Dense(hid_dim, activation='relu')(f)
        f = layers.Dense(s_dim, activation=s_activation)(f)
        f_forward = (f + u_forward)/2
        f_backward = (f + u_backward)/2

        f_net = keras.Model(inputs=[u, s], outputs=f_forward)
        b_net = keras.Model(inputs=[u, s], outputs=f_backward)
        s_next_hat = f_net([u, s_curr])
        s_prev_hat = b_net([u, s_curr])

        s1_ = s_prev - s_prev_hat
        s2_ = s_next - s_next_hat
        s_ = s_curr

        xhat_prev = d_net(s_prev_hat)
        xhat_next = d_net(s_next_hat)
        xhat_curr = d_net(s_curr)

        model = keras.Model([xseq_prev, xseq_curr, xseq_next, u], [xhat_prev, xhat_curr, xhat_next, s1_, s2_, s_])
        return model, e_net, d_net, f_net, b_net


    def train(self, x, s_dim=4, hid_dim=4, e_nlayers=1, d_nlayerss=1, f_nlayers=1, s_activation='tanh',
              optimizer='adam', batch_size=64, epochs=100, validation_split=0.1, verbose=1, best_model=True):

        """
        :param x: [x_prev, x_curr, x_next, u]
        :return: encoder, decoder, forward/backward transition fucntions
        """

        n = len(x[0])
        x_dim = int(x[0].shape[1]/self.xl)
        u_dim = x[3].shape[2]
        y = x[:3]
        x[0] = x[0].reshape(n, self.xl, x_dim)
        x[1] = x[1].reshape(n, self.xl, x_dim)
        x[2] = x[2].reshape(n, self.xl, x_dim)

        # keras.backend.clear_session()
        model, e_net, d_net, f_net, b_net = self._make_network(x_dim, u_dim, s_dim, hid_dim, e_nlayers, d_nlayerss, f_nlayers, s_activation)
        s_ = np.zeros((n, s_dim))
        model.compile(loss=['mse', 'mse', 'mse', 'mse', 'mse', 'mse'], loss_weights=[1, 1, 1, 0.1, 0.1, 0.1])

        if best_model:
            early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=epochs, restore_best_weights=True)
            model.fit(x, y + [s_] * 3, batch_size=batch_size, epochs=epochs, validation_split=validation_split, verbose=verbose, callbacks=[early_stopping])
        else:
            model.fit(x, y + [s_] * 3, batch_size=batch_size, epochs=epochs, validation_split=validation_split, verbose=verbose)

        e_net.compile(optimizer=optimizer, loss='mse')
        self.e_net = e_net
        d_net.compile(optimizer=optimizer, loss='mse')
        self.d_net = d_net
        f_net.compile(optimizer=optimizer, loss='mse')
        self.f_net = f_net
        b_net.compile(optimizer=optimizer, loss='mse')
        self.b_net = b_net
        self.model = model

        return self


    def estimate_system(self, x):
        """
        x: [x_prev, x_curr, x_next, u_curr]
        return: reconstruction error, forward/backward transition error, mean and covariance of states
        """

        x_dim = int(x[0].shape[1]/self.xl)
        x_prev = x[0].reshape(x[0].shape[0], self.xl, x_dim)
        x_curr = x[1].reshape(x[1].shape[0], self.xl, x_dim)
        x_next = x[2].reshape(x[2].shape[0], self.xl, x_dim)
        u_curr = x[3]

        s_prev = self.e_net.predict(x_prev)
        s_curr = self.e_net.predict(x_curr)
        s_next = self.e_net.predict(x_next)

        x_curr_pred = self.d_net.predict(s_curr)
        s_next_pred = self.f_net.predict([u_curr, s_curr])
        s_prev_pred = self.b_net.predict([u_curr, s_curr])

        self.Qf = np.cov(np.transpose(s_next_pred - s_next))
        self.Qb = np.cov(np.transpose(s_prev_pred - s_prev))

        self.R = np.cov(np.transpose(x_curr_pred - x[1]))
        self.Rf = np.cov(np.transpose(self.d_net.predict(s_next_pred) - x[2]))

        return self


    def _filter(self, x_t, u_t, direction):

        '''
        :param x_t: signals
        :param u_t: controls
        return: estimate of x (mean and inverse covariance), estiamte of state (mean and covariance)
        '''
        if direction == 'forward':
            trans_fun = self.f_net.predict
            Q = self.Qf
        elif direction == 'backward':
            trans_fun = self.b_net.predict
            Q = self.Qb
        else:
            raise('direction must be forward or backward !')

        points = JulierSigmaPoints(n=len(self.s), kappa=3 - len(self.s))
        sigmas = points.sigma_points(self.s, nearestPD(self.P))

        sigmas_trans = trans_fun([np.array([u_t] * len(sigmas)), sigmas])
        sigmas_d = self.d_net.predict(sigmas_trans)
        s_hat, P_hat = unscented_transform(sigmas_trans, points.Wm, points.Wc, Q)
        x_mu, x_cov = unscented_transform(sigmas_d, points.Wm, points.Wc, self.R)

        try:
            x_inv_cov = np.linalg.inv(x_cov)
        except:
            x_inv_cov = np.linalg.pinv(x_cov)
        KG = np.dot(np.sum([points.Wc[i] * np.outer(sigmas_trans[i] - s_hat, sigmas_d[i] - x_mu) for i in range(len(sigmas))], 0), x_inv_cov)
        self.s = s_hat + np.dot(KG, x_t.flatten() - x_mu.flatten())
        self.P = P_hat - np.dot(KG, x_cov).dot(np.transpose(KG))

        return x_mu, x_inv_cov, self.s, self.P



    def filter(self, x, u, smoothing=False, n_lag=None):
        """
        :param x: sequence of signals
        :param u: sequence of controls
        :return: forward/backward anomaly scores, filtering/smoothing state estimates
        """
        n = x.shape[0]
        x_dim = int(x.shape[1]/self.xl)
        x = x.reshape(n, self.xl, x_dim)
        self.s = self.e_net.predict(np.array([x[0]]))[0]
        self.P = np.diag([1e-6] * len(self.s))

        s_mu_forward, s_cov_forward, s_mu_backward, s_cov_backward, scores_forward, scores_backward = [], [], [], [], [], []

        for t in range(1, len(x)):
            print('forward: ', t, ' out of ', len(x))
            u_t = u[t - 1, :, :]
            x_t = x[t, :, :]
            x_mu, x_inv_cov, s_mu, s_cov = self._filter(x_t, u_t, direction='forward')
            forward_score = mahalanobis(x_t.flatten(), x_mu, x_inv_cov)
            scores_forward.append(forward_score)
            s_mu_forward.append(s_mu)
            s_cov_forward.append(s_cov)

        if not smoothing:
            return np.array(scores_forward), np.array(s_mu_forward)

        else:
            for t in range(len(x) - 2, -1, -1):
                print('backward: ', t, ' out of ', len(x) - 1)
                u_t = u[t + 1, :, :]
                x_t = x[t, :]

                if (n_lag is not None) and (t % n_lag == 0):
                    self.s = s_mu_forward[t]   # corresponds to u_t
                    self.P = s_cov_forward[t]

                x_mu, x_inv_cov, s_mu, s_cov = self._filter(x_t, u_t, direction='backward')
                backward_score = mahalanobis(x_t.flatten(), x_mu, x_inv_cov)
                scores_backward.append(backward_score)
                s_mu_backward.append(s_mu)
                s_cov_backward.append(s_cov)

            s_mu_forward = s_mu_forward[:-1]; scores_forward = scores_forward[:-1]
            s_mu_backward = s_mu_backward[:-1]; scores_backward = scores_backward[:-1]
            s_mu_backward.reverse(); scores_backward.reverse()

            return np.array(scores_forward), np.array(scores_backward), np.array(s_mu_forward), np.array(s_mu_backward)


    def test(self, x, u):
        x_dim = int(x.shape[1] / self.xl)
        x = x.reshape(x.shape[0], self.xl, x_dim)
        s = self.e_net.predict(x)
        s_next_pred = self.f_net.predict([u, s])
        x_hat = self.d_net.predict(s_next_pred)
        scores = np.array([mahalanobis(x[t + 1].flatten(), x_hat[t], np.linalg.inv(self.Rf)) for t in range(len(x) - 1)])
        return scores


    def save_model(self, path):
        if path is None:
            path = tempfile.gettempdir()

        self.model.save(path + '/model.h5', save_format="tf")
        self.e_net.save(path + '/e_net.h5', save_format="tf")
        self.d_net.save(path + '/d_net.h5', save_format="tf")
        self.f_net.save(path + '/f_net.h5', save_format="tf")
        self.b_net.save(path + '/b_net.h5', save_format="tf")

    def load_model(self, path):
        if path is None:
            path = tempfile.gettempdir()

        self.model = keras.models.load_model(path + '/model.h5')
        self.e_net = keras.models.load_model(path + '/e_net.h5')
        self.d_net = keras.models.load_model(path + '/d_net.h5')
        self.f_net = keras.models.load_model(path + '/f_net.h5')
        self.b_net = keras.models.load_model(path + '/b_net.h5')

        return self