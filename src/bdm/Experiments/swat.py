import numpy as np
import tensorflow as tf
np.random.seed(1234)
tf.random.set_seed(1234)
from sklearn import metrics
from Algorithm.BDM import BDM
from Algorithm.utils import f_search
from Experiments.data_loader import swat_data

################################    SWAT data  ######################################################
train_df, test_df, signal_variables, control_variables, dict_continuous_variables, dict_discrete_variables = swat_data()
xl = 16
ul = xl * 2
batch_size = 256
epochs = 100
normalization = 'min-max'
bdm = BDM(x_length=xl, u_length=ul)


# training & validation data
x_prev, x_curr, x_next, u_curr, _ = bdm.prepare_data(train_df, signal_variables=signal_variables,
        control_variables=control_variables, dict_continuous_variables=dict_continuous_variables, dict_discrete_variables=dict_discrete_variables, label=None, normalization=normalization, freq=1)
x_dim = int(x_curr.shape[1]/xl)
index = np.arange(len(x_prev))
train_pos = index[:(len(x_prev) * 3 // 4)]
val_pos = index[(len(x_prev) * 3 // 4):]

x_train_bdm = [x_prev[train_pos, :], x_curr[train_pos, :], x_next[train_pos, :], u_curr[train_pos, :]]
x_val_bdm = [x_prev[val_pos, :], x_curr[val_pos, :], x_next[val_pos, :], u_curr[val_pos, :]]
# test data
_, x_curr, x_next, u_curr, labels = bdm.prepare_data(test_df, signal_variables=signal_variables,
                                                    control_variables=control_variables, dict_continuous_variables=dict_continuous_variables,
                                                    dict_discrete_variables=dict_discrete_variables, label='label', normalization=normalization, freq=xl)
test_x_bdm = x_curr; test_u_bdm = u_curr
###################################################################################################################################






######################################   Training/Validation/Testing #########################################
# bdm.train(x_train_bdm, s_dim=4, s_activation='tanh', batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=2)
bdm = bdm.load_model(path=r'Experiments/results/SWAT/BDM')
bdm.estimate_system(x_val_bdm)
z_scores_bdm = bdm.test(test_x_bdm, test_u_bdm)
z_scores_bdm = np.nan_to_num(z_scores_bdm)
t, _ = f_search(z_scores_bdm, labels[1:])

print('BDM')
print('best-f1', t[0])
print('precision', t[1])
print('recall', t[2])
print('accuracy', (t[3] + t[4]) / (t[3] + t[4] + t[5] + t[6]))
print('TP', t[3])
print('TN', t[4])
print('FP', t[5])
print('FN', t[6])
print('AUC:', metrics.roc_auc_score(labels[1:], z_scores_bdm))