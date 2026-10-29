import numpy as np
import tensorflow as tf
np.random.seed(1234)
tf.random.set_seed(1234)
from sklearn import metrics
from Experiments.data_loader import simulation
import matplotlib.pyplot as plt
from Algorithm.BDM import BDM


################################################   Synthetic data   ############################################
train_df, test_df, _, signal_variables, control_variables, dict_continuous_variables, dict_discrete_variables = simulation(noise=True)
xl = 8
ul = xl * 2
batch_size = 64
epochs = 100
normalization = 'min-max'
bdm = BDM(x_length=xl, u_length=ul)

x_prev, x_curr, x_next, u_curr, _ = bdm.prepare_data(train_df, signal_variables=['x'], control_variables=['x', 'u'],
dict_continuous_variables=dict_continuous_variables, dict_discrete_variables=dict_discrete_variables, label=None, normalization=normalization, freq=1)
x_dim = int(x_curr.shape[1]/xl)

index = np.arange(len(x_prev))
train_pos = index[:(len(x_prev) * 3 // 4)]
val_pos = index[(len(x_prev) * 3 // 4):]

# training data
x_train_bdm = [x_prev[train_pos, :], x_curr[train_pos, :], x_next[train_pos, :], u_curr[train_pos, :]]
# validation data
x_val_bdm = [x_prev[val_pos, :], x_curr[val_pos, :], x_next[val_pos, :], u_curr[val_pos, :]]
# test data
x_prev, x_curr, x_next, u_curr, labels = bdm.prepare_data(test_df, signal_variables=['x'], control_variables=['x', 'u'],
dict_continuous_variables=dict_continuous_variables, dict_discrete_variables=dict_discrete_variables, label='label', normalization=normalization, freq=xl)
test_x_bdm = x_curr; test_u_bdm = u_curr
######################################################################################################




######################################   Training/Validation/Testing #########################################
bdm.train(x_train_bdm, s_dim=4, s_activation='tanh', batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=2)
# bdm = bdm.load_model(path=r'Experiments/results/simulation/BDM')
bdm.estimate_system(x_val_bdm)
scores_bdm = bdm.test(test_x_bdm, test_u_bdm)
scores_bdm = np.nan_to_num(scores_bdm)
auc_bdm = metrics.roc_auc_score(labels[1:], scores_bdm)
print('BDM AUC:', auc_bdm)



plt.figure()
T = np.linspace(1, len(test_x_bdm) * xl, len(test_x_bdm) - 1)
plt.plot([0, 1], [0, 1], 'k--')
fpr, tpr, thresholds = metrics.roc_curve(labels[1:], scores_bdm, pos_label=1)
plt.plot(fpr, tpr, label='Our method AUC: ' + str(round(auc_bdm, 3)))

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Synthetic data ROC curve')
plt.legend()
plt.show()