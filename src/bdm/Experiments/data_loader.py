import numpy as np
import pandas as pd
import zipfile
import os

def simulation(noise=True):

    np.random.seed(1234)
    N = 10000
    u = np.tile(np.repeat(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 100), 10)
    s = np.sin(np.arange(N)) + np.sin(u)
    s = s + np.random.normal(0, 0.5 if noise else 0.1, N)
    x = s
    x = x + np.random.normal(0, 1 if noise else 0.1, N)
    train_df = pd.DataFrame({'x': x, 'u': u})

    N = 10000
    labels = np.tile(np.concatenate([[0] * 900, [1] * 100]), 10)
    u = np.tile(np.repeat(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 100), 10)
    s = np.sin(np.arange(N)) + np.sin(u)
    test_true = pd.DataFrame({'x': s, 'u': u})

    s = s + (labels == 0) * np.random.normal(0, 0.5 if noise else 0.1, N)
    s = s + (labels == 1) * np.random.normal(0, 1 if noise else 0.1, N)
    x = s
    x = x + (labels == 0) * np.random.normal(0, 1 if noise else 0.1, N)
    x = x + (labels == 1) * np.random.normal(0, 2 if noise else 0.1, N)
    test_df = pd.DataFrame({'x': x, 'u': u, 'label': labels})

    signal_variables = ['x']
    control_variables = ['x', 'u']
    dict_continuous_variable = {'x': {'mean': train_df['x'].mean(), 'std': train_df['x'].std(), 'min': train_df['x'].min(), 'max': train_df['x'].max()}}
    dict_discrete_variables = {'u': train_df['u'].unique()}

    return train_df, test_df, test_true, signal_variables, control_variables, dict_continuous_variable, dict_discrete_variables


def swat_data():

    # See "Time Series Anomaly Detection for Cyber-physical Systems via Neural System Identification and Bayesian Filtering"
    # and https://github.com/NSIBF/ for datasets and processing details

    z_tr = zipfile.ZipFile('bdm/Experiments/data/SWAT/SWaT_train.zip', "r")
    f_tr = z_tr.open(z_tr.namelist()[0])
    train_df = pd.read_csv(f_tr)
    f_tr.close()
    z_tr.close()

    z_tr = zipfile.ZipFile('bdm/Experiments/data/SWAT/SWaT_test.zip', "r")
    f_tr = z_tr.open(z_tr.namelist()[0])
    test_df = pd.read_csv(f_tr)
    f_tr.close()
    z_tr.close()

    train_df = train_df.dropna()
    test_df.loc[test_df['label'] >= 1, 'label'] = 1
    test_df = test_df.dropna()


    signal_variables = ['FIT101', 'LIT101', 'FIT201', 'DPIT301', 'FIT301', 'LIT301', 'FIT401', 'LIT401',  'FIT501','FIT502',  'FIT601']
    control_variables = ['MV101', 'P101', 'P102', 'MV201', 'P202',
                   'P203', 'P204', 'P205', 'P206', 'MV301', 'MV302',
                   'MV303', 'MV304', 'P301', 'P302', 'P401', 'P402',
                   'P403', 'P404', 'UV401', 'P501', 'P502', 'P601',
                   'P602', 'P603']

    dict_continuous_variables, dict_discrete_variables = {}, {}

    for signal in signal_variables:
        dict_continuous_variables.update({signal: {'mean': train_df[signal].mean(), 'std': train_df[signal].std(),
                                                   'max': train_df[signal].max(), 'min': train_df[signal].min()}})

    for control in control_variables.copy():
        unique_vals = np.unique(train_df[control])
        dict_discrete_variables.update({control: unique_vals})

    return train_df, test_df, signal_variables, signal_variables + control_variables, dict_continuous_variables, dict_discrete_variables



def wadi_data():

    # See "Time Series Anomaly Detection for Cyber-physical Systems via Neural System Identification and Bayesian Filtering"
    # and https://github.com/NSIBF/ for datasets and processing details

    z_tr = zipfile.ZipFile('_Experiments/datasets/WADI/WADI_train.zip', "r")
    f_tr = z_tr.open(z_tr.namelist()[0])
    train_df = pd.read_csv(f_tr)
    f_tr.close()
    z_tr.close()

    z_tr = zipfile.ZipFile('_Experiments/datasets/WADI/WADI_test.zip', "r")
    f_tr = z_tr.open(z_tr.namelist()[0])
    test_df = pd.read_csv(f_tr)
    f_tr.close()
    z_tr.close()

    test_df.loc[test_df['label'] >= 1, 'label'] = 1


    signal_variables = ['1_AIT_002_PV', '1_FIT_001_PV', '1_LT_001_PV', '2_DPIT_001_PV',
               '2_FIC_101_CO', '2_FIC_101_PV', '2_FIC_101_SP', '2_FIC_201_CO',
               '2_FIC_201_PV', '2_FIC_201_SP', '2_FIC_301_CO', '2_FIC_301_PV',
               '2_FIC_301_SP', '2_FIC_401_CO', '2_FIC_401_PV', '2_FIC_401_SP',
               '2_FIC_501_CO', '2_FIC_501_PV', '2_FIC_501_SP', '2_FIC_601_CO',
               '2_FIC_601_PV', '2_FIC_601_SP', '2_FIT_001_PV', '2_FIT_002_PV',
               '2_FIT_003_PV', '2_FQ_101_PV', '2_FQ_201_PV', '2_FQ_301_PV', '2_FQ_401_PV',
               '2_FQ_501_PV', '2_FQ_601_PV', '2_LT_002_PV', '2_MCV_101_CO',
               '2_MCV_201_CO', '2_MCV_301_CO', '2_MCV_401_CO', '2_MCV_501_CO', '2_MCV_601_CO',
               '2_P_003_SPEED', '2_P_004_SPEED', '2_PIC_003_CO', '2_PIC_003_PV',
               '2_PIT_002_PV', '2_PIT_003_PV',  '2A_AIT_002_PV',
               '3_AIT_001_PV', '3_AIT_002_PV', '3_AIT_003_PV', '3_AIT_004_PV',
               '3_FIT_001_PV', '3_LT_001_PV', 'LEAK_DIFF_PRESSURE', 'TOTAL_CONS_REQUIRED_FLOW']

    control_variables = ['1_MV_001_STATUS', '1_MV_004_STATUS', '1_P_001_STATUS', '1_P_003_STATUS',
                 '1_P_005_STATUS', '2_LS_101_AH', '2_LS_101_AL', '2_LS_201_AH', '2_LS_201_AL',
                 '2_LS_301_AH', '2_LS_301_AL', '2_LS_401_AH', '2_LS_401_AL', '2_LS_501_AH',
                 '2_LS_501_AL', '2_LS_601_AH', '2_LS_601_AL', '2_MV_003_STATUS', '2_MV_006_STATUS',
                 '2_MV_101_STATUS', '2_MV_201_STATUS', '2_MV_301_STATUS', '2_MV_401_STATUS',
                 '2_MV_501_STATUS', '2_MV_601_STATUS', '2_P_003_STATUS']


    dict_continuous_variables, dict_discrete_variables = {}, {}

    for signal in signal_variables:
        dict_continuous_variables.update({signal: {'mean': train_df[signal].mean(), 'std': train_df[signal].std(),
                                                   'max': train_df[signal].max(), 'min': train_df[signal].min()}})

    for control in control_variables.copy():
        unique_vals = np.unique(train_df[control])
        if len(unique_vals) == 1:
            control_variables.remove(control)
        else:
            dict_discrete_variables.update({control: unique_vals})

    return train_df, test_df, signal_variables, signal_variables + control_variables, dict_continuous_variables, dict_discrete_variables