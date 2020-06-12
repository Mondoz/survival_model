import numpy as np
import pandas as pd
import os
import SimpleITK as sitk
import time
import argparse
import math
from lifelines.utils import concordance_index


def metrics_ci(time, y_pred, event):
    """Compute the concordance-index value.

    Parameters
    ----------
    label_true : dict
        Status and Time of patients in survival analyze,
        example like as {'e': event, 't': time}.
    y_pred : np.array
        Proportional risk.

    Returns
    -------
    float
        Concordance index.
    """
    hr_pred =  y_pred
    ci = concordance_index(time, hr_pred, event)
    return ci

if __name__ == '__main__':
    path = './record.csv'
    df = pd.read_csv(path,sep = ',')
    import pdb
    pdb.set_trace()
    data_array = df.values
    radio = metrics_ci(data_array[:,2], data_array[:,0], data_array[:,3])
    report = metrics_ci(data_array[:,2], data_array[:,1], data_array[:,3])
    Norm = np.zeros((data_array.shape[0], 2))
    Norm[:, 0] = (data_array[:,0] - min(data_array[:,0]))/(max(data_array[:,0]) - min(data_array[:,0]))
    Norm[:,1] = (data_array[:,1] - min(data_array[:,1]))/(max(data_array[:,1]) - min(data_array[:,1]))
    Mean = np.mean(Norm, axis=1)
    Max = np.max(Norm, axis=1)
    Multi = Norm[:, 0] * Norm[:, 1]
    Mean_CI = metrics_ci(data_array[:,2], Mean, data_array[:,3])
    Max_CI = metrics_ci(data_array[:,2], Max, data_array[:,3])
    Multi_CI = metrics_ci(data_array[:,2], Multi, data_array[:,3])
    print('radio_CI: %.4f, report_CI: %.4f, Mean_CI: %.4f, max_CI: %.4f, Multi_CI: %.4f' % (radio, report, Mean_CI, Max_CI, Multi_CI))