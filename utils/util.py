import importlib
import os.path as osp
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.integrate import simps


def get_config(config_file):
    temp_config_name = osp.basename(config_file)
    temp_module_name = osp.splitext(temp_config_name)[0]
    config2 = importlib.import_module("configs.%s"%temp_module_name)
    importlib.reload(config2)
    cfg = config2.config
    return cfg


def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]


def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))


def AUCError(errors, failureThreshold=0.08, step=0.0001, showCurve=True):
    nErrors = len(errors)
    xAxis = list(np.arange(0., failureThreshold + step, step))
    ced = [float(np.count_nonzero([errors <= x])) / nErrors for x in xAxis]
    AUC = simps(ced, x=xAxis) / failureThreshold
    failureRate = 1. - ced[-1]
    print("AUC @ {0}: {1}".format(failureThreshold, AUC))
    print("Failure rate: {0}".format(failureRate))
    if showCurve:
        plt.plot(xAxis, ced)
        # plt.draw()
        plt.show(block=False)


def plot_multi_auc(error_list, legends, failureThreshold=0.08, step=0.0001):
    for errors in error_list:
        nErrors = len(errors)
        xAxis = list(np.arange(0., failureThreshold + step, step))
        ced = [float(np.count_nonzero([errors <= x])) / nErrors for x in xAxis]
        plt.plot(xAxis, ced)

    plt.legend(legends)
    plt.show()




