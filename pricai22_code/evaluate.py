import sys
import os

import pickle

import joblib

from MBRL.logger import Logger as Log
Log.VERBOSE = True

import MBRL.evaluation as evaluation
from MBRL.plotting import *

def sort_by_config(results, configs, key):
    vals = np.array([cfg[key] for cfg in configs])
    I_vals = np.argsort(vals)

    for k in results['train'].keys():
        results['train'][k] = results['train'][k][I_vals,]
        results['valid'][k] = results['valid'][k][I_vals,]

        if k in results['test']:
            results['test'][k] = results['test'][k][I_vals,]

    configs_sorted = []
    for i in I_vals:
        configs_sorted.append(configs[i])

    return results, configs_sorted

def load_config(config_file):
    with open(config_file, 'r') as f:
        cfg = [l.split(': ') for l in f.read().split('\n') if ': ' in l]
        cfg = dict([(kv[0], kv[1]) for kv in cfg])
    return cfg

def evaluate(config_file, overwrite=False, filters=None):

    if not os.path.isfile(config_file):
        raise Exception('Could not find config file at path: %s' % config_file)

    cfg = load_config(config_file)
    output_dir = cfg['outdir']

    if not os.path.isdir(output_dir):
        raise Exception('Could not find output at path: %s' % output_dir)

    data_train = cfg['datadir']+'/'+cfg['dataform']
    data_test = cfg['datadir']+'/'+cfg['data_test']
    binary = False
    if cfg['loss'] == 'log':
        binary = True

    early_stop_criterion = EARLYSTOP_CRITERION
    if early_stop_criterion not in ['purt_rmse_fact']:
        epsilonlist = [0]
    else:
        epsilonlist = [0.1]
    for epsilon in epsilonlist:
        eval_path = '%s/evaluation.npz' % output_dir
        if overwrite or (not os.path.isfile(eval_path)):
            eval_results, configs = evaluation.evaluate(output_dir,
                                    data_path_train=data_train,
                                    data_path_test=data_test,
                                    binary=binary, epsilon=epsilon)
            # Save evaluation
            pickle.dump((eval_results, configs), open(eval_path, "wb"))
        else:
            if Log.VERBOSE:
                print('Loading evaluation results from %s...' % eval_path)
            # Load evaluation
            eval_results, configs = pickle.load(open(eval_path, "rb"))

        if binary:
            plot_evaluation_bin(eval_results, configs, output_dir, data_train, data_test, filters, epsilon, early_stop_criterion)
        else:
            plot_evaluation_cont(eval_results, configs, output_dir, data_train, data_test, filters, epsilon, early_stop_criterion)

def summary(config_file, exp_list):
    cfg = load_config(config_file)
    outdir = cfg['outdir']
    npzfile = outdir+'result'
    npzfile_test = outdir+'result.test'
    all_losses = []
    all_preds_train = []
    all_preds_test = []
    all_tpre_train = []
    all_tpre_test = []
    all_valid = []

    for i_exp in exp_list:
        losses, preds_train, preds_test, t_pre_train, t_pre_test, I_valid = joblib.load(outdir + '/each_result/exp_' + str(i_exp))
        ''' Collect all reps '''
        all_preds_train.append(preds_train)
        all_preds_test.append(preds_test)
        all_tpre_train.append(t_pre_train)
        all_tpre_test.append(t_pre_test)
        all_losses.append(losses)
        all_valid.append(I_valid)
    ''' Fix shape for output (n_units, dim, n_reps, n_outputs) '''
    out_preds_train = np.swapaxes(np.swapaxes(all_preds_train,1,3),0,2)
    out_tpre_train = np.swapaxes(np.swapaxes(all_tpre_train, 1, 3), 0, 2)
    out_preds_test = np.swapaxes(np.swapaxes(all_preds_test,1,3),0,2)
    out_tpre_test = np.swapaxes(np.swapaxes(all_tpre_test, 1, 3), 0, 2)
    # print(all_losses)
    out_losses = np.swapaxes(np.swapaxes(all_losses,0,2),0,1)

    ''' Store predictions '''


    ''' Compute weights if doing variable selection '''
    # if FLAGS.varsel:
    #     if i_exp == 1:
    #         all_weights = sess.run(ABNet.weights_in[0])
    #         all_beta = sess.run(ABNet.weights_pred)
    #     else:
    #         all_weights = np.dstack((all_weights, sess.run(ABNet.weights_in[0])))
    #         all_beta = np.dstack((all_beta, sess.run(ABNet.weights_pred)))

    ''' Save results and predictions '''

    np.savez(npzfile, pred=out_preds_train, tpre=out_tpre_train, loss=out_losses, val=np.array(all_valid))
    np.savez(npzfile_test, pred=out_preds_test, tpre=out_tpre_test)


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print('Usage: python evaluate.py <config_file> <overwrite (default 0)> <filters (optional)>')
    # else:
    #     config_file = sys.argv[1]
    #
    #     overwrite = False
    #     if len(sys.argv)>2 and sys.argv[2] == '1':
    #         overwrite = True
    #
    #     filters = None
    #     if len(sys.argv)>3:
    #         filters = eval(sys.argv[3])
    #
    #     evaluate(config_file, overwrite, filters=filters)
    EARLYSTOP_CRITERION = 'purt_rmse_fact'
    exp_list = list(range(1, 1001))
    config_file = r'./results\ihdp/config.txt'
    summary(config_file, exp_list)
    overwrite = 1
    evaluate(config_file, overwrite, filters=None)