import tensorflow as tf
import numpy as np
import sys, os
import getopt
import random
import datetime
import traceback

import MBRL.MBRL_net as MBRLnet
from MBRL.util import *
from sklearn import metrics
import joblib

''' Define parameter flags '''
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('loss', 'l2', """Which loss function to use (l1/l2/log)""")
tf.app.flags.DEFINE_integer('n_in', 4, """Number of representation layers. """)
tf.app.flags.DEFINE_integer('n_out', 3, """Number of regression layers. """)
tf.app.flags.DEFINE_integer('n_dc', 3, """Number of discriminator layers. """)
tf.app.flags.DEFINE_float('p_alpha', 1, """distance regularization. """)
tf.app.flags.DEFINE_float('p_lambda', 1e-4, """Weight decay regularization parameter. """)
tf.app.flags.DEFINE_float('p_epsilon_t', 0.01, """regularization for t. """)
tf.app.flags.DEFINE_float('p_epsilon_y', 0.01, """regularization for y. """)
tf.app.flags.DEFINE_integer('rep_weight_decay', 0, """Whether to penalize representation layers with weight decay""")
tf.app.flags.DEFINE_float('dropout_in', 1, """Input layers dropout keep rate. """)
tf.app.flags.DEFINE_float('dropout_out', 1, """Output layers dropout keep rate. """)
tf.app.flags.DEFINE_string('nonlin', 'elu', """Kind of non-linearity. Default relu. """)
tf.app.flags.DEFINE_float('lrate', 1e-3, """Learning rate. """)
tf.app.flags.DEFINE_float('decay', 0.3, """RMSProp decay. """)
tf.app.flags.DEFINE_integer('batch_size', 100, """Batch size. """)
tf.app.flags.DEFINE_integer('dim_in', 200, """Pre-representation layer dimensions. """)
tf.app.flags.DEFINE_integer('dim_out', 100, """Post-representation layer dimensions. """)
tf.app.flags.DEFINE_integer('dim_d', 200, """Discriminator layer dimensions. """)
tf.app.flags.DEFINE_integer('batch_norm', 0, """Whether to use batch normalization. """)
tf.app.flags.DEFINE_string('normalization', 'divide', """How to normalize representation (after batch norm). none/bn_fixed/divide/project """)
tf.app.flags.DEFINE_integer('experiments_start', 1, """Number of experiments. """)
tf.app.flags.DEFINE_integer('experiments_end', 1000, """Number of experiments. """)
tf.app.flags.DEFINE_integer('iterations', 1000, """Number of iterations. """)
tf.app.flags.DEFINE_float('weight_init', 0.1, """Weight initialization scale. """)
tf.app.flags.DEFINE_float('lrate_decay', 0.97, """Decay of learning rate every 100 iterations """)
tf.app.flags.DEFINE_integer('varsel', 0, """Whether the first layer performs variable selection. """)
tf.app.flags.DEFINE_string('outdir', './results/ihdp/', """Output directory. """)
tf.app.flags.DEFINE_string('datadir', './dataset_1-1000/', """Data directory. """)
tf.app.flags.DEFINE_string('dataform', 'ihdp_npci_1-1000.train.npz', """Training data filename form. """)
tf.app.flags.DEFINE_string('data_test', 'ihdp_npci_1-1000.test.npz', """Test data filename form. """)
tf.app.flags.DEFINE_integer('sparse', 0, """Whether data is stored in sparse format (.x, .y). """)
tf.app.flags.DEFINE_integer('seed', 1, """Seed. """)
tf.app.flags.DEFINE_integer('repetitions', 1, """Repetitions with different seed.""")
tf.app.flags.DEFINE_integer('use_p_correction', 0, """Whether to use population size p(t) in mmd/disc/wass.""")
tf.app.flags.DEFINE_integer('wass_iterations', 10, """Number of iterations in Wasserstein computation. """)
tf.app.flags.DEFINE_float('wass_lambda', 10, """Wasserstein lambda. """)
tf.app.flags.DEFINE_integer('wass_bpt', 1, """Backprop through T matrix? """)
tf.app.flags.DEFINE_string('optimizer', 'Adam', """Which optimizer to use. (RMSProp/Adagrad/GradientDescent/Adam)""")
tf.app.flags.DEFINE_integer('output_csv',0,"""Whether to save a CSV file with the results""")
tf.app.flags.DEFINE_integer('output_delay', 1, """Number of iterations between log/loss outputs. """)
tf.app.flags.DEFINE_integer('pred_output_delay', 1, """Number of iterations between prediction outputs. (-1 gives no intermediate output). """)
tf.app.flags.DEFINE_integer('save_rep', 1, """Save representations after training. """)
tf.app.flags.DEFINE_float('val_part', 0.3, """Validation part. """)
tf.app.flags.DEFINE_boolean('split_output', 1, """Whether to split output layers between treated and control. """)
tf.app.flags.DEFINE_integer('reweight_sample', 1, """Whether to reweight sample for prediction loss with average treatment probability. """)
tf.app.flags.DEFINE_integer('reweight_sample_t', 0, """Whether to reweight sample for adversarial loss with average treatment probability. """)
tf.app.flags.DEFINE_integer('NUM_ITERATIONS_PER_DECAY', 100, """iter """)
tf.app.flags.DEFINE_integer('t_pre_smooth', 1, """ smooth t? """)
tf.app.flags.DEFINE_integer('y_pre_smooth', 1, """ smooth y if y is binary? """)
if FLAGS.sparse:
    import scipy.sparse as sparse
config = tf.ConfigProto()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config.gpu_options.per_process_gpu_memory_fraction = 0.1
tf.Session(config=config)
NUM_ITERATIONS_PER_DECAY = FLAGS.NUM_ITERATIONS_PER_DECAY


def train(MBRLNet, sess, train_step, train_discriminator_step, train_encoder_step, train_epsilon_t_step, train_epsilon_y_step, D, I_valid, D_test, logfile, i_exp):

    ''' Train/validation split '''
    n = D['x'].shape[0]
    I = range(n); I_train = list(set(I)-set(I_valid))
    n_train = len(I_train)

    ''' Compute treatment probability'''
    p_treated = np.mean(D['t'][I_train,:])

    z_norm = np.random.normal(0.,1.,(1,FLAGS.dim_in))

    ''' Set up loss feed_dicts'''
    dict_factual = {MBRLNet.x: D['x'][I_train,:], MBRLNet.t: D['t'][I_train,:], MBRLNet.y_: D['yf'][I_train,:], \
      MBRLNet.do_in: 1.0, MBRLNet.do_out: 1.0, MBRLNet.r_lambda: FLAGS.p_lambda, MBRLNet.p_t: p_treated, MBRLNet.z_norm: z_norm}

    if FLAGS.val_part > 0:
        dict_valid = {MBRLNet.x: D['x'][I_valid,:], MBRLNet.t: D['t'][I_valid,:], MBRLNet.y_: D['yf'][I_valid,:], \
          MBRLNet.do_in: 1.0, MBRLNet.do_out: 1.0, MBRLNet.r_lambda: FLAGS.p_lambda, MBRLNet.p_t: p_treated, MBRLNet.z_norm: z_norm}

    if D['HAVE_TRUTH']:
        dict_cfactual = {MBRLNet.x: D['x'][I_train,:], MBRLNet.t: 1-D['t'][I_train,:], MBRLNet.y_: D['ycf'][I_train,:], \
          MBRLNet.do_in: 1.0, MBRLNet.do_out: 1.0, MBRLNet.z_norm: z_norm}

    ''' Initialize TensorFlow variables '''
    sess.run(tf.global_variables_initializer())

    ''' Set up for storing predictions '''
    preds_train = []
    preds_test = []
    tpre_train = []
    tpre_test = []
    ''' Compute losses '''
    losses = []
    obj_loss, f_error, discriminator_loss, rep_loss = \
    sess.run([MBRLNet.tot_loss, MBRLNet.pred_loss, MBRLNet.discriminator_loss, MBRLNet.rep_loss],\
      feed_dict=dict_factual)

    cf_error = np.nan
    if D['HAVE_TRUTH']:
        cf_error = sess.run(MBRLNet.pred_loss, feed_dict=dict_cfactual)

    valid_obj = np.nan; valid_f_error = np.nan;
    if FLAGS.val_part > 0:
        valid_obj, valid_f_error, valid_dc, valid_rep_r= \
        sess.run([MBRLNet.tot_loss, MBRLNet.pred_loss, MBRLNet.discriminator_loss, MBRLNet.rep_loss],\
          feed_dict=dict_valid)

    losses.append([obj_loss, f_error, cf_error, discriminator_loss, rep_loss,\
        valid_f_error, valid_dc, valid_rep_r, valid_obj])

    objnan = False

    reps = []
    reps_test = []

    ''' Train for multiple iterations '''
    for i in range(FLAGS.iterations):
        I = list(range(0, n_train))
        np.random.shuffle(I)
        for i_batch in range(n_train // FLAGS.batch_size):
            if i_batch < n_train // FLAGS.batch_size - 1:
                I_b = I[i_batch * FLAGS.batch_size:(i_batch+1) * FLAGS.batch_size]
            else:
                I_b = I[i_batch * FLAGS.batch_size:]
            x_batch = D['x'][I_train,:][I_b,:]
            t_batch = D['t'][I_train,:][I_b]
            y_batch = D['yf'][I_train,:][I_b]

            z_norm_batch = np.random.normal(0.,1.,(1,FLAGS.dim_in))
            ''' Do one step of gradient descent '''
            if not objnan:
                #train discriminator
                sess.run(train_discriminator_step, feed_dict={MBRLNet.x: x_batch, MBRLNet.t: t_batch, \
                    MBRLNet.do_in: FLAGS.dropout_in, MBRLNet.do_out: FLAGS.dropout_out, MBRLNet.z_norm: z_norm_batch, MBRLNet.p_t: p_treated})
                #train encoder
                # for sub_enc in range(0,2):
                sess.run(train_encoder_step, feed_dict={MBRLNet.x: x_batch, MBRLNet.t: t_batch, \
                    MBRLNet.do_in: FLAGS.dropout_in, MBRLNet.do_out: FLAGS.dropout_out, MBRLNet.z_norm: z_norm_batch, MBRLNet.p_t: p_treated})

                sess.run(train_step, feed_dict={MBRLNet.x: x_batch, MBRLNet.t: t_batch, \
                    MBRLNet.y_: y_batch, MBRLNet.do_in: FLAGS.dropout_in, MBRLNet.do_out: FLAGS.dropout_out, \
                    MBRLNet.r_lambda: FLAGS.p_lambda, MBRLNet.p_t: p_treated})

                sess.run(train_epsilon_y_step, feed_dict={MBRLNet.x: x_batch, MBRLNet.t: t_batch, \
                                                MBRLNet.y_: y_batch, MBRLNet.do_in: FLAGS.dropout_in,
                                                MBRLNet.do_out: FLAGS.dropout_out, \
                                                MBRLNet.r_lambda: FLAGS.p_lambda, MBRLNet.p_t: p_treated})

                sess.run(train_epsilon_t_step, feed_dict={MBRLNet.x: x_batch, MBRLNet.t: t_batch, \
                                                              MBRLNet.do_in: FLAGS.dropout_in,
                                                              MBRLNet.do_out: FLAGS.dropout_out,
                                                              MBRLNet.z_norm: z_norm_batch, MBRLNet.p_t: p_treated})
            ''' Project variable selection weights '''
            if FLAGS.varsel:
                wip = simplex_project(sess.run(MBRLNet.weights_in[0]), 1)
                sess.run(MBRLNet.projection, feed_dict={MBRLNet.w_proj: wip})

        ''' Compute loss every N iterations '''
        if i % FLAGS.output_delay == 0 or i==FLAGS.iterations-1:
            obj_loss,f_error, discriminator_loss,epsilon_t, epsilon_y, rep_loss= \
            sess.run([MBRLNet.tot_loss, MBRLNet.pred_loss, MBRLNet.discriminator_loss, MBRLNet.epsilon_t, MBRLNet.epsilon_y, MBRLNet.rep_loss],
                feed_dict=dict_factual)

            cf_error = np.nan
            if D['HAVE_TRUTH']:
                cf_error = sess.run(MBRLNet.pred_loss, feed_dict=dict_cfactual)

            valid_obj = np.nan; valid_f_error = np.nan;
            if FLAGS.val_part > 0:
                valid_obj, valid_f_error, valid_dc, valid_rep_r = \
                sess.run([MBRLNet.tot_loss, MBRLNet.pred_loss, MBRLNet.discriminator_loss, MBRLNet.rep_loss], \
                    feed_dict=dict_valid)

            losses.append([obj_loss, f_error, cf_error, discriminator_loss, rep_loss,\
                valid_f_error, valid_dc, valid_rep_r, valid_obj])
            loss_str = str(i) + '\tObj: %.3f,\tF: %.3f,\tCf: %.3f, \tdc_loss: %.3f, \trep_loss: %.3f, \tepsilon_t: %.3f, \tepsilon_y: %.3f, \tVal: %.3f, \tValdc: %.3f, \tValrep: %.3f, \tValObj: %.2f' \
                        % (obj_loss, f_error, cf_error, discriminator_loss, rep_loss, epsilon_t, epsilon_y, valid_f_error, valid_dc, valid_rep_r, valid_obj)

            if FLAGS.loss == 'log':
                y_pred = sess.run(MBRLNet.output, feed_dict={MBRLNet.x: x_batch, \
                    MBRLNet.t: t_batch, MBRLNet.do_in: 1.0, MBRLNet.do_out: 1.0})

                fpr, tpr, thresholds = metrics.roc_curve(y_batch, y_pred)
                auc = metrics.auc(fpr, tpr)

                loss_str += ',\tAuc_batch: %.2f' % auc


            if np.isnan(obj_loss):
                log(logfile,'Experiment %d: Objective is NaN. Skipping.' % i_exp)
                objnan = True

        ''' Compute predictions every M iterations '''
        if (FLAGS.pred_output_delay > 0 and i % FLAGS.pred_output_delay == 0) or i==FLAGS.iterations-1:

            y_pred_f = sess.run(MBRLNet.output, feed_dict={MBRLNet.x: D['x'], \
                MBRLNet.t: D['t'], MBRLNet.do_in: 1.0, MBRLNet.do_out: 1.0})
            y_pred_cf = sess.run(MBRLNet.output, feed_dict={MBRLNet.x: D['x'], \
                MBRLNet.t: 1-D['t'], MBRLNet.do_in: 1.0, MBRLNet.do_out: 1.0})
            preds_train.append(np.concatenate((y_pred_f, y_pred_cf),axis=1))
            t_pre_train = sess.run(MBRLNet.t_pre, feed_dict={MBRLNet.x: D['x'], \
                                                          MBRLNet.t: D['t'], MBRLNet.do_in: 1.0, MBRLNet.do_out: 1.0})
            tpre_train.append(t_pre_train)
            if FLAGS.loss == 'log' and D['HAVE_TRUTH']:
                fpr, tpr, thresholds = metrics.roc_curve(np.concatenate((D['yf'], D['ycf']),axis=0), \
                    np.concatenate((y_pred_f, y_pred_cf),axis=0))
                auc = metrics.auc(fpr, tpr)
                loss_str += ',\tAuc_train: %.2f' % auc

            if D_test is not None:
                y_pred_f_test = sess.run(MBRLNet.output, feed_dict={MBRLNet.x: D_test['x'], \
                    MBRLNet.t: D_test['t'], MBRLNet.do_in: 1.0, MBRLNet.do_out: 1.0})
                y_pred_cf_test = sess.run(MBRLNet.output, feed_dict={MBRLNet.x: D_test['x'], \
                    MBRLNet.t: 1-D_test['t'], MBRLNet.do_in: 1.0, MBRLNet.do_out: 1.0})
                preds_test.append(np.concatenate((y_pred_f_test, y_pred_cf_test),axis=1))
                t_pre_test = sess.run(MBRLNet.t_pre, feed_dict={MBRLNet.x: D_test['x'], \
                    MBRLNet.t: D_test['t'], MBRLNet.do_in: 1.0, MBRLNet.do_out: 1.0})
                tpre_test.append(t_pre_test)
                if FLAGS.loss == 'log' and D['HAVE_TRUTH']:
                    fpr, tpr, thresholds = metrics.roc_curve(np.concatenate((D_test['yf'], D_test['ycf']),axis=0), \
                        np.concatenate((y_pred_f_test, y_pred_cf_test),axis=0))
                    auc = metrics.auc(fpr, tpr)
                    loss_str += ',\tAuc_test: %.2f' % auc

            if FLAGS.save_rep and i_exp == 1:
                reps_i = sess.run([MBRLNet.h_rep], feed_dict={MBRLNet.x: D['x'], \
                    MBRLNet.do_in: 1.0, MBRLNet.do_out: 0.0})
                reps.append(reps_i)

                if D_test is not None:
                    reps_test_i = sess.run([MBRLNet.h_rep], feed_dict={MBRLNet.x: D_test['x'], \
                        MBRLNet.do_in: 1.0, MBRLNet.do_out: 0.0})
                    reps_test.append(reps_test_i)

            log(logfile, loss_str)

    return losses, preds_train, preds_test, tpre_train, tpre_test, reps, reps_test

def run(outdir):
    """ Runs an experiment and stores result in outdir """

    ''' Set up paths and start log '''
    npzfile = outdir+'result'
    npzfile_test = outdir+'result.test'
    logfile = outdir+'log.txt'
    f = open(logfile,'w')
    f.close()
    dataform = FLAGS.datadir + FLAGS.dataform

    has_test = False
    if not FLAGS.data_test == '': # if test set supplied
        has_test = True
        dataform_test = FLAGS.datadir + FLAGS.data_test

    ''' Set random seeds '''
    random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    ''' Save parameters '''
    save_config(outdir+'config.txt')
    ''' Load Data '''
    npz_input = False
    if dataform[-3:] == 'npz':
        npz_input = True
    if npz_input:
        datapath = dataform
        if has_test:
            datapath_test = dataform_test
    else:
        datapath = dataform % 1
        if has_test:
            datapath_test = dataform_test % 1

    log(logfile,     'Training data: ' + datapath)
    if has_test:
        log(logfile, 'Test data:     ' + datapath_test)
    D = load_data(datapath)
    D_test = None
    if has_test:
        D_test = load_data(datapath_test)

    log(logfile, 'Loaded data with shape [%d,%d]' % (D['n'], D['dim']))

    ''' Start Session '''
    sess = tf.Session()

    ''' Initialize input placeholders '''
    x = tf.placeholder("float", shape=[None, D['dim']], name='x') # Features
    t = tf.placeholder("float", shape=[None, 1], name='t')   # Treatent
    y_ = tf.placeholder("float", shape=[None, 1], name='y_')  # Outcome

    znorm = tf.placeholder("float", shape=[None, FLAGS.dim_in], name='z_norm')

    ''' Parameter placeholders '''
    r_lambda = tf.placeholder("float", name='r_lambda')
    do_in = tf.placeholder("float", name='dropout_in')
    do_out = tf.placeholder("float", name='dropout_out')
    p = tf.placeholder("float", name='p_treated')
    

    ''' Define model graph '''
    log(logfile, 'Defining graph...\n')
    dims = [D['dim'], FLAGS.dim_in, FLAGS.dim_out, FLAGS.dim_d]
    MBRLNet = MBRLnet.MBRL_net(x, t, y_, p, znorm, FLAGS, r_lambda, do_in, do_out, dims)

    lr_ad = 5e-5
    ''' Set up optimizer '''
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(FLAGS.lrate, global_step, \
        NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay, staircase=True)

    counter_enc = tf.Variable(0, trainable=False)
    lr_enc = tf.train.exponential_decay(lr_ad, counter_enc, \
        NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay, staircase=True)

    counter_dc = tf.Variable(0, trainable=False)
    lr_dc = tf.train.exponential_decay(lr_ad, counter_dc, \
        NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay, staircase=True)

    counter_epsilon_t = tf.Variable(0, trainable=False)
    lr_epsilon_t = tf.train.exponential_decay(lr_ad, counter_epsilon_t, \
        NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay, staircase=True)

    counter_epsilon_y = tf.Variable(0, trainable=False)
    lr_epsilon_y = tf.train.exponential_decay(lr_ad, counter_epsilon_y, \
        NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay, staircase=True)

    if FLAGS.optimizer == 'Adam':
        opt = tf.train.AdamOptimizer(lr)
        opt_enc = tf.train.AdamOptimizer(
            learning_rate=lr_enc, 
            beta1=0.5, 
            beta2=0.9)
        opt_dc = tf.train.AdamOptimizer(
            learning_rate=lr_dc, 
            beta1=0.5, 
            beta2=0.9)
        opt_epsilon_t = tf.train.AdamOptimizer(
            learning_rate=lr_epsilon_t,
            beta1=0.5,
            beta2=0.9)
        opt_epsilon_y = tf.train.AdamOptimizer(
            learning_rate=lr_epsilon_y,
            beta1=0.5,
            beta2=0.9)

    #var_scope_get
    var_epsilon_t = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='epsilon_t')
    var_epsilon_y = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='epsilon_y')
    var_enc = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    var_dc = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    var_pred = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='pred')
    var_pred.extend(var_enc)

    print("var_enc:",[v.name for v in var_enc])
    print()
    print("var_dc:",[v.name for v in var_dc])
    print()
    print("var_pred:",[v.name for v in var_pred])
    print()
    train_discriminator_step = opt_dc.minimize(MBRLNet.discriminator_loss,global_step=counter_dc,var_list=var_dc)
    train_encoder_step = opt_enc.minimize(MBRLNet.rep_loss,global_step=counter_enc,var_list=var_enc)
    train_step = opt.minimize(MBRLNet.tot_loss,global_step=global_step,var_list=var_pred)
    train_epsilon_t_step = opt_epsilon_t.minimize(MBRLNet.discriminator_loss,global_step=counter_dc,var_list=var_epsilon_t)
    train_epsilon_y_step = opt_epsilon_y.minimize(MBRLNet.tot_loss,global_step=global_step,var_list=var_epsilon_y)

    ''' Run for all repeated experiments '''
    for i_exp in range(FLAGS.experiments_start, FLAGS.experiments_end+1):

        if FLAGS.repetitions>1:
            log(logfile, 'Training on repeated initialization %d/%d...' % (i_exp, FLAGS.repetitions))
        else:
            log(logfile, 'Training on experiment %d/%d...' % (i_exp, FLAGS.experiments_end))

        ''' Load Data (if multiple repetitions, reuse first set)'''

        if i_exp==1 or FLAGS.experiments_end>1:
            D_exp_test = None
            if npz_input:
                D_exp = {}
                D_exp['x']  = D['x'][:,:,i_exp-1]
                D_exp['t']  = D['t'][:,i_exp-1:i_exp]
                D_exp['yf'] = D['yf'][:,i_exp-1:i_exp]
                if D['HAVE_TRUTH']:
                    D_exp['ycf'] = D['ycf'][:,i_exp-1:i_exp]
                else:
                    D_exp['ycf'] = None

                if has_test:
                    D_exp_test = {}
                    D_exp_test['x']  = D_test['x'][:,:,i_exp-1]
                    D_exp_test['t']  = D_test['t'][:,i_exp-1:i_exp]
                    D_exp_test['yf'] = D_test['yf'][:,i_exp-1:i_exp]
                    if D_test['HAVE_TRUTH']:
                        D_exp_test['ycf'] = D_test['ycf'][:,i_exp-1:i_exp]
                    else:
                        D_exp_test['ycf'] = None
            else:
                datapath = dataform % i_exp
                D_exp = load_data(datapath)
                if has_test:
                    datapath_test = dataform_test % i_exp
                    D_exp_test = load_data(datapath_test)

            D_exp['HAVE_TRUTH'] = D['HAVE_TRUTH']
            if has_test:
                D_exp_test['HAVE_TRUTH'] = D_test['HAVE_TRUTH']

        ''' Split into training and validation sets '''
        I_train, I_valid = validation_split(D_exp, FLAGS.val_part)

        ''' Run training loop '''
        losses, preds_train, preds_test, t_pre_train, t_pre_test, reps, reps_test = \
            train(MBRLNet, sess, train_step, train_discriminator_step, train_encoder_step, train_epsilon_t_step, train_epsilon_y_step, D_exp, I_valid, \
                D_exp_test, logfile, i_exp)
        each_result_dir = outdir + '/each_result/'
        if not os.path.isdir(each_result_dir):
            os.mkdir(each_result_dir)
        each_result_list = [losses, preds_train, preds_test, t_pre_train, t_pre_test, I_valid]
        joblib.dump(each_result_list, each_result_dir + 'exp_' + str(i_exp))

        # ''' Save representations '''
        if not os.path.isdir(outdir +'/rep'):
            os.mkdir(outdir +'/rep')
        if FLAGS.save_rep and i_exp < 3:
            np.savez(outdir + 'rep/rep_' + str(i_exp) + '.npz', rep=reps)

            if has_test:
                np.savez(outdir + 'rep/rep_test_' + str(i_exp) + '.npz', rep=reps_test)
def main(argv=None):  # pylint: disable=unused-argument
    """ Main entry point """
    if not os.path.isdir('./results'):
        os.mkdir('./results')
    outdir = FLAGS.outdir
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    try:
        run(outdir)
    except Exception as e:
        with open(outdir+'error.txt','w') as errfile:
            errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
        raise


if __name__ == '__main__':
    tf.app.run()
