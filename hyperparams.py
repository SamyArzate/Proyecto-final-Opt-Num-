import tensorflow as tf

import os.path
from datetime import datetime
import numpy as np

import gps
from gps import __file__ as gps_filepath
from gps.agent.lto.agent_lto import AgentLTO
from gps.agent.lto.lto_world import LTOWorld
from gps.algorithm.algorithm import Algorithm
from gps.algorithm.cost.cost import Cost
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.traj_opt.traj_opt import TrajOpt
from gps.algorithm.policy_opt.policy_opt import PolicyOpt
from gps.algorithm.policy_opt.lto_model import fully_connected_tf_network
from gps.algorithm.policy.lin_gauss_init import init_lto_controller
from gps.proto.gps_pb2 import CUR_LOC, PAST_OBJ_VAL_DELTAS, PAST_GRADS, CUR_GRAD, PAST_LOC_DELTAS, ACTION
from gps.agent.lto.fcn import LogisticRegressionFcnFamily, LogisticRegressionFcn
from gps.algorithm.cost.cost_utils import RAMP_CONSTANT

try:
   import cPickle as pickle
except:
   import pickle
import copy

def gen_fcns(input_dim, num_fcns, session, num_inits_per_fcn = 1, num_points_per_class = 50):

    fcn_family = LogisticRegressionFcnFamily(input_dim, gpu_id = 0, session = session, tensor_prefix = "logistic_reg")
    
    # Dimensionality of the space over which optimization is performed
    param_dim = fcn_family.get_total_num_dim()

    fcn_objs = []

    for i in range(num_fcns):
        
        data = []
        for j in range(2):
            mu = np.random.randn(input_dim)
            sigma = np.random.randn(input_dim, input_dim)
            sigma_sq = np.dot(sigma, sigma.T)
            data.append(np.random.multivariate_normal(mu, sigma_sq, num_points_per_class))
        data = np.vstack(data)
        labels = np.vstack((np.zeros((num_points_per_class,1),dtype=np.int),np.ones((num_points_per_class,1),dtype=np.int)))
        
        fcn = LogisticRegressionFcn(fcn_family, data, labels, disable_subsampling = True)
        for j in range(num_inits_per_fcn):
            fcn_objs.append(fcn)

    init_locs = np.random.randn(param_dim,num_fcns*num_inits_per_fcn) 

    fcns = [{'fcn_obj': fcn_objs[i], 'dim': param_dim, 'init_loc': init_locs[:,i][:,None]} for i in range(num_fcns*num_inits_per_fcn)]
    
    return fcns,fcn_family

def lto_on_exit(config):
    config['agent']['fcn_family'].destroy()

session = tf.Session()
history_len = 25

num_fcns = 10 #100

input_dim = 3

cur_dir = os.path.dirname(os.path.abspath(__file__))
dataset_file = cur_dir + "/trainset.pkl"

if os.path.isfile(dataset_file):
    print("Dataset already exists. Loading from %s. " % (dataset_file))
    with open(dataset_file, "rb") as f:
        fcns,fcn_family = pickle.load(f)
    fcn_family.start_session(session)
else:
    print("Generating new dataset.")
    fcns,fcn_family = gen_fcns(input_dim, num_fcns, session)
    with open(dataset_file, "wb") as f:
        pickle.dump((fcns,fcn_family), f)
    print("Saved to %s. " % (dataset_file))
    
param_dim = fcns[0]['dim']

SENSOR_DIMS = { 
    CUR_LOC: param_dim,
    PAST_OBJ_VAL_DELTAS: history_len,
    PAST_GRADS: history_len*param_dim,
    PAST_LOC_DELTAS: history_len*param_dim,
    CUR_GRAD: param_dim, 
    ACTION: param_dim
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/lto/'


common = {
    'experiment_name': 'lto' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': num_fcns
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentLTO,
    'world' : LTOWorld,
    'substeps': 1,
    'conditions': common['conditions'],
    'dt': 0.05,
    'T': 40,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [CUR_LOC, PAST_OBJ_VAL_DELTAS, PAST_GRADS, CUR_GRAD, PAST_LOC_DELTAS],
    'obs_include': [PAST_OBJ_VAL_DELTAS, PAST_GRADS, CUR_GRAD, PAST_LOC_DELTAS],
    'history_len': history_len,
    'fcns': fcns,
    'fcn_family': fcn_family     # Only used to destroy these at the end
}

algorithm = {
    'type': Algorithm,
    'conditions': common['conditions'],
    'iterations': 10,
    'inner_iterations': 4,
    'policy_dual_rate': 0.2, 
    'init_pol_wt': 0.01, 
    'ent_reg_schedule': 0.0,
    'fixed_lg_step': 3,
    'kl_step': 0.2, 
    'min_step_mult': 0.01, 
    'max_step_mult': 10.0, 
    'sample_decrease_var': 0.05,
    'sample_increase_var': 0.1,
    'policy_sample_mode': 'replace'
}

algorithm['init_traj_distr'] = {
    'type': init_lto_controller,
    'init_var': 0.01, 
    'dt': agent['dt'],
    'T': agent['T'],
    'all_possible_momentum_params': np.array([0.82, 0.84, 0.86, 0.88, 0.9, 0.92]),
    'all_possible_learning_rates': np.array([0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6])
}

algorithm['cost'] = {
    'type': Cost,
    'ramp_option': RAMP_CONSTANT, 
    'wp_final_multiplier': 1.0, 
    'weight': 1.0,
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-3,     # Increase this if Qtt is not PD during DGD
    'clipping_thresh': None,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20, 
        'min_samples_per_cluster': 20,
        'max_samples': 20,
        'strength': 1.0         # How much weight to give to prior relative to samples
    }
}

algorithm['traj_opt'] = {
    'type': TrajOpt,
}

algorithm['policy_opt'] = {
    'type': PolicyOpt,
    'network_model': fully_connected_tf_network,
    'iterations': 20000, 
    'init_var': 0.01, 
    'batch_size': 25,
    'solver_type': 'adam',
    'lr': 0.0001, 
    'lr_policy': 'fixed',
    'momentum': 0.9,
    'weight_decay': 0.005,
    'use_gpu': 1,
    'weights_file_prefix': EXP_DIR + 'policy',
    'network_params': {
        'obs_include': agent['obs_include'],
        'sensor_dims': agent['sensor_dims'],
        'dim_hidden': [50]
    }
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20, 
    'min_samples_per_cluster': 20, 
    'max_samples': 20,
    'strength': 1.0,
    'clipping_thresh': None, 
    'init_regularization': 1e-3, 
    'subsequent_regularization': 1e-3 
}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 20,
    'common': common,
    'agent': agent,
    'algorithm': algorithm,
    'on_exit': lto_on_exit,
}
