""" Initializations for linear Gaussian controllers. """
import copy
import numpy as np
import scipy as sp

from gps.algorithm.policy.lin_gauss_policy import LinearGaussianPolicy
from gps.algorithm.policy.config import INIT_LG

from gps.agent.lto.lto_world import LTOWorld
from gps.proto.gps_pb2 import PAST_GRADS, CUR_GRAD

def init_lto_controller(hyperparams, agent):
    
    config = copy.deepcopy(INIT_LG)
    config.update(hyperparams)

    dX, dU = config['dX'], config['dU']
    T = config['T']
    cur_cond_idx = config['cur_cond_idx']
    history_len = agent.history_len
    fcn = agent.fcns[cur_cond_idx]
    # Create new world to avoiding changing the state of the original world
    world = LTOWorld(fcn['fcn_obj'], fcn['dim'], fcn['init_loc'], history_len)
    
    # Compute initial state. 
    world.reset_world()
    world.run()
    x0 = agent.get_vectorized_state(world.get_state(), cur_cond_idx)
    
    best_momentum = None
    best_learning_rate = None
    min_obj_val = float('Inf')
    
    if config['verbose']:
        print("Finding Initial Linear Gaussian Controller")
    
    for i in range(config['all_possible_momentum_params'].shape[0]):
        cur_momentum = config['all_possible_momentum_params'][i]
        for j in range(config['all_possible_learning_rates'].shape[0]):
            cur_learning_rate = config['all_possible_learning_rates'][j]
            
            cur_Kt = np.zeros((dU, dX)) # K matrix for a single time step. 
            # Equivalent to Kt[:,sensor_start_idx:sensor_end_idx] = np.eye(dU)
            agent.pack_data_x(cur_Kt, np.eye(dU), data_types=[CUR_GRAD])
            # Oldest gradients come first
            agent.pack_data_x(cur_Kt, np.tile(np.eye(dU),(1,history_len)) * (cur_momentum ** np.ravel(np.tile(np.arange(history_len,0,-1)[:,None],(1,dU))))[None,:], data_types=[PAST_GRADS])
            cur_Kt = -cur_learning_rate*cur_Kt
            
            cur_kt = np.dot(cur_Kt, x0)
            
            cur_policy = LinearGaussianPolicy(cur_Kt[None,:,:], cur_kt[None,:], np.zeros((1,dU,dU)), np.zeros((1,dU,dU)), np.zeros((1,dU,dU)))
            
            world.reset_world()
            world.run()
            for t in range(T):
                X_t = agent.get_vectorized_state(world.get_state(), cur_cond_idx)
                U_t = cur_policy.act(X_t, None, 0, np.zeros((dU,)))
                world.run_next(U_t)
            fcn['fcn_obj'].new_sample(batch_size="all")
            cur_obj_val = fcn['fcn_obj'].evaluate(world.cur_loc)
            if config['verbose']:
                print("Learning Rate: %.4f, Momentum: %.4f, Final Objective Value: %.4f" % (cur_learning_rate,cur_momentum,cur_obj_val))
            if cur_obj_val < min_obj_val:
                min_obj_val = cur_obj_val
                best_momentum = cur_momentum
                best_learning_rate = cur_learning_rate
    
    if config['verbose']:
        print("")
        print("Best Final Objective Value: %.4f" % (min_obj_val))
        print("Best Momentum: %.4f" % (best_momentum))
        print("Best Learning Rate: %.4f" % (best_learning_rate))
        print("------------------------------------------------------")
            
    Kt = np.zeros((dU, dX)) # K matrix for a single time step. 
    # Equivalent to Kt[:,sensor_start_idx:sensor_end_idx] = np.eye(dU)
    agent.pack_data_x(Kt, np.eye(dU), data_types=[CUR_GRAD])
    # Oldest gradients come first
    agent.pack_data_x(Kt, np.tile(np.eye(dU),(1,history_len)) * (best_momentum ** np.ravel(np.tile(np.arange(history_len,0,-1)[:,None],(1,dU))))[None,:], data_types=[PAST_GRADS])
    Kt = -best_learning_rate*Kt
    
    kt = np.dot(Kt, x0)
    
    K = np.tile(Kt[None,:,:], (T, 1, 1))     # Controller gains matrix.
    k = np.tile(kt[None,:], (T, 1))
    PSig = np.tile((config['init_var']*np.eye(dU))[None,:,:], (T, 1, 1))
    cholPSig = np.tile((np.sqrt(config['init_var'])*np.eye(dU))[None,:,:], (T, 1, 1))
    invPSig = np.tile(((1./config['init_var'])*np.eye(dU))[None,:,:], (T, 1, 1))
    
    return LinearGaussianPolicy(K, k, PSig, cholPSig, invPSig)

