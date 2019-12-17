import numpy as np

class Display(object):
    
    def __init__(self, hyperparams):
        self._hyperparams = hyperparams
        self._log_filename = self._hyperparams['log_filename']
        self._first_update = True
    
    def _output_column_titles(self, algorithm, policy_titles=False):
        """
        Setup iteration data column titles: iteration, average cost, and for
        each condition the mean cost over samples, step size, linear Guassian
        controller entropies, and initial/final KL divergences for BADMM.
        """
        condition_titles = '%3s | %8s %12s' % ('', '', '')
        itr_data_fields  = '%3s | %8s %12s' % ('itr', 'avg_cost', 'avg_pol_cost')
        for m in range(algorithm.M):
            condition_titles += ' | %8s %9s %-7d' % ('', 'condition', m)
            itr_data_fields  += ' | %8s %8s %8s' % ('  cost  ', '  step  ', 'entropy ')
            condition_titles += ' %8s %8s %8s' % ('', '', '')
            itr_data_fields  += ' %8s %8s %8s' % ('pol_cost', 'kl_div_i', 'kl_div_f')
        self.append_output_text(condition_titles)
        self.append_output_text(itr_data_fields)

    def _update_iteration_data(self, itr, algorithm, costs, pol_sample_lists):
        """
        Update iteration data information: iteration, average cost, and for
        each condition the mean cost over samples, step size, linear Guassian
        controller entropies, and initial/final KL divergences for BADMM.
        """
        avg_cost = np.mean(costs)
        if pol_sample_lists is not None:
            pol_costs = [np.mean([np.sum(algorithm.cost[m].eval(pol_sample_lists[m][i],True)[0]) for i in range(len(pol_sample_lists[m]))])
                     for m in range(algorithm.M)]
            itr_data = '%3d | %8.2f %12.2f' % (itr, avg_cost, np.mean(pol_costs))
        else:
            pol_costs = None
            itr_data = '%3d | %8.2f' % (itr, avg_cost)
        for m in range(algorithm.M):
            cost = costs[m]
            step = algorithm.prev[m].step_mult * algorithm.base_kl_step
            entropy = 2*np.sum(np.log(np.diagonal(algorithm.prev[m].traj_distr.chol_pol_covar,
                    axis1=1, axis2=2)))
            itr_data += ' | %8.2f %8.4f %8.2f' % (cost, step, entropy)
            kl_div_i = algorithm.cur[m].pol_info.init_kl.mean()
            kl_div_f = algorithm.cur[m].pol_info.prev_kl.mean()
            itr_data += ' %8.2f %8.2f %8.2f' % (pol_costs[m], kl_div_i, kl_div_f)
        self.append_output_text(itr_data)
        return pol_costs
    
    def update(self, itr, algorithm, agent, traj_sample_lists, pol_sample_lists):
        
        if self._first_update:
            self._output_column_titles(algorithm)
            self._first_update = False

        costs = [np.mean(np.sum(algorithm.prev[m].cs, axis=1)) for m in range(algorithm.M)]
        pol_costs = self._update_iteration_data(itr, algorithm, costs, pol_sample_lists)
        
        return costs, pol_costs

    def append_output_text(self, text):
        with open(self._log_filename, 'a') as f:
            f.write(text + '\n')
        print(text)
