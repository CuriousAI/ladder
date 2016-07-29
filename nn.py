from collections import OrderedDict
import logging

import scipy
import numpy as np
from theano import tensor
from theano.tensor.signal.pool import pool_2d, Pool

from blocks.extensions import SimpleExtension
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          MonitoringExtension)
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.monitoring.evaluators import DatasetEvaluator
from blocks.roles import VariableRole

logger = logging.getLogger('main.nn')


class BnParamRole(VariableRole):
    pass

# Batch normalization parameters that have to be replaced when testing
BNPARAM = BnParamRole()


class ZCA(object):
    def __init__(self, n_components=None, data=None, filter_bias=0.1):
        self.filter_bias = np.float32(filter_bias)
        self.P = None
        self.P_inv = None
        self.n_components = 0
        self.is_fit = False
        if n_components and data:
            self.fit(n_components, data)

    def fit(self, n_components, data):
        if len(data.shape) == 2:
            self.reshape = None
        else:
            assert n_components == np.product(data.shape[1:]), \
                'ZCA whitening components should be %d for convolutional data'\
                % np.product(data.shape[1:])
            self.reshape = data.shape[1:]

        data = self._flatten_data(data)
        assert len(data.shape) == 2
        n, m = data.shape
        self.mean = np.mean(data, axis=0)

        bias = self.filter_bias * scipy.sparse.identity(m, 'float32')
        cov = np.cov(data, rowvar=0, bias=1) + bias
        eigs, eigv = scipy.linalg.eigh(cov)

        assert not np.isnan(eigs).any()
        assert not np.isnan(eigv).any()
        assert eigs.min() > 0

        if self.n_components:
            eigs = eigs[-self.n_components:]
            eigv = eigv[:, -self.n_components:]

        sqrt_eigs = np.sqrt(eigs)
        self.P = np.dot(eigv * (1.0 / sqrt_eigs), eigv.T)
        assert not np.isnan(self.P).any()
        self.P_inv = np.dot(eigv * sqrt_eigs, eigv.T)

        self.P = np.float32(self.P)
        self.P_inv = np.float32(self.P_inv)

        self.is_fit = True

    def apply(self, data, remove_mean=True):
        data = self._flatten_data(data)
        d = data - self.mean if remove_mean else data
        return self._reshape_data(np.dot(d, self.P))

    def inv(self, data, add_mean=True):
        d = np.dot(self._flatten_data(data), self.P_inv)
        d += self.mean if add_mean else 0.
        return self._reshape_data(d)

    def _flatten_data(self, data):
        if self.reshape is None:
            return data
        assert data.shape[1:] == self.reshape
        return data.reshape(data.shape[0], np.product(data.shape[1:]))

    def _reshape_data(self, data):
        assert len(data.shape) == 2
        if self.reshape is None:
            return data
        return np.reshape(data, (data.shape[0],) + self.reshape)


class ContrastNorm(object):
    def __init__(self, scale=55, epsilon=1e-8):
        self.scale = np.float32(scale)
        self.epsilon = np.float32(epsilon)

    def apply(self, data, copy=False):
        if copy:
            data = np.copy(data)
        data_shape = data.shape
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], np.product(data.shape[1:]))

        assert len(data.shape) == 2, 'Contrast norm on flattened data'

        data -= data.mean(axis=1)[:, np.newaxis]

        norms = np.sqrt(np.sum(data ** 2, axis=1)) / self.scale
        norms[norms < self.epsilon] = np.float32(1.)

        data /= norms[:, np.newaxis]

        if data_shape != data.shape:
            data = data.reshape(data_shape)

        return data


class TestMonitoring(object):
    def _get_bn_params(self, output_vars):
        # Pick out the nodes with batch normalization vars
        cg = ComputationGraph(output_vars)
        var_filter = VariableFilter(roles=[BNPARAM])
        bn_ps = var_filter(cg.variables)

        if len(bn_ps) == 0:
            logger.warn('No batch normalization parameters found - is' +
                        ' batch normalization turned off?')
            self._bn = False
            self._counter = None
            self._counter_max = None
            bn_share = []
            output_vars_replaced = output_vars
        else:
            self._bn = True
            assert len(set([p.name for p in bn_ps])) == len(bn_ps), \
                'Some batch norm params have the same name'
            logger.info('Batch norm parameters: %s' % ', '.join([p.name for p in bn_ps]))

            # Filter out the shared variables from the model updates
            def filter_share(par):
                lst = [up for up in cg.updates if up.name == 'shared_%s' % par.name]
                assert len(lst) == 1
                return lst[0]
            bn_share = map(filter_share, bn_ps)

            # Replace the BN coefficients in the test data model - Replace the
            # theano variables in the test graph with the shareds
            output_vars_replaced = cg.replace(zip(bn_ps, bn_share)).outputs

            # Pick out the counter
            self._counter = self._param_from_updates(cg.updates, 'counter')
            self._counter_max = self._param_from_updates(cg.updates, 'counter_max')

        return bn_ps, bn_share, output_vars_replaced

    def _param_from_updates(self, updates, p_name):
        var_filter = VariableFilter(roles=[BNPARAM])
        bn_ps = var_filter(updates.keys())
        p = [p for p in bn_ps if p.name == p_name]
        assert len(p) == 1, 'No %s of more than one %s' % (p_name, p_name)
        return p[0]

    def reset_counter(self):
        if self._bn:
            self._counter.set_value(np.float32(1))

    def replicate_vars(self, output_vars):
        # Problem in Blocks with multiple monitors monitoring the
        # same value in a graph. Therefore, they are all "replicated" to a new
        # Theano variable
        if isinstance(output_vars, (list, tuple)):
            return map(self.replicate_vars, output_vars)
        assert not hasattr(output_vars.tag, 'aggregation_scheme'), \
            'The variable %s already has an aggregator ' % output_vars.name + \
            'assigned to it - are you using a datasetmonitor with the same' + \
            ' variable as output? This might cause trouble in Blocks'
        new_var = 1 * output_vars
        new_var.name = output_vars.name
        return new_var


class ApproxTestMonitoring(DataStreamMonitoring, TestMonitoring):
    def __init__(self, output_vars, *args, **kwargs):
        output_vars = self.replicate_vars(output_vars)
        _, _, replaced_vars = self._get_bn_params(output_vars)
        super(ApproxTestMonitoring, self).__init__(replaced_vars, *args,
                                                   **kwargs)

    def do(self, which_callback, *args, **kwargs):
        assert not which_callback == "after_batch", "Do not monitor each mb"
        self.reset_counter()
        super(ApproxTestMonitoring, self).do(which_callback, *args, **kwargs)


class FinalTestMonitoring(SimpleExtension, MonitoringExtension, TestMonitoring):
    """Monitors validation and test set data with batch norm

    Calculates the training set statistics for batch normalization and adds
    them to the model before calculating the validation and test set values.
    This is done in two steps: First the training set is iterated and the
    statistics are saved in shared variables, then the model iterates through
    the test/validation set using the saved shared variables.
    When the training set is iterated, it is done for the full set, layer by
    layer so that the statistics are correct. This is expensive for very deep
    models, in which case some approximation could be in order
    """
    def __init__(self, output_vars, train_data_stream, test_data_stream,
                 **kwargs):
        output_vars = self.replicate_vars(output_vars)
        super(FinalTestMonitoring, self).__init__(**kwargs)
        self.trn_stream = train_data_stream
        self.tst_stream = test_data_stream

        bn_ps, bn_share, output_vars_replaced = self._get_bn_params(output_vars)

        if self._bn:
            updates = self._get_updates(bn_ps, bn_share)
            trn_evaluator = DatasetEvaluator(bn_ps, updates=updates)
        else:
            trn_evaluator = None

        self._trn_evaluator = trn_evaluator
        self._tst_evaluator = DatasetEvaluator(output_vars_replaced)

    def _get_updates(self, bn_ps, bn_share):
        cg = ComputationGraph(bn_ps)
        # Only store updates that relate to params or the counter
        updates = OrderedDict([(up, cg.updates[up]) for up in
                               cg.updates if up.name == 'counter' or
                               up in bn_share])
        assert self._counter == self._param_from_updates(cg.updates, 'counter')
        assert self._counter_max == self._param_from_updates(cg.updates,
                                                             'counter_max')
        assert len(updates) == len(bn_ps) + 1, \
            'Counter or var missing from update'
        return updates

    def do(self, which_callback, *args):
        """Write the values of monitored variables to the log."""
        assert not which_callback == "after_batch", "Do not monitor each mb"
        # Run on train data and get the statistics
        if self._bn:
            self._counter_max.set_value(np.float32(np.inf))
            self.reset_counter()
            self._trn_evaluator.evaluate(self.trn_stream)
            self.reset_counter()

        value_dict = self._tst_evaluator.evaluate(self.tst_stream)
        self.add_records(self.main_loop.log, value_dict.items())


class LRDecay(SimpleExtension):
    def __init__(self, lr, decay_first, decay_last, **kwargs):
        super(LRDecay, self).__init__(**kwargs)
        self.iter = 0
        self.decay_first = decay_first
        self.decay_last = decay_last
        self.lr = lr
        self.lr_init = np.float32(lr)

    def do(self, which_callback, *args):
        self.iter += 1
        if self.iter > self.decay_first:
            ratio = 1.0 * (self.decay_last - self.iter)
            ratio = np.maximum(0, ratio / (self.decay_last - self.decay_first))
            self.lr = np.float32(ratio * self.lr_init)
        logger.info("Iter %d, lr %f" % (self.iter, self.lr))


def global_meanpool_2d(x, num_filters):
    mean = tensor.mean(x.flatten(3), axis=2)
    mean = mean.dimshuffle(0, 1, 'x', 'x')
    return mean, (num_filters, 1, 1)


def pool_2d(x, mode="average", ws=(2, 2), stride=(2, 2)):
    import theano.sandbox.cuda as cuda
    assert cuda.dnn.dnn_available()
    return cuda.dnn.dnn_pool(x, ws=ws, stride=stride, mode=mode)


def maxpool_2d(z, in_dim, poolsize, poolstride):
    z = pool_2d(z, ds=poolsize, st=poolstride)
    output_size = tuple(Pool.out_shape(in_dim, poolsize, st=poolstride))
    return z, output_size

def softmax_n(x, axis=-1):
    e_x = tensor.exp(x - x.max(axis=axis, keepdims=True))
    out = e_x / e_x.sum(axis=axis, keepdims=True)
    return out
