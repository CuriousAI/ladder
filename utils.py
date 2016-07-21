import os

import logging
import numpy as np
import theano
from pandas import DataFrame, read_hdf

from blocks.extensions import Printing, SimpleExtension
from blocks.main_loop import MainLoop
from blocks.roles import add_role

logger = logging.getLogger('main.utils')


def shared_param(init, name, cast_float32, role, **kwargs):
    if cast_float32:
        v = np.float32(init)
    p = theano.shared(v, name=name, **kwargs)
    add_role(p, role)
    return p


class AttributeDict(dict):
    __getattr__ = dict.__getitem__

    def __setattr__(self, a, b):
        self.__setitem__(a, b)


class DummyLoop(MainLoop):
    def __init__(self, extensions):
        return super(DummyLoop, self).__init__(algorithm=None,
                                               data_stream=None,
                                               extensions=extensions)

    def run(self):
        for extension in self.extensions:
            extension.main_loop = self
        self._run_extensions('before_training')
        self._run_extensions('after_training')


class ShortPrinting(Printing):
    def __init__(self, to_print, use_log=True, **kwargs):
        self.to_print = to_print
        self.use_log = use_log
        super(ShortPrinting, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        log = self.main_loop.log

        # Iteration
        msg = "e {}, i {}:".format(
            log.status['epochs_done'],
            log.status['iterations_done'])

        # Requested channels
        items = []
        for k, vars in self.to_print.iteritems():
            for shortname, vars in vars.iteritems():
                if vars is None:
                    continue
                if type(vars) is not list:
                    vars = [vars]

                s = ""
                for var in vars:
                    try:
                        name = k + '_' + var.name
                        val = log.current_row[name]
                    except:
                        continue
                    try:
                        s += ' ' + ' '.join(["%.3g" % v for v in val])
                    except:
                        s += " %.3g" % val
                if s != "":
                    items += [shortname + s]
        msg = msg + ", ".join(items)
        if self.use_log:
            logger.info(msg)
        else:
            print msg


class SaveParams(SimpleExtension):
    """Finishes the training process when triggered."""
    def __init__(self, trigger_var, params, save_path, **kwargs):
        super(SaveParams, self).__init__(**kwargs)
        if trigger_var is None:
            self.var_name = None
        else:
            self.var_name = trigger_var[0] + '_' + trigger_var[1].name
        self.save_path = save_path
        self.params = params
        self.to_save = {}
        self.best_value = None
        self.add_condition(['after_training'], self.save)
        self.add_condition(['on_interrupt'], self.save)

    def save(self, which_callback, *args):
        if self.var_name is None:
            self.to_save = {v.name: v.get_value() for v in self.params}
        path = self.save_path + '/trained_params'
        logger.info('Saving to %s' % path)
        np.savez_compressed(path, **self.to_save)

    def do(self, which_callback, *args):
        if self.var_name is None:
            return
        val = self.main_loop.log.current_row[self.var_name]
        if self.best_value is None or val < self.best_value:
            self.best_value = val
        self.to_save = {v.name: v.get_value() for v in self.params}


class SaveExpParams(SimpleExtension):
    def __init__(self, experiment_params, dir, **kwargs):
        super(SaveExpParams, self).__init__(**kwargs)
        self.dir = dir
        self.experiment_params = experiment_params

    def do(self, which_callback, *args):
        df = DataFrame.from_dict(self.experiment_params, orient='index')
        df.to_hdf(os.path.join(self.dir, 'params'), 'params', mode='w',
                  complevel=5, complib='blosc')


class SaveLog(SimpleExtension):
    def __init__(self, dir, show=None, **kwargs):
        super(SaveLog, self).__init__(**kwargs)
        self.dir = dir
        self.show = show if show is not None else []

    def do(self, which_callback, *args):
        df = DataFrame.from_dict(self.main_loop.log, orient='index')
        df.to_hdf(os.path.join(self.dir, 'log'), 'log', mode='w',
                  complevel=5, complib='blosc')


def prepare_dir(save_to, results_dir='results'):
    base = os.path.join(results_dir, save_to)
    i = 0

    while True:
        name = base + str(i)
        try:
            os.makedirs(name)
            break
        except:
            i += 1

    return name


def load_df(dirpath, filename, varname=None):
    varname = filename if varname is None else varname
    fn = os.path.join(dirpath, filename)
    return read_hdf(fn, varname)


def filter_funcs_prefix(d, pfx):
    pfx = 'cmd_'
    fp = lambda x: x.find(pfx)
    return {n[fp(n) + len(pfx):]: v for n, v in d.iteritems() if fp(n) >= 0}
