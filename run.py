#!/usr/bin/env python

import functools
import logging
import os
import subprocess
from argparse import ArgumentParser, Action
from collections import OrderedDict
import sys
from pandas import DataFrame

import numpy
import time
import theano
from theano.tensor.type import TensorType

from blocks.algorithms import GradientDescent, Adam
from blocks.extensions import FinishAfter
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.roles import PARAMETER
from fuel.datasets import MNIST, CIFAR10
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import Transformer

from picklable_itertools import cycle, imap
from itertools import izip, product, tee

logger = logging.getLogger('main')

from utils import ShortPrinting, prepare_dir, load_df, DummyLoop
from utils import SaveExpParams, SaveLog, SaveParams, AttributeDict
from nn import ZCA, ContrastNorm
from nn import ApproxTestMonitoring, FinalTestMonitoring, TestMonitoring
from nn import LRDecay
from ladder import LadderAE


class Whitening(Transformer):
    """ Makes a copy of the examples in the underlying dataset and whitens it
        if necessary.
    """
    def __init__(self, data_stream, iteration_scheme, whiten, cnorm=None,
                 **kwargs):
        super(Whitening, self).__init__(data_stream,
                                        iteration_scheme=iteration_scheme,
                                        **kwargs)
        data = data_stream.get_data(slice(data_stream.dataset.num_examples))
        self.data = []
        for s, d in zip(self.sources, data):
            if 'features' == s:
                # Fuel provides Cifar in uint8, convert to float32
                d = numpy.require(d, dtype=numpy.float32)
                if cnorm is not None:
                    d = cnorm.apply(d)
                if whiten is not None:
                    d = whiten.apply(d)
                self.data += [d]
            elif 'targets' == s:
                d = unify_labels(d)
                self.data += [d]
            else:
                raise Exception("Unsupported Fuel target: %s" % s)

    def get_data(self, request=None):
        return (s[request] for s in self.data)


class SemiDataStream(Transformer):
    """ Combines two datastreams into one such that 'target' source (labels)
        is used only from the first one. The second one is renamed
        to avoid collision. Upon iteration, the first one is repeated until
        the second one depletes.
        """
    def __init__(self, data_stream_labeled, data_stream_unlabeled, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.ds_labeled = data_stream_labeled
        self.ds_unlabeled = data_stream_unlabeled
        # Rename the sources for clarity
        self.ds_labeled.sources = ('features_labeled', 'targets_labeled')
        # Rename the source for input pixels and hide its labels!
        self.ds_unlabeled.sources = ('features_unlabeled',)

    @property
    def sources(self):
        if hasattr(self, '_sources'):
            return self._sources
        return self.ds_labeled.sources + self.ds_unlabeled.sources

    @sources.setter
    def sources(self, value):
        self._sources = value

    def close(self):
        self.ds_labeled.close()
        self.ds_unlabeled.close()

    def reset(self):
        self.ds_labeled.reset()
        self.ds_unlabeled.reset()

    def next_epoch(self):
        self.ds_labeled.next_epoch()
        self.ds_unlabeled.next_epoch()

    def get_epoch_iterator(self, **kwargs):
        unlabeled = self.ds_unlabeled.get_epoch_iterator(**kwargs)
        labeled = self.ds_labeled.get_epoch_iterator(**kwargs)
        assert type(labeled) == type(unlabeled)

        return imap(self.mergedicts, cycle(labeled), unlabeled)

    def mergedicts(self, x, y):
        return dict(list(x.items()) + list(y.items()))


def unify_labels(y):
    """ Work-around for Fuel bug where MNIST and Cifar-10
    datasets have different dimensionalities for the targets:
    e.g. (50000, 1) vs (60000,) """
    yshape = y.shape
    y = y.flatten()
    assert y.shape[0] == yshape[0]
    return y


def make_datastream(dataset, indices, batch_size,
                    n_labeled=None, n_unlabeled=None,
                    balanced_classes=True, whiten=None, cnorm=None,
                    scheme=ShuffledScheme):
    if n_labeled is None or n_labeled == 0:
        n_labeled = len(indices)
    if batch_size is None:
        batch_size = len(indices)
    if n_unlabeled is None:
        n_unlabeled = len(indices)
    assert n_labeled <= n_unlabeled, 'need less labeled than unlabeled'

    if balanced_classes and n_labeled < n_unlabeled:
        # Ensure each label is equally represented
        logger.info('Balancing %d labels...' % n_labeled)
        all_data = dataset.data_sources[dataset.sources.index('targets')]
        y = unify_labels(all_data)[indices]
        n_classes = y.max() + 1
        assert n_labeled % n_classes == 0
        n_from_each_class = n_labeled / n_classes

        i_labeled = []
        for c in range(n_classes):
            i = (indices[y == c])[:n_from_each_class]
            i_labeled += list(i)
    else:
        i_labeled = indices[:n_labeled]

    # Get unlabeled indices
    i_unlabeled = indices[:n_unlabeled]

    ds = SemiDataStream(
        data_stream_labeled=Whitening(
            DataStream(dataset),
            iteration_scheme=scheme(i_labeled, batch_size),
            whiten=whiten, cnorm=cnorm),
        data_stream_unlabeled=Whitening(
            DataStream(dataset),
            iteration_scheme=scheme(i_unlabeled, batch_size),
            whiten=whiten, cnorm=cnorm)
    )
    return ds


def setup_model(p):
    ladder = LadderAE(p)
    # Setup inputs
    input_type = TensorType('float32', [False] * (len(p.encoder_layers[0]) + 1))
    x_only = input_type('features_unlabeled')
    x = input_type('features_labeled')
    y = theano.tensor.lvector('targets_labeled')
    ladder.apply(x, y, x_only)

    # Load parameters if requested
    if p.get('load_from'):
        with open(p.load_from + '/trained_params.npz') as f:
            loaded = numpy.load(f)
            cg = ComputationGraph([ladder.costs.total])
            current_params = VariableFilter(roles=[PARAMETER])(cg.variables)
            logger.info('Loading parameters: %s' % ', '.join(loaded.keys()))
            for param in current_params:
                assert param.get_value().shape == loaded[param.name].shape
                param.set_value(loaded[param.name])

    return ladder


def load_and_log_params(cli_params):
    cli_params = AttributeDict(cli_params)
    if cli_params.get('load_from'):
        p = load_df(cli_params.load_from, 'params').to_dict()[0]
        p = AttributeDict(p)
        for key in cli_params.iterkeys():
            if key not in p:
                p[key] = None
        new_params = cli_params
        loaded = True
    else:
        p = cli_params
        new_params = {}
        loaded = False

        # Make dseed seed unless specified explicitly
        if p.get('dseed') is None and p.get('seed') is not None:
            p['dseed'] = p['seed']

    logger.info('== COMMAND LINE ==')
    logger.info(' '.join(sys.argv))

    logger.info('== PARAMETERS ==')
    for k, v in p.iteritems():
        if new_params.get(k) is not None:
            p[k] = new_params[k]
            replace_str = "<- " + str(new_params.get(k))
        else:
            replace_str = ""
        logger.info(" {:20}: {:<20} {}".format(k, v, replace_str))
    return p, loaded


def setup_data(p, test_set=False):
    dataset_class, training_set_size = {
        'cifar10': (CIFAR10, 40000),
        'mnist': (MNIST, 50000),
    }[p.dataset]

    # Allow overriding the default from command line
    if p.get('unlabeled_samples') is not None:
        training_set_size = p.unlabeled_samples

    train_set = dataset_class(["train"])

    # Make sure the MNIST data is in right format
    if p.dataset == 'mnist':
        d = train_set.data_sources[train_set.sources.index('features')]
        assert numpy.all(d <= 1.0) and numpy.all(d >= 0.0), \
            'Make sure data is in float format and in range 0 to 1'

    # Take all indices and permutate them
    all_ind = numpy.arange(train_set.num_examples)
    if p.get('dseed'):
        rng = numpy.random.RandomState(seed=p.dseed)
        rng.shuffle(all_ind)

    d = AttributeDict()

    # Choose the training set
    d.train = train_set
    d.train_ind = all_ind[:training_set_size]

    # Then choose validation set from the remaining indices
    d.valid = train_set
    d.valid_ind = numpy.setdiff1d(all_ind, d.train_ind)[:p.valid_set_size]
    logger.info('Using %d examples for validation' % len(d.valid_ind))

    # Only touch test data if requested
    if test_set:
        d.test = dataset_class(["test"])
        d.test_ind = numpy.arange(d.test.num_examples)

    # Setup optional whitening, only used for Cifar-10
    in_dim = train_set.data_sources[train_set.sources.index('features')].shape[1:]
    if len(in_dim) > 1 and p.whiten_zca > 0:
        assert numpy.product(in_dim) == p.whiten_zca, \
            'Need %d whitening dimensions, not %d' % (numpy.product(in_dim),
                                                      p.whiten_zca)
    cnorm = ContrastNorm(p.contrast_norm) if p.contrast_norm != 0 else None

    def get_data(d, i):
        data = d.get_data(request=list(i))[d.sources.index('features')]
        # Fuel provides Cifar in uint8, convert to float32
        data = numpy.require(data, dtype=numpy.float32)
        return data if cnorm is None else cnorm.apply(data)

    if p.whiten_zca > 0:
        logger.info('Whitening using %d ZCA components' % p.whiten_zca)
        whiten = ZCA()
        whiten.fit(p.whiten_zca, get_data(d.train, d.train_ind))
    else:
        whiten = None

    return in_dim, d, whiten, cnorm


def get_error(args):
    """ Calculate the classification error """
    args['data_type'] = args.get('data_type', 'test')
    args['no_load'] = 'g_'

    targets, acts = analyze(args)
    guess = numpy.argmax(acts, axis=1)
    correct = numpy.sum(numpy.equal(guess, targets.flatten()))

    return (1. - correct / float(len(guess))) * 100.


def analyze(cli_params):
    p, _ = load_and_log_params(cli_params)
    _, data, whiten, cnorm = setup_data(p, test_set=True)
    ladder = setup_model(p)

    # Analyze activations
    dset, indices, calc_batchnorm = {
        'train': (data.train, data.train_ind, False),
        'valid': (data.valid, data.valid_ind, True),
        'test':  (data.test, data.test_ind, True),
    }[p.data_type]

    if calc_batchnorm:
        logger.info('Calculating batch normalization for clean.labeled path')
        main_loop = DummyLoop(
            extensions=[
                FinalTestMonitoring(
                    [ladder.costs.class_clean, ladder.error.clean]
                    + ladder.costs.denois.values(),
                    make_datastream(data.train, data.train_ind,
                                    # These need to match with the training
                                    p.batch_size,
                                    n_labeled=p.labeled_samples,
                                    n_unlabeled=len(data.train_ind),
                                    cnorm=cnorm,
                                    whiten=whiten, scheme=ShuffledScheme),
                    make_datastream(data.valid, data.valid_ind,
                                    p.valid_batch_size,
                                    n_labeled=len(data.valid_ind),
                                    n_unlabeled=len(data.valid_ind),
                                    cnorm=cnorm,
                                    whiten=whiten, scheme=ShuffledScheme),
                    prefix="valid_final", before_training=True),
                ShortPrinting({
                    "valid_final": OrderedDict([
                        ('VF_C_class', ladder.costs.class_clean),
                        ('VF_E', ladder.error.clean),
                        ('VF_C_de', [ladder.costs.denois.get(0),
                                     ladder.costs.denois.get(1),
                                     ladder.costs.denois.get(2),
                                     ladder.costs.denois.get(3)]),
                    ]),
                }, after_training=True, use_log=False),
            ])
        main_loop.run()

    # Make a datastream that has all the indices in the labeled pathway
    ds = make_datastream(dset, indices,
                         batch_size=p.get('batch_size'),
                         n_labeled=len(indices),
                         n_unlabeled=len(indices),
                         balanced_classes=False,
                         whiten=whiten,
                         cnorm=cnorm,
                         scheme=SequentialScheme)

    # We want out the values after softmax
    outputs = ladder.act.clean.labeled.h[len(ladder.layers) - 1]

    # Replace the batch normalization paramameters with the shared variables
    if calc_batchnorm:
        outputreplacer = TestMonitoring()
        _, _,  outputs = outputreplacer._get_bn_params(outputs)

    cg = ComputationGraph(outputs)
    f = cg.get_theano_function()

    it = ds.get_epoch_iterator(as_dict=True)
    res = []
    inputs = {'features_labeled': [],
              'targets_labeled': [],
              'features_unlabeled': []}
    # Loop over one epoch
    for d in it:
        # Store all inputs
        for k, v in d.iteritems():
            inputs[k] += [v]
        # Store outputs
        res += [f(*[d[str(inp)] for inp in cg.inputs])]

    # Concatenate all minibatches
    res = [numpy.vstack(minibatches) for minibatches in zip(*res)]
    inputs = {k: numpy.vstack(v) for k, v in inputs.iteritems()}

    return inputs['targets_labeled'], res[0]


def train(cli_params):
    cli_params['save_dir'] = prepare_dir(cli_params['save_to'])
    logfile = os.path.join(cli_params['save_dir'], 'log.txt')

    # Log also DEBUG to a file
    fh = logging.FileHandler(filename=logfile)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    logger.info('Logging into %s' % logfile)

    p, loaded = load_and_log_params(cli_params)
    in_dim, data, whiten, cnorm = setup_data(p, test_set=False)
    if not loaded:
        # Set the zero layer to match input dimensions
        p.encoder_layers = (in_dim,) + p.encoder_layers

    ladder = setup_model(p)

    # Training
    all_params = ComputationGraph([ladder.costs.total]).parameters
    logger.info('Found the following parameters: %s' % str(all_params))

    # Fetch all batch normalization updates. They are in the clean path.
    bn_updates = ComputationGraph([ladder.costs.class_clean]).updates
    assert 'counter' in [u.name for u in bn_updates.keys()], \
        'No batch norm params in graph - the graph has been cut?'

    training_algorithm = GradientDescent(
        cost=ladder.costs.total, parameters=all_params,
        step_rule=Adam(learning_rate=ladder.lr))
    # In addition to actual training, also do BN variable approximations
    training_algorithm.add_updates(bn_updates)

    short_prints = {
        "train": {
            'T_C_class': ladder.costs.class_corr,
            'T_C_de': ladder.costs.denois.values(),
        },
        "valid_approx": OrderedDict([
            ('V_C_class', ladder.costs.class_clean),
            ('V_E', ladder.error.clean),
            ('V_C_de', ladder.costs.denois.values()),
        ]),
        "valid_final": OrderedDict([
            ('VF_C_class', ladder.costs.class_clean),
            ('VF_E', ladder.error.clean),
            ('VF_C_de', ladder.costs.denois.values()),
        ]),
    }

    main_loop = MainLoop(
        training_algorithm,
        # Datastream used for training
        make_datastream(data.train, data.train_ind,
                        p.batch_size,
                        n_labeled=p.labeled_samples,
                        n_unlabeled=p.unlabeled_samples,
                        whiten=whiten,
                        cnorm=cnorm),
        model=Model(ladder.costs.total),
        extensions=[
            FinishAfter(after_n_epochs=p.num_epochs),

            # This will estimate the validation error using
            # running average estimates of the batch normalization
            # parameters, mean and variance
            ApproxTestMonitoring(
                [ladder.costs.class_clean, ladder.error.clean]
                + ladder.costs.denois.values(),
                make_datastream(data.valid, data.valid_ind,
                                p.valid_batch_size, whiten=whiten, cnorm=cnorm,
                                scheme=ShuffledScheme),
                prefix="valid_approx"),

            # This Monitor is slower, but more accurate since it will first
            # estimate batch normalization parameters from training data and
            # then do another pass to calculate the validation error.
            FinalTestMonitoring(
                [ladder.costs.class_clean, ladder.error.clean]
                + ladder.costs.denois.values(),
                make_datastream(data.train, data.train_ind,
                                p.batch_size,
                                n_labeled=p.labeled_samples,
                                whiten=whiten, cnorm=cnorm,
                                scheme=ShuffledScheme),
                make_datastream(data.valid, data.valid_ind,
                                p.valid_batch_size,
                                n_labeled=len(data.valid_ind),
                                whiten=whiten, cnorm=cnorm,
                                scheme=ShuffledScheme),
                prefix="valid_final",
                after_n_epochs=p.num_epochs),

            TrainingDataMonitoring(
                [ladder.costs.total, ladder.costs.class_corr,
                 training_algorithm.total_gradient_norm]
                + ladder.costs.denois.values(),
                prefix="train", after_epoch=True),

            SaveParams(None, all_params, p.save_dir, after_epoch=True),
            SaveExpParams(p, p.save_dir, before_training=True),
            SaveLog(p.save_dir, after_training=True),
            ShortPrinting(short_prints),
            LRDecay(ladder.lr, p.num_epochs * p.lrate_decay, p.num_epochs,
                    after_epoch=True),
        ])
    main_loop.run()

    # Get results
    df = DataFrame.from_dict(main_loop.log, orient='index')
    col = 'valid_final_error_rate_clean'
    logger.info('%s %g' % (col, df[col].iloc[-1]))

    if main_loop.log.status['epoch_interrupt_received']:
        return None
    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rep = lambda s: s.replace('-', ',')
    chop = lambda s: s.split(',')
    to_int = lambda ss: [int(s) for s in ss if s.isdigit()]
    to_float = lambda ss: [float(s) for s in ss]

    def to_bool(s):
        if s.lower() in ['true', 't']:
            return True
        elif s.lower() in ['false', 'f']:
            return False
        else:
            raise Exception("Unknown bool value %s" % s)

    def compose(*funs):
        return functools.reduce(lambda f, g: lambda x: f(g(x)), funs)

    # Functional parsing logic to allow flexible function compositions
    # as actions for ArgumentParser
    def funcs(additional_arg):
        class customAction(Action):
            def __call__(self, parser, args, values, option_string=None):

                def process(arg, func_list):
                    if arg is None:
                        return None
                    elif type(arg) is list:
                        return map(compose(*func_list), arg)
                    else:
                        return compose(*func_list)(arg)

                setattr(args, self.dest, process(values, additional_arg))
        return customAction

    def add_train_params(parser, use_defaults):
        a = parser.add_argument
        default = lambda x: x if use_defaults else None

        # General hyper parameters and settings
        a("save_to", help="Destination to save the state and results",
          default=default("noname"), nargs="?")
        a("--num-epochs", help="Number of training epochs",
          type=int, default=default(150))
        a("--seed", help="Seed",
          type=int, default=default([1]), nargs='+')
        a("--dseed", help="Data permutation seed, defaults to 'seed'",
          type=int, default=default([None]), nargs='+')
        a("--labeled-samples", help="How many supervised samples are used",
          type=int, default=default(None), nargs='+')
        a("--unlabeled-samples", help="How many unsupervised samples are used",
          type=int, default=default(None), nargs='+')
        a("--dataset", type=str, default=default(['mnist']), nargs='+',
          choices=['mnist', 'cifar10'], help="Which dataset to use")
        a("--lr", help="Initial learning rate",
          type=float, default=default([0.002]), nargs='+')
        a("--lrate-decay", help="When to linearly start decaying lrate (0-1)",
          type=float, default=default([0.67]), nargs='+')
        a("--batch-size", help="Minibatch size",
          type=int, default=default([100]), nargs='+')
        a("--valid-batch-size", help="Minibatch size for validation data",
          type=int, default=default([100]), nargs='+')
        a("--valid-set-size", help="Number of examples in validation set",
          type=int, default=default([10000]), nargs='+')

        # Hyperparameters controlling supervised path
        a("--super-noise-std", help="Noise added to supervised learning path",
          type=float, default=default([0.3]), nargs='+')
        a("--f-local-noise-std", help="Noise added encoder path",
          type=str, default=default([0.3]), nargs='+',
          action=funcs([tuple, to_float, chop]))
        a("--act", nargs='+', type=str, action=funcs([tuple, chop, rep]),
          default=default(["relu"]), help="List of activation functions")
        a("--encoder-layers", help="List of layers for f",
          type=str, default=default(()), action=funcs([tuple, chop, rep]))

        # Hyperparameters controlling unsupervised training
        a("--denoising-cost-x", help="Weight of the denoising cost.",
          type=str, default=default([(0.,)]), nargs='+',
          action=funcs([tuple, to_float, chop]))
        a("--decoder-spec", help="List of decoding function types", nargs='+',
          type=str, default=default(['sig']), action=funcs([tuple, chop, rep]))
        a("--zestbn", type=str, default=default(['bugfix']), nargs='+',
          choices=['bugfix', 'no'], help="How to do zest bn")

        # Hyperparameters used for Cifar training
        a("--contrast-norm", help="Scale of contrast normalization (0=off)",
          type=int, default=default([0]), nargs='+')
        a("--top-c", help="Have c at softmax?", action=funcs([to_bool]),
          default=default([True]), nargs='+')
        a("--whiten-zca", help="Whether to whiten the data with ZCA",
          type=int, default=default([0]), nargs='+')

    ap = ArgumentParser("Semisupervised experiment")
    subparsers = ap.add_subparsers(dest='cmd', help='sub-command help')

    # TRAIN
    train_cmd = subparsers.add_parser('train', help='Train a new model')
    add_train_params(train_cmd, use_defaults=True)

    # EVALUATE
    load_cmd = subparsers.add_parser('evaluate', help='Evaluate test error')
    load_cmd.add_argument('load_from', type=str,
                          help="Destination to load the state from")
    load_cmd.add_argument('--data-type', type=str, default='test',
                          help="Data set to evaluate on")

    args = ap.parse_args()

    subp = subprocess.Popen(['git', 'rev-parse', 'HEAD'],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out, err = subp.communicate()
    args.commit = out.strip()
    if err.strip():
        logger.error('Subprocess returned %s' % err.strip())

    t_start = time.time()
    if args.cmd == 'evaluate':
        for k, v in vars(args).iteritems():
            if type(v) is list:
                assert len(v) == 1, "should not be a list when loading: %s" % k
                logger.info("%s" % str(v[0]))
                vars(args)[k] = v[0]

        err = get_error(vars(args))
        logger.info('Test error: %f' % err)

    elif args.cmd == "train":
        listdicts = {k: v for k, v in vars(args).iteritems() if type(v) is list}
        therest = {k: v for k, v in vars(args).iteritems() if type(v) is not list}

        gen1, gen2 = tee(product(*listdicts.itervalues()))

        l = len(list(gen1))
        for i, d in enumerate(dict(izip(listdicts, x)) for x in gen2):
            if l > 1:
                logger.info('Training configuration %d / %d' % (i+1, l))
            d.update(therest)
            if train(d) is None:
                break
    logger.info('Took %.1f minutes' % ((time.time() - t_start) / 60.))
