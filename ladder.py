import logging

import numpy as np
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet.conv import conv2d, ConvOp
from theano.sandbox.cuda.blas import GpuCorrMM
from theano.sandbox.cuda.basic_ops import gpu_contiguous

from blocks.bricks.cost import SquaredError
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.graph import add_annotation, Annotation
from blocks.roles import add_role, PARAMETER, WEIGHT, BIAS

from utils import shared_param, AttributeDict
from nn import maxpool_2d, global_meanpool_2d, BNPARAM, softmax_n

logger = logging.getLogger('main.model')
floatX = theano.config.floatX


class LadderAE():
    def __init__(self, p):
        self.p = p
        self.init_weights_transpose = False
        self.default_lr = p.lr
        self.shareds = OrderedDict()
        self.rstream = RandomStreams(seed=p.seed)
        self.rng = np.random.RandomState(seed=p.seed)

        n_layers = len(p.encoder_layers)
        assert n_layers > 1, "Need to define encoder layers"
        assert n_layers == len(p.denoising_cost_x), (
            "Number of denoising costs does not match with %d layers: %s" %
            (n_layers, str(p.denoising_cost_x)))

        def one_to_all(x):
            """ (5.,) -> 5 -> (5., 5., 5.)
                ('relu',) -> 'relu' -> ('relu', 'relu', 'relu')
            """
            if type(x) is tuple and len(x) == 1:
                x = x[0]

            if type(x) is float:
                x = (np.float32(x),) * n_layers

            if type(x) is str:
                x = (x,) * n_layers
            return x

        p.decoder_spec = one_to_all(p.decoder_spec)
        p.f_local_noise_std = one_to_all(p.f_local_noise_std)
        acts = one_to_all(p.get('act', 'relu'))

        assert n_layers == len(p.decoder_spec), "f and g need to match"
        assert (n_layers == len(acts)), (
            "Not enough activations given. Requires %d. Got: %s" %
            (n_layers, str(acts)))
        acts = acts[:-1] + ('softmax',)

        def parse_layer(spec):
            """ 'fc:5' -> ('fc', 5)
                '5'    -> ('fc', 5)
                5      -> ('fc', 5)
                'convv:3:2:2' -> ('convv', [3,2,2])
            """
            if type(spec) is not str:
                return "fc", spec
            spec = spec.split(':')
            l_type = spec.pop(0) if len(spec) >= 2 else "fc"
            spec = map(int, spec)
            spec = spec[0] if len(spec) == 1 else spec
            return l_type, spec

        enc = map(parse_layer, p.encoder_layers)
        self.layers = list(enumerate(zip(enc, p.decoder_spec, acts)))

    def weight(self, init, name, cast_float32=True, for_conv=False):
        weight = self.shared(init, name, cast_float32, role=WEIGHT)
        if for_conv:
            return weight.dimshuffle('x', 0, 'x', 'x')
        return weight

    def bias(self, init, name, cast_float32=True, for_conv=False):
        b = self.shared(init, name, cast_float32, role=BIAS)
        if for_conv:
            return b.dimshuffle('x', 0, 'x', 'x')
        return b

    def shared(self, init, name, cast_float32=True, role=PARAMETER, **kwargs):
        p = self.shareds.get(name)
        if p is None:
            p = shared_param(init, name, cast_float32, role, **kwargs)
            self.shareds[name] = p
        return p

    def counter(self):
        name = 'counter'
        p = self.shareds.get(name)
        update = []
        if p is None:
            p_max_val = np.float32(10)
            p = self.shared(np.float32(1), name, role=BNPARAM)
            p_max = self.shared(p_max_val, name + '_max', role=BNPARAM)
            update = [(p, T.clip(p + np.float32(1), np.float32(0), p_max)),
                      (p_max, p_max_val)]
        return (p, update)

    def noise_like(self, x):
        noise = self.rstream.normal(size=x.shape, avg=0.0, std=1.0)
        return T.cast(noise, dtype=floatX)

    def rand_init(self, in_dim, out_dim):
        """ Random initialization for fully connected layers """
        W = self.rng.randn(in_dim, out_dim) / np.sqrt(in_dim)
        return W

    def rand_init_conv(self, dim):
        """ Random initialization for convolution filters """
        fan_in = np.prod(dtype=floatX, a=dim[1:])
        bound = np.sqrt(3. / max(1.0, (fan_in)))
        W = np.asarray(
            self.rng.uniform(low=-bound, high=bound, size=dim), dtype=floatX)
        return W

    def new_activation_dict(self):
        return AttributeDict({'z': {}, 'h': {}, 's': {}, 'm': {}})

    def annotate_update(self, update, tag_to):
        a = Annotation()
        for (var, up) in update:
            a.updates[var] = up
        add_annotation(tag_to, a)

    def apply(self, input_labeled, target_labeled, input_unlabeled):
        self.layer_counter = 0
        input_dim = self.p.encoder_layers[0]

        # Store the dimension tuples in the same order as layers.
        layers = self.layers
        self.layer_dims = {0: input_dim}

        self.lr = self.default_lr

        self.costs = costs = AttributeDict()
        self.costs.denois = AttributeDict()

        self.act = AttributeDict()
        self.error = AttributeDict()

        top = len(layers) - 1

        if input_labeled is None:
            N = 0
        else:
            N = input_labeled.shape[0]
        self.join = lambda l, u: T.concatenate([l, u], axis=0) if l else u
        self.labeled = lambda x: x[:N] if x is not None else x
        self.unlabeled = lambda x: x[N:] if x is not None else x
        self.split_lu = lambda x: (self.labeled(x), self.unlabeled(x))

        input_concat = self.join(input_labeled, input_unlabeled)

        def encoder(input_, path_name, input_noise_std=0, noise_std=[]):
            h = input_

            logger.info('  0: noise %g' % input_noise_std)
            if input_noise_std > 0.:
                h = h + self.noise_like(h) * input_noise_std

            d = AttributeDict()
            d.unlabeled = self.new_activation_dict()
            d.labeled = self.new_activation_dict()
            d.labeled.z[0] = self.labeled(h)
            d.unlabeled.z[0] = self.unlabeled(h)
            prev_dim = input_dim
            for i, (spec, _, act_f) in layers[1:]:
                d.labeled.h[i - 1], d.unlabeled.h[i - 1] = self.split_lu(h)
                noise = noise_std[i] if i < len(noise_std) else 0.
                curr_dim, z, m, s, h = self.f(h, prev_dim, spec, i, act_f,
                                              path_name=path_name,
                                              noise_std=noise)
                assert self.layer_dims.get(i) in (None, curr_dim)
                self.layer_dims[i] = curr_dim
                d.labeled.z[i], d.unlabeled.z[i] = self.split_lu(z)
                d.unlabeled.s[i] = s
                d.unlabeled.m[i] = m
                prev_dim = curr_dim
            d.labeled.h[i], d.unlabeled.h[i] = self.split_lu(h)
            return d

        # Clean, supervised
        logger.info('Encoder: clean, labeled')
        clean = self.act.clean = encoder(input_concat, 'clean')

        # Corrupted, supervised
        logger.info('Encoder: corr, labeled')
        corr = self.act.corr = encoder(input_concat, 'corr',
                                       input_noise_std=self.p.super_noise_std,
                                       noise_std=self.p.f_local_noise_std)
        est = self.act.est = self.new_activation_dict()

        # Decoder path in opposite order
        logger.info('Decoder: z_corr -> z_est')
        for i, ((_, spec), l_type, act_f) in layers[::-1]:
            z_corr = corr.unlabeled.z[i]
            z_clean = clean.unlabeled.z[i]
            z_clean_s = clean.unlabeled.s.get(i)
            z_clean_m = clean.unlabeled.m.get(i)
            fspec = layers[i+1][1][0] if len(layers) > i+1 else (None, None)

            if i == top:
                ver = corr.unlabeled.h[i]
                ver_dim = self.layer_dims[i]
                top_g = True
            else:
                ver = est.z.get(i + 1)
                ver_dim = self.layer_dims.get(i + 1)
                top_g = False

            z_est = self.g(z_lat=z_corr,
                           z_ver=ver,
                           in_dims=ver_dim,
                           out_dims=self.layer_dims[i],
                           l_type=l_type,
                           num=i,
                           fspec=fspec,
                           top_g=top_g)

            if z_est is not None:
                # Denoising cost

                if z_clean_s and self.p.zestbn == 'bugfix':
                    z_est_norm = (z_est - z_clean_m) / T.sqrt(z_clean_s + np.float32(1e-10))
                elif z_clean_s is None or self.p.zestbn == 'no':
                    z_est_norm = z_est
                else:
                    assert False, 'Not supported path'

                se = SquaredError('denois' + str(i))
                costs.denois[i] = se.apply(z_est_norm.flatten(2),
                                           z_clean.flatten(2)) \
                    / np.prod(self.layer_dims[i], dtype=floatX)
                costs.denois[i].name = 'denois' + str(i)
                denois_print = 'denois %.2f' % self.p.denoising_cost_x[i]
            else:
                denois_print = ''

            # Store references for later use
            est.h[i] = self.apply_act(z_est, act_f)
            est.z[i] = z_est
            est.s[i] = None
            est.m[i] = None
            logger.info('  g%d: %10s, %s, dim %s -> %s' % (
                i, l_type,
                denois_print,
                self.layer_dims.get(i+1),
                self.layer_dims.get(i)
                ))

        # Costs
        y = target_labeled.flatten()

        costs.class_clean = CategoricalCrossEntropy().apply(y, clean.labeled.h[top])
        costs.class_clean.name = 'cost_class_clean'

        costs.class_corr = CategoricalCrossEntropy().apply(y, corr.labeled.h[top])
        costs.class_corr.name = 'cost_class_corr'

        # This will be used for training
        costs.total = costs.class_corr * 1.0
        for i in range(top + 1):
            if costs.denois.get(i) and self.p.denoising_cost_x[i] > 0:
                costs.total += costs.denois[i] * self.p.denoising_cost_x[i]
        costs.total.name = 'cost_total'

        # Classification error
        mr = MisclassificationRate()
        self.error.clean = mr.apply(y, clean.labeled.h[top]) * np.float32(100.)
        self.error.clean.name = 'error_rate_clean'

    def apply_act(self, input, act_name):
        if input is None:
            return input
        act = {
            'relu': lambda x: T.maximum(0, x),
            'leakyrelu': lambda x: T.switch(x > 0., x, 0.1 * x),
            'linear': lambda x: x,
            'softplus': lambda x: T.log(1. + T.exp(x)),
            'sigmoid': lambda x: T.nnet.sigmoid(x),
            'softmax': lambda x: softmax_n(x),
        }.get(act_name)
        assert act, 'unknown act %s' % act_name
        if act_name == 'softmax':
            input = input.flatten(2)
        return act(input)

    def annotate_bn(self, var, id, var_type, mb_size, size, norm_ax):
        var_shape = np.array((1,) + size)
        out_dim = np.prod(var_shape) / np.prod(var_shape[list(norm_ax)])
        # Flatten the var - shared variable updating is not trivial otherwise,
        # as theano seems to believe a row vector is a matrix and will complain
        # about the updates
        orig_shape = var.shape
        var = var.flatten()
        # Here we add the name and role, the variables will later be identified
        # by these values
        var.name = id + '_%s_clean' % var_type
        add_role(var, BNPARAM)
        shared_var = self.shared(np.zeros(out_dim),
                                 name='shared_%s' % var.name, role=None)

        # Update running average estimates. When the counter is reset to 1, it
        # will clear its memory
        cntr, c_up = self.counter()
        one = np.float32(1)
        run_avg = lambda new, old: one / cntr * new + (one - one / cntr) * old
        if var_type == 'mean':
            new_value = run_avg(var, shared_var)
        elif var_type == 'var':
            mb_size = T.cast(mb_size, 'float32')
            new_value = run_avg(mb_size / (mb_size - one) * var, shared_var)
        else:
            raise NotImplemented('Unknown batch norm var %s' % var_type)
        # Add the counter update to the annotated update if it is the first
        # instance of a counter
        self.annotate_update([(shared_var, new_value)] + c_up, var)

        return var.reshape(orig_shape)

    def f(self, h, in_dim, spec, num, act_f, path_name, noise_std=0):
        # Generates identifiers used for referencing shared variables.
        # E.g. clean and corrupted encoders will end up using the same
        # variable name and hence sharing parameters
        gen_id = lambda s: '_'.join(['f', str(num), s])
        layer_type, _ = spec

        # Pooling
        if layer_type in ['maxpool', 'globalmeanpool']:
            z, output_size = self.f_pool(h, spec, in_dim)
            norm_ax = (0, -2, -1)
            # after pooling, no activation func for now unless its softmax
            act_f = "linear" if act_f != "softmax" else act_f

        # Convolution
        elif layer_type in ['convv', 'convf']:
            z, output_size = self.f_conv(h, spec, in_dim, gen_id('W'))
            norm_ax = (0, -2, -1)

        # Fully connected
        elif layer_type == "fc":
            h = h.flatten(2) if h.ndim > 2 else h
            _, dim = spec
            W = self.weight(self.rand_init(np.prod(in_dim), dim), gen_id('W'))
            z, output_size = T.dot(h, W), (dim,)
            norm_ax = (0,)
        else:
            raise ValueError("Unknown layer spec: %s" % layer_type)

        m = s = None
        is_normalizing = True
        if is_normalizing:
            keep_dims = True
            z_l = self.labeled(z)
            z_u = self.unlabeled(z)
            m = z_u.mean(norm_ax, keepdims=keep_dims)
            s = z_u.var(norm_ax, keepdims=keep_dims)

            m_l = z_l.mean(norm_ax, keepdims=keep_dims)
            s_l = z_l.var(norm_ax, keepdims=keep_dims)
            if path_name == 'clean':
                # Batch normalization estimates the mean and variance of
                # validation and test sets based on the training set
                # statistics. The following annotates the computation of
                # running average to the graph.
                m_l = self.annotate_bn(m_l, gen_id('bn'), 'mean', z_l.shape[0],
                                       output_size, norm_ax)
                s_l = self.annotate_bn(s_l, gen_id('bn'), 'var', z_l.shape[0],
                                       output_size, norm_ax)
            z = self.join(
                (z_l - m_l) / T.sqrt(s_l + np.float32(1e-10)),
                (z_u - m) / T.sqrt(s + np.float32(1e-10)))

        if noise_std > 0:
            z += self.noise_like(z) * noise_std

        # z for lateral connection
        z_lat = z
        b_init, c_init = 0.0, 1.0
        b_c_size = output_size[0]

        # Add bias
        if act_f != 'linear':
            z += self.bias(b_init * np.ones(b_c_size), gen_id('b'),
                           for_conv=len(output_size) > 1)

        if is_normalizing:
            # Add free parameter (gamma in original Batch Normalization paper)
            # if needed by the activation. For instance ReLU does't need one
            # and we only add it to softmax if hyperparameter top_c is set.
            if (act_f not in ['relu', 'leakyrelu', 'linear', 'softmax'] or
                    (act_f == 'softmax' and self.p.top_c is True)):
                c = self.weight(c_init * np.ones(b_c_size), gen_id('c'),
                                for_conv=len(output_size) > 1)
                z *= c

        h = self.apply_act(z, act_f)

        logger.info('  f%d: %s, %s,%s noise %.2f, params %s, dim %s -> %s' % (
            num, layer_type, act_f, ' BN,' if is_normalizing else '',
            noise_std, spec[1], in_dim, output_size))
        return output_size, z_lat, m, s, h

    def f_pool(self, x, spec, in_dim):
        layer_type, dims = spec
        num_filters = in_dim[0]
        if "globalmeanpool" == layer_type:
            y, output_size = global_meanpool_2d(x, num_filters)
            # scale the variance to match normal conv layers with xavier init
            y = y * np.float32(in_dim[-1]) * np.float32(np.sqrt(3))
        else:
            assert dims[0] != 1 or dims[1] != 1
            y, output_size = maxpool_2d(x, in_dim,
                                        poolsize=(dims[1], dims[1]),
                                        poolstride=(dims[0], dims[0]))
        return y, output_size

    def f_conv(self, x, spec, in_dim, weight_name):
        layer_type, dims = spec
        num_filters = dims[0]
        filter_size = (dims[1], dims[1])
        stride = (dims[2], dims[2])

        bm = 'full' if 'convf' in layer_type else 'valid'

        num_channels = in_dim[0]

        W = self.weight(self.rand_init_conv(
            (num_filters, num_channels) + filter_size), weight_name)

        if stride != (1, 1):
            f = GpuCorrMM(subsample=stride, border_mode=bm, pad=(0, 0))
            y = f(gpu_contiguous(x), gpu_contiguous(W))
        else:
            assert self.p.batch_size == self.p.valid_batch_size
            y = conv2d(x, W, image_shape=(2*self.p.batch_size, ) + in_dim,
                       filter_shape=((num_filters, num_channels) +
                                     filter_size), border_mode=bm)
        output_size = ((num_filters,) +
                       ConvOp.getOutputShape(in_dim[1:], filter_size,
                                             stride, bm))

        return y, output_size

    def g(self, z_lat, z_ver, in_dims, out_dims, l_type, num, fspec, top_g):
        f_layer_type, dims = fspec
        is_conv = f_layer_type is not None and ('conv' in f_layer_type or
                                                'pool' in f_layer_type)
        gen_id = lambda s: '_'.join(['g', str(num), s])

        in_dim = np.prod(dtype=floatX, a=in_dims)
        out_dim = np.prod(dtype=floatX, a=out_dims)
        num_filters = out_dims[0] if is_conv else out_dim

        if l_type[-1] in ['0']:
            g_type, u_type = l_type[:-1], l_type[-1]
        else:
            g_type, u_type = l_type, None

        # Mapping from layer above: u
        if u_type in ['0'] or z_ver is None:
            if z_ver is None and u_type not in ['0']:
                logger.warn('Decoder %d:%s without vertical input' %
                            (num, g_type))
            u = None
        else:
            if top_g:
                u = z_ver
            elif is_conv:
                u = self.g_deconv(z_ver, in_dims, out_dims, gen_id('W'), fspec)
            else:
                W = self.weight(self.rand_init(in_dim, out_dim), gen_id('W'))
                u = T.dot(z_ver, W)

        # Batch-normalize u
        if u is not None:
            norm_ax = (0,) if u.ndim <= 2 else (0, -2, -1)
            keep_dims = True
            u -= u.mean(norm_ax, keepdims=keep_dims)
            u /= T.sqrt(u.var(norm_ax, keepdims=keep_dims) +
                        np.float32(1e-10))

        # Define the g function
        if not is_conv:
            z_lat = z_lat.flatten(2)
        bi = lambda inits, name: self.bias(inits * np.ones(num_filters),
                                           gen_id(name), for_conv=is_conv)
        wi = lambda inits, name: self.weight(inits * np.ones(num_filters),
                                             gen_id(name), for_conv=is_conv)

        if g_type == '':
            z_est = None

        elif g_type == 'i':
            z_est = z_lat

        elif g_type in ['sig']:
            sigval = bi(0., 'c1') + wi(1., 'c2') * z_lat
            if u is not None:
                sigval += wi(0., 'c3') * u + wi(0., 'c4') * z_lat * u
            sigval = T.nnet.sigmoid(sigval)

            z_est = bi(0., 'a1') + wi(1., 'a2') * z_lat + wi(1., 'b1') * sigval
            if u is not None:
                z_est += wi(0., 'a3') * u + wi(0., 'a4') * z_lat * u

        elif g_type in ['lin']:
            a1 = wi(1.0, 'a1')
            b = bi(0.0, 'b')

            z_est = a1 * z_lat + b

        elif g_type in ['relu']:
            assert u is not None
            b = bi(0., 'b')
            x = u + b
            z_est = self.apply_act(x, 'relu')

        elif g_type in ['sigmoid']:
            assert u is not None
            b = bi(0., 'b')
            c = wi(1., 'c')
            z_est = self.apply_act((u + b) * c, 'sigmoid')

        elif g_type in ['comparison_g2']:
            # sig without the uz cross term
            sigval = bi(0., 'c1') + wi(1., 'c2') * z_lat
            if u is not None:
                sigval += wi(0., 'c3') * u
            sigval = T.nnet.sigmoid(sigval)

            z_est = bi(0., 'a1') + wi(1., 'a2') * z_lat + wi(1., 'b1') * sigval
            if u is not None:
                z_est += wi(0., 'a3') * u

        elif g_type in ['comparison_g3']:
            # sig without the sigmoid nonlinearity
            z_est = bi(0., 'a1') + wi(1., 'a2') * z_lat
            if u is not None:
                z_est += wi(0., 'a3') * u + wi(0., 'a4') * z_lat * u

        elif g_type in ['comparison_g4']:
            # No mixing between z_lat and u before final sum, otherwise similar
            # to sig
            def nonlin(inp, in_name='input', add_bias=True):
                w1 = wi(1., 'w1_%s' % in_name)
                b1 = bi(0., 'b1')
                w2 = wi(1., 'w2_%s' % in_name)
                b2 = bi(0., 'b2') if add_bias else 0
                w3 = wi(0., 'w3_%s' % in_name)
                return w2 * T.nnet.sigmoid(b1 + w1 * inp) + w3 * inp + b2

            z_est = nonlin(z_lat, 'lat') if u is None else \
                nonlin(z_lat, 'lat') + nonlin(u, 'ver', False)

        elif g_type in ['comparison_g5', 'gauss']:
            # Gaussian assumption on z: (z - mu) * v + mu
            if u is None:
                b1 = bi(0., 'b1')
                w1 = wi(1., 'w1')
                z_est = w1 * z_lat + b1
            else:
                a1 = bi(0., 'a1')
                a2 = wi(1., 'a2')
                a3 = bi(0., 'a3')
                a4 = bi(0., 'a4')
                a5 = bi(0., 'a5')

                a6 = bi(0., 'a6')
                a7 = wi(1., 'a7')
                a8 = bi(0., 'a8')
                a9 = bi(0., 'a9')
                a10 = bi(0., 'a10')

                mu = a1 * T.nnet.sigmoid(a2 * u + a3) + a4 * u + a5
                v = a6 * T.nnet.sigmoid(a7 * u + a8) + a9 * u + a10

                z_est = (z_lat - mu) * v + mu
        elif 'gauss_stable_v' in g_type:
            # Gaussian assumption on z: (z - mu) * v + mu
            if u is None:
                b1 = bi(0., 'b1')
                w1 = wi(1., 'w1')
                z_est = w1 * z_lat + b1
            elif z_lat is None:
                b1 = bi(0., 'b1')
                w1 = wi(1., 'w1')
                z_est = w1 * u + b1
            else:
                a1 = bi(0., 'a1')
                a2 = wi(1., 'a2')
                a3 = bi(0., 'a3')
                a4 = bi(0., 'a4')
                a5 = bi(0., 'a5')
                a6 = bi(0., 'a6')
                a7 = wi(1., 'a7')
                a8 = bi(0., 'a8')
                a9 = bi(0., 'a9')
                a10 = bi(0., 'a10')

                mu = a1 * T.nnet.sigmoid(a2 * u + a3) + a4 * u + a5
                v = a6 * T.nnet.sigmoid(a7 * u + a8) + a9 * u + a10
                v = T.nnet.sigmoid(v)

                z_est = (z_lat - mu) * v + mu
        else:
            raise NotImplementedError("unknown g type: %s" % str(g_type))

        # Reshape the output if z is for conv but u from fc layer
        if (z_est is not None and type(out_dims) == tuple and
                len(out_dims) > 1.0 and z_est.ndim < 4):
            z_est = z_est.reshape((z_est.shape[0],) + out_dims)

        return z_est

    def g_deconv(self, z_ver, in_dims, out_dims, weight_name, fspec):
        """ Inverse operation for each type of f used in convnets """
        f_type, f_dims = fspec
        assert z_ver is not None
        num_channels = in_dims[0] if in_dims is not None else None
        num_filters, width, height = out_dims[:3]

        if f_type in ['globalmeanpool']:
            u = T.addbroadcast(z_ver, 2, 3)
            assert in_dims[1] == 1 and in_dims[2] == 1, \
                "global pooling needs in_dims (1,1): %s" % str(in_dims)

        elif f_type in ['maxpool']:
            sh, str, size = z_ver.shape, f_dims[0], f_dims[1]
            assert str == size, "depooling requires stride == size"
            u = T.zeros((sh[0], sh[1], sh[2] * str, sh[3] * str),
                        dtype=z_ver.dtype)
            for x in xrange(str):
                for y in xrange(str):
                    u = T.set_subtensor(u[:, :, x::str, y::str], z_ver)
            u = u[:, :, :width, :height]

        elif f_type in ['convv', 'convf']:
            filter_size, str = (f_dims[1], f_dims[1]), f_dims[2]
            W_shape = (num_filters, num_channels) + filter_size
            W = self.weight(self.rand_init_conv(W_shape), weight_name)
            if str > 1:
                # upsample if strided version
                sh = z_ver.shape
                u = T.zeros((sh[0], sh[1], sh[2] * str, sh[3] * str),
                            dtype=z_ver.dtype)
                u = T.set_subtensor(u[:, :, ::str, ::str], z_ver)
            else:
                u = z_ver  # no strides, only deconv
            u = conv2d(u, W, filter_shape=W_shape,
                       border_mode='valid' if 'convf' in f_type else 'full')
            u = u[:, :, :width, :height]
        else:
            raise NotImplementedError('Layer %s has no convolutional decoder'
                                      % f_type)

        return u
