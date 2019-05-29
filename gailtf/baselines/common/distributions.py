import tensorflow as tf
import numpy as np
from gailtf.baselines.common import tf_util as U
from tensorflow.python.ops import math_ops


def linear(input_tensor, scope, n_hidden, *, init_scale=1.0, init_bias=0.0):
    """
    Creates a fully connected layer for TensorFlow

    :param input_tensor: (TensorFlow Tensor) The input tensor for the fully connected layer
    :param scope: (str) The TensorFlow variable scope
    :param n_hidden: (int) The number of hidden neurons
    :param init_scale: (int) The initialization scale
    :param init_bias: (int) The initialization offset bias
    :return: (TensorFlow Tensor) fully connected layer
    """
    with tf.variable_scope(scope):
        n_input = input_tensor.get_shape()[1].value
        weight = tf.get_variable("w", [n_input, n_hidden], initializer=ortho_init(init_scale))
        bias = tf.get_variable("b", [n_hidden], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(input_tensor, weight) + bias


def ortho_init(scale=1.0):
    """
    Orthogonal initialization for the policy weights

    :param scale: (float) Scaling factor for the weights.
    :return: (function) an initialization function for the weights
    """

    # _ortho_init(shape, dtype, partition_info=None)
    def _ortho_init(shape, *_, **_kwargs):
        """Intialize weights as Orthogonal matrix.

        Orthogonal matrix initialization [1]_. For n-dimensional shapes where
        n > 2, the n-1 trailing axes are flattened. For convolutional layers, this
        corresponds to the fan-in, so this makes the initialization usable for
        both dense and convolutional layers.

        References
        ----------
        .. [1] Saxe, Andrew M., James L. McClelland, and Surya Ganguli.
               "Exact solutions to the nonlinear dynamics of learning in deep
               linear
        """
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        gaussian_noise = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(gaussian_noise, full_matrices=False)
        weights = u if u.shape == flat_shape else v  # pick the one with the correct shape
        weights = weights.reshape(shape)
        return (scale * weights[:shape[0], :shape[1]]).astype(np.float32)

    return _ortho_init


class ProbabilityDistribution(object):
    """
    A particular probability distribution
    """

    def flatparam(self):
        """
        Return the direct probabilities

        :return: ([float]) the probabilites
        """
        raise NotImplementedError

    def mode(self):
        """
        Returns the probability

        :return: (Tensorflow Tensor) the deterministic action
        """
        raise NotImplementedError

    def neglogp(self, x):
        """
        returns the of the negative log likelihood

        :param x: (str) the labels of each index
        :return: ([float]) The negative log likelihood of the distribution
        """
        # Usually it's easier to define the negative logprob
        raise NotImplementedError

    def kl(self, other):
        """
        Calculates the Kullback-Leiber divergence from the given probabilty distribution

        :param other: ([float]) the distibution to compare with
        :return: (float) the KL divergence of the two distributions
        """
        raise NotImplementedError

    def entropy(self):
        """
        Returns shannon's entropy of the probability

        :return: (float) the entropy
        """
        raise NotImplementedError

    def sample(self):
        """
        returns a sample from the probabilty distribution

        :return: (Tensorflow Tensor) the stochastic action
        """
        raise NotImplementedError

    def logp(self, x):
        """
        returns the of the log likelihood

        :param x: (str) the labels of each index
        :return: ([float]) The log likelihood of the distribution
        """
        return - self.neglogp(x)

    def get_shape(self):
        return self.flatparam().shape

    @property
    def shape(self):
        return self.get_shape()


class ProbabilityDistributionType(object):
    """
    Parametrized family of probability distributions
    """

    def probability_distribution_class(self):
        """
        returns the ProbabilityDistribution class of this type

        :return: (Type ProbabilityDistribution) the probability distribution class associated
        """
        raise NotImplementedError

    def proba_distribution_from_flat(self, flat):
        """
        Returns the probability distribution from flat probabilities
        flat: flattened vector of parameters of probability distribution

        :param flat: ([float]) the flat probabilities
        :return: (ProbabilityDistribution) the instance of the ProbabilityDistribution associated
        """
        return self.probability_distribution_class()(flat)

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        """
        returns the probability distribution from latent values

        :param pi_latent_vector: ([float]) the latent pi values
        :param vf_latent_vector: ([float]) the latent vf values
        :param init_scale: (float) the inital scale of the distribution
        :param init_bias: (float) the inital bias of the distribution
        :return: (ProbabilityDistribution) the instance of the ProbabilityDistribution associated
        """
        raise NotImplementedError

    def param_shape(self):
        """
        returns the shape of the input parameters

        :return: ([int]) the shape
        """
        raise NotImplementedError

    def sample_shape(self):
        """
        returns the shape of the sampling

        :return: ([int]) the shape
        """
        raise NotImplementedError

    def sample_dtype(self):
        """
        returns the type of the sampling

        :return: (type) the type
        """
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        """
        returns the TensorFlow placeholder for the input parameters

        :param prepend_shape: ([int]) the prepend shape
        :param name: (str) the placeholder name
        :return: (TensorFlow Tensor) the placeholder
        """
        return tf.placeholder(dtype=tf.float32, shape=prepend_shape + self.param_shape(), name=name)

    def sample_placeholder(self, prepend_shape, name=None):
        """
        returns the TensorFlow placeholder for the sampling

        :param prepend_shape: ([int]) the prepend shape
        :param name: (str) the placeholder name
        :return: (TensorFlow Tensor) the placeholder
        """
        return tf.placeholder(dtype=self.sample_dtype(), shape=prepend_shape + self.sample_shape(), name=name)

    def __eq__(self, other):
        return (type(self) == type(other)) and (self.__dict__ == other.__dict__)


class CategoricalProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, n_cat):
        """
        The probability distribution type for categorical input

        :param n_cat: (int) the number of categories
        """
        self.n_cat = n_cat

    def probability_distribution_class(self):
        return CategoricalProbabilityDistribution

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        pdparam = linear(pi_latent_vector, 'pi', self.n_cat, init_scale=init_scale, init_bias=init_bias)
        q_values = linear(vf_latent_vector, 'q', self.n_cat, init_scale=init_scale, init_bias=init_bias)
        return self.proba_distribution_from_flat(pdparam), pdparam, q_values

    def param_shape(self):
        return [self.n_cat]

    def sample_shape(self):
        return []

    def sample_dtype(self):
        return tf.int32


class MultiCategoricalProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, n_vec):
        """
        The probability distribution type for multiple categorical input

        :param n_vec: ([int]) the vectors
        """
        # Cast the variable because tf does not allow uint32
        self.n_vec = n_vec.astype(np.int32)
        # Check that the cast was valid
        assert (self.n_vec > 0).all(), "Casting uint32 to int32 was invalid"

    def probability_distribution_class(self):
        return MultiCategoricalProbabilityDistribution

    def proba_distribution_from_flat(self, flat):
        return MultiCategoricalProbabilityDistribution(self.n_vec, flat)

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        pdparam = linear(pi_latent_vector, 'pi', sum(self.n_vec), init_scale=init_scale, init_bias=init_bias)
        q_values = linear(vf_latent_vector, 'q', sum(self.n_vec), init_scale=init_scale, init_bias=init_bias)
        return self.proba_distribution_from_flat(pdparam), pdparam, q_values

    def param_shape(self):
        return [sum(self.n_vec)]

    def sample_shape(self):
        return [len(self.n_vec)]

    def sample_dtype(self):
        return tf.int32


class DiagGaussianProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, size):
        """
        The probability distribution type for multivariate gaussian input

        :param size: (int) the number of dimensions of the multivariate gaussian
        """
        self.size = size

    def probability_distribution_class(self):
        return DiagGaussianProbabilityDistribution

    def proba_distribution_from_flat(self, flat):
        """
        returns the probability distribution from flat probabilities

        :param flat: ([float]) the flat probabilities
        :return: (ProbabilityDistribution) the instance of the ProbabilityDistribution associated
        """
        return self.probability_distribution_class()(flat)

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        mean = linear(pi_latent_vector, 'pi', self.size, init_scale=init_scale, init_bias=init_bias)
        logstd = tf.get_variable(name='pi/logstd', shape=[1, self.size], initializer=tf.zeros_initializer())
        pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        q_values = linear(vf_latent_vector, 'q', self.size, init_scale=init_scale, init_bias=init_bias)
        return self.proba_distribution_from_flat(pdparam), mean, q_values

    def param_shape(self):
        return [2 * self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.float32


class BernoulliProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, size):
        """
        The probability distribution type for bernoulli input

        :param size: (int) the number of dimensions of the bernoulli distribution
        """
        self.size = size

    def probability_distribution_class(self):
        return BernoulliProbabilityDistribution

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        pdparam = linear(pi_latent_vector, 'pi', self.size, init_scale=init_scale, init_bias=init_bias)
        q_values = linear(vf_latent_vector, 'q', self.size, init_scale=init_scale, init_bias=init_bias)
        return self.proba_distribution_from_flat(pdparam), pdparam, q_values

    def param_shape(self):
        return [self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.int32


# WRONG SECOND DERIVATIVES
# class CategoricalPd(Pd):
#     def __init__(self, logits):
#         self.logits = logits
#         self.ps = tf.nn.softmax(logits)
#     @classmethod
#     def fromflat(cls, flat):
#         return cls(flat)
#     def flatparam(self):
#         return self.logits
#     def mode(self):
#         return U.argmax(self.logits, axis=-1)
#     def logp(self, x):
#         return -tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, x)
#     def kl(self, other):
#         return tf.nn.softmax_cross_entropy_with_logits(other.logits, self.ps) \
#                 - tf.nn.softmax_cross_entropy_with_logits(self.logits, self.ps)
#     def entropy(self):
#         return tf.nn.softmax_cross_entropy_with_logits(self.logits, self.ps)
#     def sample(self):
#         u = tf.random_uniform(tf.shape(self.logits))
#         return U.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)

class CategoricalProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, logits):
        """
        Probability distributions from categorical input

        :param logits: ([float]) the categorical logits input
        """
        self.logits = logits

    def flatparam(self):
        return self.logits

    def mode(self):
        return U.argmax(self.logits, axis=-1)

    @property
    def mean(self):
        return tf.nn.softmax(self.logits)

    def neglogp(self, x):
        # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        one_hot_actions = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        return tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits,
            labels=one_hot_actions)

    def kl(self, other):
        a0 = self.logits - U.max(self.logits, axis=-1, keepdims=True)
        a1 = other.logits - U.max(other.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = U.sum(ea0, axis=-1, keepdims=True)
        z1 = U.sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return U.sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)

    def em(self, other):
        return tf.reduce_mean(self.logits) - tf.reduce_mean(other.logits)

    def entropy(self):
        a0 = self.logits - U.max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = U.sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return U.sum(p0 * (tf.log(z0) - a0), axis=-1)

    def sample(self):
        u = tf.random_uniform(tf.shape(self.logits))
        return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new logits values

        :param flat: ([float]) the categorical logits input
        :return: (ProbabilityDistribution) the instance from the given categorical input
        """
        return cls(flat)


class MultiCategoricalProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, nvec, flat):
        """
        Probability distributions from multicategorical input

        :param nvec: ([int]) the sizes of the different categorical inputs
        :param flat: ([float]) the categorical logits input
        """
        self.flat = flat
        self.categoricals = list(map(CategoricalProbabilityDistribution, tf.split(flat, nvec, axis=-1)))

    def flatparam(self):
        return self.flat

    def mode(self):
        return tf.cast(tf.stack([p.mode() for p in self.categoricals], axis=-1), tf.int32)

    def neglogp(self, x):
        return tf.add_n([p.neglogp(px) for p, px in zip(self.categoricals, tf.unstack(x, axis=-1))])

    def kl(self, other):
        return tf.add_n([p.kl(q) for p, q in zip(self.categoricals, other.categoricals)])

    def em(self, other):
        return tf.add_n([
            p.em(q) for p, q in zip(self.categoricals, other.categoricals)
        ])

    def entropy(self):
        return tf.add_n([p.entropy() for p in self.categoricals])

    def sample(self):
        return tf.cast(tf.stack([p.sample() for p in self.categoricals], axis=-1), tf.int32)

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new logits values

        :param flat: ([float]) the multi categorical logits input
        :return: (ProbabilityDistribution) the instance from the given multi categorical input
        """
        raise NotImplementedError


class DiagGaussianProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, flat):
        """
        Probability distributions from multivariate gaussian input

        :param flat: ([float]) the multivariate gaussian input data
        """
        self.flat = flat
        mean, logstd = tf.split(axis=len(flat.shape) - 1, num_or_size_splits=2, value=flat)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)

    def flatparam(self):
        return self.flat

    def mode(self):
        # Bounds are taken into account outside this class (during training only)
        return self.mean

    def neglogp(self, x):
        return 0.5 * U.sum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
               + U.sum(self.logstd, axis=-1)

    def kl(self, other):
        assert isinstance(other, DiagGaussianProbabilityDistribution)
        return U.sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) / (
                2.0 * tf.square(other.std)) - 0.5, axis=-1)

    def entropy(self):
        return U.sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new multivariate gaussian input

        :param flat: ([float]) the multivariate gaussian input data
        :return: (ProbabilityDistribution) the instance from the given multivariate gaussian input data
        """
        return cls(flat)


class BernoulliProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, logits):
        """
        Probability distributions from bernoulli input

        :param logits: ([float]) the bernoulli input data
        """
        self.logits = logits
        self.probabilities = tf.sigmoid(logits)

    def flatparam(self):
        return self.logits

    @property
    def mean(self):
        return self.probabilities

    def mode(self):
        return tf.round(self.probabilities)

    def neglogp(self, x):
        return U.sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.to_float(x)), axis=-1)

    def kl(self, other):
        return U.sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=other.logits, labels=self.probabilities),
                     axis=-1) - U.sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.probabilities), axis=-1)

    def entropy(self):
        return U.sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.probabilities), axis=-1)

    def sample(self):
        samples_from_uniform = tf.random_uniform(tf.shape(self.probabilities))
        return tf.to_float(math_ops.less(samples_from_uniform, self.probabilities))

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new bernoulli input

        :param flat: ([float]) the bernoulli input data
        :return: (ProbabilityDistribution) the instance from the given bernoulli input data
        """
        return cls(flat)


def make_proba_dist_type(ac_space):
    from gym import spaces
    """
    return an instance of ProbabilityDistributionType for the correct type of action space

    :param ac_space: (Gym Space) the input action space
    :return: (ProbabilityDistributionType) the approriate instance of a ProbabilityDistributionType
    """
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1, "Error: the action space must be a vector"
        return DiagGaussianProbabilityDistributionType(ac_space.shape[0])
    elif isinstance(ac_space, spaces.Discrete):
        return CategoricalProbabilityDistributionType(ac_space.n)
    elif isinstance(ac_space, spaces.MultiDiscrete):
        return MultiCategoricalProbabilityDistributionType(ac_space.nvec)
    elif isinstance(ac_space, spaces.MultiBinary):
        return BernoulliProbabilityDistributionType(ac_space.n)
    else:
        raise NotImplementedError("Error: probability distribution, not implemented for action space of type {}."
                                  .format(type(ac_space)) +
                                  " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary.")


def shape_el(tensor, index):
    """
    get the shape of a TensorFlow Tensor element

    :param tensor: (TensorFlow Tensor) the input tensor
    :param index: (int) the element
    :return: ([int]) the shape
    """
    maybe = tensor.get_shape()[index]
    if maybe is not None:
        return maybe
    else:
        return tf.shape(tensor)[index]


@U.in_session
def test_probtypes():
    np.random.seed(0)

    pdparam_diag_gauss = np.array([-.2, .3, .4, -.5, .1, -.5, .1, 0.8])
    diag_gauss = DiagGaussianProbabilityDistributionType(pdparam_diag_gauss.size // 2)  # pylint: disable=E1101
    validate_probtype(diag_gauss, pdparam_diag_gauss)

    pdparam_categorical = np.array([-.2, .3, .5])
    categorical = CategoricalProbabilityDistributionType(pdparam_categorical.size)  # pylint: disable=E1101
    validate_probtype(categorical, pdparam_categorical)

    pdparam_bernoulli = np.array([-.2, .3, .5])
    bernoulli = BernoulliProbabilityDistributionType(pdparam_bernoulli.size)  # pylint: disable=E1101
    validate_probtype(bernoulli, pdparam_bernoulli)


def validate_probtype(probtype, pdparam):
    N = 100000
    # Check to see if mean negative log likelihood == differential entropy
    Mval = np.repeat(pdparam[None, :], N, axis=0)
    M = probtype.param_placeholder([N])
    X = probtype.sample_placeholder([N])
    pd = probtype.probability_distribution_class()(M)
    calcloglik = U.function([X, M], pd.logp(X))
    calcent = U.function([M], pd.entropy())
    Xval = U.eval(pd.sample(), feed_dict={M: Mval})
    logliks = calcloglik(Xval, Mval)
    entval_ll = - logliks.mean()  # pylint: disable=E1101
    entval_ll_stderr = logliks.std() / np.sqrt(N)  # pylint: disable=E1101
    entval = calcent(Mval).mean()  # pylint: disable=E1101
    assert np.abs(entval - entval_ll) < 3 * entval_ll_stderr  # within 3 sigmas

    # Check to see if kldiv[p,q] = - ent[p] - E_p[log q]
    M2 = probtype.param_placeholder([N])
    pd2 = probtype.probability_distribution_class()(M2)
    q = pdparam + np.random.randn(pdparam.size) * 0.1
    Mval2 = np.repeat(q[None, :], N, axis=0)
    calckl = U.function([M, M2], pd.kl(pd2))
    klval = calckl(Mval, Mval2).mean()  # pylint: disable=E1101
    logliks = calcloglik(Xval, Mval2)
    klval_ll = - entval - logliks.mean()  # pylint: disable=E1101
    klval_ll_stderr = logliks.std() / np.sqrt(N)  # pylint: disable=E1101
    assert np.abs(klval - klval_ll) < 3 * klval_ll_stderr  # within 3 sigmas
    print('ok on', probtype, pdparam)
