import tensorflow as tf
from tensorflow.contrib import rnn
from configuration.configuration import ConfigurationManager
# Used by Implicit Quantile Networks


def get_uniform_dist(psi: tf.Tensor, N_placeholder):
    tau_dist = tf.distributions.Uniform(low=0., high=1.)
    tau = tau_dist.sample(sample_shape=[tf.shape(psi)[0], N_placeholder])
    return tau


# Used for the single parameter functions (from the IQN paper)
def parse_eta(cfg_parser, psi: tf.Tensor):
    required_params = ["ADAPTIVE"]
    cfg = cfg_parser.parse_and_return_dictionary("POLICY.EXPECTATION", required_params)
    if cfg["ADAPTIVE"]:
        return tf.layers.dense(inputs=psi, units=1, activation=None)
    else:
        required_params = ["eta"]
        cfg = cfg_parser.parse_and_return_dictionary("POLICY.EXPECTATION",
                                                     required_params)
        return tf.constant(name="eta", value=cfg["eta"])


def get_tau_and_eta(cfg_parser, psi: tf.Tensor, N_placeholder):
    return get_uniform_dist(psi, N_placeholder), parse_eta(cfg_parser, psi)


def distorted_expectation(cfg_parser: ConfigurationManager, psi: tf.Tensor, N_placeholder):
    required_params = ["TYPE"]

    cfg = cfg_parser.parse_and_return_dictionary("POLICY.EXPECTATION", required_params)

    if cfg["TYPE"] == "ADAPTIVE_LSTM":
        return AdaptiveQuantileChoiceLSTM(cfg_parser, psi, N_placeholder)
    elif cfg["TYPE"] == "IDENTITY":
        return get_uniform_dist(psi, N_placeholder)
    else:
        tau, eta = get_tau_and_eta(cfg_parser, psi, N_placeholder)
        if cfg["TYPE"] == "CPW":
            return CPW(tau, eta)
        elif cfg["TYPE"] == "Wang":
            return Wang(tau, eta)
        elif cfg["TYPE"] == "Pow":
            return Pow(tau, eta)
        elif cfg["TYPE"] == "CVaR":
            return CVaR(tau, eta)
        else:
            raise NotImplementedError("Please check the Distorted Expectation Type!")


def AdaptiveQuantileChoiceLSTM(cfg_parser: ConfigurationManager, psi, N_placeholder):
    required_params = ["LSTM_UNITS"]

    cfg = cfg_parser.parse_and_return_dictionary("POLICY.EXPECTATION", required_params)

    initial_state = tf.layers.dense(inputs=psi, units=cfg["LSTM_UNITS"])

    cell = rnn.GRUCell(num_units=cfg["LSTM_UNITS"])

    tau_unif = tf.reshape(tf.expand_dims(get_uniform_dist(psi, N_placeholder), dim=-1),
                          [tf.shape(psi)[0], N_placeholder, 1])

    # No need to reset: https://stackoverflow.com/questions/38441589/
    # is-rnn-initial-state-reset-for-subsequent-mini-batches
    outputs, state = tf.nn.dynamic_rnn(dtype=tf.float32, cell=cell, inputs=tau_unif,
                                      time_major=False, initial_state=initial_state)
    output = tf.layers.dense(inputs=outputs, units=1)
    reshaped_out = tf.reshape(output, [tf.shape(psi)[0], N_placeholder])
    tau = tf.nn.sigmoid(reshaped_out)
    # tau = (reshaped_out - tf.reduce_min(reshaped_out)) / \
    #       (tf.reduce_max(reshaped_out) - tf.reduce_min(reshaped_out))
    return tau

def CPW(tau, eta):
    eta = tf.clip_by_value(tf.nn.sigmoid(eta), 1e-1, 1 - 1e-1)
    tau_to_power_eta = tf.pow(tau, eta)
    beta_tau = tau_to_power_eta / tf.pow(tau_to_power_eta +
                                         tf.pow((1 - tau), eta), 1. / eta)
    return beta_tau


def Wang(tau, eta):
    eta = tf.Print(eta, [tf.reduce_mean(tau[0]), tf.reduce_mean(eta[0])], summarize=10)
    unit_normal = tf.distributions.Normal(loc=0., scale=1.)
    beta_tau = unit_normal.cdf(unit_normal.quantile(tau) + eta)
    return beta_tau


def Pow(tau, eta):
    power = 1. / (1 + abs(eta))
    beta_tau = tf.cond(eta >= 0, true_fn=tf.pow(tau, power),
                       false_fn=1 - tf.pow(1 - tau, power))
    return beta_tau


def CVaR(tau, eta):
    eta = tf.nn.sigmoid(eta)
    beta_tau = tau * eta
    return beta_tau

# from scipy.special import comb, factorial
# def Norm(tau, eta):
#     s = 0
#     for k in range(0, eta + 1):
#         s += eta/(2 * factorial(eta - 1)) * comb(eta, k) * np.power(-1, k) * \
#             np.power(eta * tau - k, eta - 1) * \
#             np.sign(eta * tau - k)
#     s = (s - np.min(s)) / (np.max(s) - np.min(s))
#     return s

# Requires integer eta and looping...