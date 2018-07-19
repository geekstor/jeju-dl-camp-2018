import tensorflow as tf
from configuration import ConfigurationManager
from function_approximator import GeneralNetwork, Head


def get_vars_with_scope(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)


def get_copy_op(scope1, scope2):
    train_variables = get_vars_with_scope(scope1)
    target_variables = get_vars_with_scope(scope2)

    assign_ops = []
    for main_var, target_var in zip(sorted(train_variables, key=lambda x: x.name),
                                    sorted(target_variables, key=lambda x: x.name)):
        assign_ops.append(tf.assign(target_var, main_var))

    print("Copying Ops.:", len(assign_ops))

    return tf.group(*assign_ops)


def get_session(cfg_params: ConfigurationManager):
    required_params = []

    tf_params = cfg_params.parse_and_return_dictionary("TENSORFLOW",
                                                       required_params)

    config = tf.ConfigProto()

    if "ALLOW_GPU_GROWTH" not in tf_params or not tf_params["ALLOW_GPU_GROWTH"]:
        config.gpu_options.allow_growth = True

    if "INTRA_OP_PARALLELISM" in tf_params:
        config.intra_op_parallelism_threads = tf_params["INTRA_OP_PARALLELISM"]
    if "INTER_OP_PARALLELISM" in tf_params:
        config.inter_op_parallelism_threads = tf_params["INTER_OP_PARALLELISM"]

    return tf.Session(config=config)


def build_train_and_target_general_network_with_head(
    head, cfg_parser
):
    with tf.variable_scope("train_net"):
        train_network_base = GeneralNetwork(cfg_parser)
        train_network = head(
            cfg_parser, train_network_base)
    with tf.variable_scope("target_net"):
        target_network_base = GeneralNetwork(cfg_parser)
        target_network = head(
            cfg_parser, target_network_base)

    copy_operation = get_copy_op("train_net", "target_net")

    saver = tf.train.Saver(var_list=get_vars_with_scope("train_net") +
                           get_vars_with_scope("target_net"),
                           max_to_keep=100, keep_checkpoint_every_n_hours=1)

    return [train_network_base, train_network,
            target_network_base, target_network, copy_operation, saver]
