import tensorflow as tf
from configuration import ConfigurationManager


def get_vars_with_scope(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)


def get_copy_op(scope1, scope2):
    train_variables = get_vars_with_scope(scope1)
    target_variables = get_vars_with_scope(scope2)

    assign_ops = []
    for main_var, target_var in zip(sorted(train_variables, key=lambda x: x.name),
                                    sorted(target_variables, key=lambda x: x.name)):
        assign_ops.append(tf.assign(target_var, main_var))

    print(len(assign_ops))

    return tf.group(*assign_ops)


def build_session(cfg_params: ConfigurationManager):
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
