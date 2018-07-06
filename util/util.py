import tensorflow as tf


def get_copy_op(scope1, scope2):
    train_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope1)
    target_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope2)

    assign_ops = []
    for main_var, target_var in zip(sorted(train_variables, key=lambda x: x.name),
                                    sorted(target_variables, key=lambda x: x.name)):
        assign_ops.append(tf.assign(target_var, main_var))

    print(len(assign_ops))

    return tf.group(*assign_ops)