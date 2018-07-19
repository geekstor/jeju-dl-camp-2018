'''
File: optimizer.py

Usage:
    Features and methods for optimizing neural networks.
    Adam Optimizer, Gradient Clipping. 

'''


def get_optimizer(cfg_parser, loss_op, var_list, global_step):
    required_params = ["OPTIMIZER_TYPE"]
    optim_cfg = cfg_parser.parse_and_return_dictionary(
        "OPTIMIZER", required_params)

    gradient_clipping = None

    if "GRADIENT_CLIPPING" in optim_cfg:
        print("Found Gradient Clipping, will use", optim_cfg["GRADIENT_CLIPPING"],
              " for clipping norm.")
        gradient_clipping = optim_cfg["GRADIENT_CLIPPING"]

    if optim_cfg["OPTIMIZER_TYPE"] == "ADAM":
        required_params = ["LEARNING_RATE", "EPSILON"]
        adam_cfg = cfg_parser.parse_and_return_dictionary(
            "OPTIMIZER", required_params
        )

        from tensorflow import train
        optimizer = train.AdamOptimizer(learning_rate=adam_cfg["LEARNING_RATE"],
                            epsilon=adam_cfg["EPSILON"])

    else:
        raise NotImplementedError

    if gradient_clipping is None:
        return optimizer.minimize(loss_op, var_list=var_list, global_step=global_step)

    else:
        import tensorflow as tf

        gradients, variables = zip(*optimizer.compute_gradients(loss_op, var_list))
        gradients, _ = tf.clip_by_global_norm(gradients, gradient_clipping)
        return optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

        # gradients, variables = zip(*obj.optimizer.compute_gradients(obj.loss))
        # gradients, _ = tf.clip_by_global_norm(gradients, params.GRAD_NORM_CLIP)
        # #self.summ_op = tf.Print(tf.identity(self.delta_z), [tf.gradients(obj.loss, obj.q_dist)], summarize=22)
        # obj.train_step = obj.optimizer.apply_gradients(zip(gradients, variables))
