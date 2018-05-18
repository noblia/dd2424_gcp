import tensorflow as tf


class Network:
    def __init__(self, eta=0.01, n_epochs=120, n_batch=500):
        self.eta = eta
        self.n_epochs = n_epochs
        self.n_batch = n_batch
        self.boundaries = [60, 100]
        self.values = [1e-2, 1e-3, 1e-4]
        self.cnn_classifier = self.init_estimator()

    def cnn_model_fn(self, features, labels, mode):
        input_layer = tf.reshape(features["x"], [-1, 27, 27, 1])

        conv1 = tf.layers.conv2d(inputs = input_layer,
                                 filters = 36,
                                 kernel_size = [4, 4],
                                 activation = tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(inputs=pool1, filters=48, kernel_size=[3, 3], activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 48])
        dense = tf.layers.dense(inputs=pool2_flat, units=512, activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)
        dense2 = tf.layers.dense(inputs=dropout, units=512, activation=tf.nn.relu)
        dropout2 = tf.layers.dropout(inputs=dense2, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)
        logits = tf.layers.dense(inputs=dropout2, units=4)

        predictions = {
            "classes": tf.argmax(input = logits, axis = 1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.piecewise_constant(global_step, boundaries = self.boundaries, values=self.values)
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        preds = predictions["classes"]
        eval_metric_ops = {
            'accuracy' : tf.metrics.accuracy(labels, predictions["classes"]),
            'precision' : tf.metrics.precision(labels, predictions["classes"]),
            'recall' : tf.metrics.recall(labels, predictions["classes"])
        }
        return tf.estimator.EstimatorSpec(mode = mode, loss=loss, eval_metric_ops=eval_metric_ops)

    def init_estimator(self):
        return tf.estimator.Estimator(model_fn=self.cnn_model_fn)

    def set_logging_hook(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        tensors_to_log = {"probabilities": "softmax_tensor"}
        return tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    def train_network(self, features, labels):
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": features},
            y=labels,
            batch_size=self.n_batch,
            num_epochs=self.n_epochs,
            shuffle=True
        )
        logging_hook = self.set_logging_hook()
        self.cnn_classifier.train(input_fn = train_input_fn, hooks=[logging_hook])

    def eval_network(self, features, labels):
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": features},
            y=labels,
            num_epochs=1,
            shuffle=False
        )
        return self.cnn_classifier.evaluate(input_fn=eval_input_fn)
