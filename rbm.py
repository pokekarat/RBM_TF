import errno  
import tensorflow as tf # 1.6.0
import numpy as np
import os


class RBM(object):

    """ Restricted Boltzmann Machine implementation using TensorFlow."""

    def __init__(self, num_visible, num_hidden, visible_unit_type='bin', main_dir='rbm', model_name='rbm_model',
                 gibbs_sampling_steps=1, learning_rate=0.01, batch_size=10, num_epochs=10, stddev=0.1, verbose=0):

        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.visible_unit_type = visible_unit_type
        self.main_dir = main_dir
        self.model_name = model_name
        self.gibbs_sampling_steps = gibbs_sampling_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.stddev = stddev
        self.verbose = verbose

        self.models_dir, self.data_dir, self.summary_dir = self._create_data_directories()
        self.model_path = self.models_dir + self.model_name

        self.W = None
        self.bh_ = None
        self.bv_ = None

        self.w_upd8 = None
        self.bh_upd8 = None
        self.bv_upd8 = None

        self.encode = None

        self.loss_function = None

        self.input_data = None
        self.hrand = None
        self.vrand = None
        self.validation_size = None

        self.tf_merged_summaries = None
        self.tf_summary_writer = None
        self.tf_session = None
        self.tf_saver = None

    def fit(self, train_set, validation_set=None, restore_previous_model=False):

        """ Fit the model to the training data."""

        if validation_set is not None:
            self.validation_size = validation_set.shape[0]

        # Building a model
        self._build_model()

        # Running tensorflow's operations
        with tf.Session() as self.tf_session:
            self._initialize_tf_utilities_and_ops()
            self._train_model(train_set, validation_set)
            self.tf_saver.save(self.tf_session, self.model_path)

    def _initialize_tf_utilities_and_ops(self):

        """ 
        Initialize TensorFlow operations: 
        - summary, 
        - init operations, 
        - saver, 
        - FileWriter.
        """

        #Merges all summaries collected in the default graph.
        self.tf_merged_summaries = tf.summary.merge_all()

        #initializes global variables in the graph.
        init_op = tf.global_variables_initializer()

        #Save and restore variables
        self.tf_saver = tf.train.Saver()
        self.tf_session.run(init_op)
        self.tf_saver.restore(self.tf_session, self.model_path)
        self.tf_summary_writer = tf.summary.FileWriter(self.summary_dir, self.tf_session.graph)

    def _train_model(self, train_set, validation_set):

        """ Train the model & validate errors in each step   """

        for i in range(self.num_epochs):
            self._run_train_step(train_set)

            if validation_set is not None:
                self._run_validation_error_and_summaries(i, validation_set)

    def _run_train_step(self, train_set):

        """ Run a training step. 
        A training step is made by randomly shuffling the training set,
        divide into batches and run the variable update nodes for each batch.
        """

        #Shuffling the traning set, to reduce variance, overfit loss and make sure generality.
        np.random.shuffle(train_set)

        #Divide input data into batches
        batches = [_ for _ in self.gen_batches(train_set,self.batch_size)]
        updates = [self.w_upd8, self.bh_upd8, self.bv_upd8]

        #Update each batch
        for batch in batches:
            self.tf_session.run(updates, feed_dict=self._create_feed_dict(batch))

    def gen_batches(self, data, batch_size):
        """ Divide input data into batches.
        """
        data = np.array(data)

        for i in range(0, data.shape[0], batch_size):
            yield data[i:i+batch_size]

    def _run_validation_error_and_summaries(self, epoch, validation_set):

        """ Run the summaries and error computation on the validation set. """

        result = self.tf_session.run([self.tf_merged_summaries, self.loss_function],
                                     feed_dict=self._create_feed_dict(validation_set))

        summary_str = result[0]
        err = result[1]

        self.tf_summary_writer.add_summary(summary_str, 1)

        if self.verbose == 1:
            print("Validation cost at step %s: %s" % (epoch, err))

    def _create_feed_dict(self, data):

        """ Create the dictionary of data to feed to TensorFlow's session during training.    """
        return {
            self.input_data: data,

            # Create an array of the given shape and populate it with random samples
            # from a uniform distribution over [0, 1).
            self.hrand: np.random.rand(data.shape[0], self.num_hidden),
            self.vrand: np.random.rand(data.shape[0], self.num_visible)
        }

    def _build_model(self):

        """ Build the Restricted Boltzmann Machine model in TensorFlow.
        """

        #Initialize variables
        self.input_data, self.hrand, self.vrand = self._create_placeholders()
        self.W, self.bh_, self.bv_ = self._create_variables()

        hprobs0, hstates0, vprobs, hprobs1, hstates1 = self.gibbs_sampling_step(self.input_data)
        positive = self.compute_positive_association(self.input_data, hprobs0, hstates0)

        nn_input = vprobs

        for step in range(self.gibbs_sampling_steps - 1):
            hprobs, hstates, vprobs, hprobs1, hstates1 = self.gibbs_sampling_step(nn_input)
            nn_input = vprobs

        negative = tf.matmul(tf.transpose(vprobs), hprobs1)

        self.encode = hprobs1  # encoded data, used by the transform method

        self.w_upd8 = self.W.assign_add(self.learning_rate * (positive - negative))
        self.bh_upd8 = self.bh_.assign_add(self.learning_rate * tf.reduce_mean(hprobs0 - hprobs1, 0))
        self.bv_upd8 = self.bv_.assign_add(self.learning_rate * tf.reduce_mean(self.input_data - vprobs, 0))

        self.loss_function = tf.sqrt(tf.reduce_mean(tf.square(self.input_data - vprobs)))
        _ = tf.summary.scalar("cost", self.loss_function)

    def _create_placeholders(self):

        """ Create the TensorFlow placeholders for the model.
        tf.placeholder is used to feed actual training examples.
        """

        x = tf.placeholder('float', [None, self.num_visible], name='x-input')
        hrand = tf.placeholder('float', [None, self.num_hidden], name='hrand')
        vrand = tf.placeholder('float', [None, self.num_visible], name='vrand')

        return x, hrand, vrand

    def _create_variables(self):

        """ Create the TensorFlow variables for the model.
            tf.Variable for trainable variables such as weights (W) and biases (B) for your model.
        """

        W = tf.Variable(tf.random_normal((self.num_visible, self.num_hidden), mean=0.0, stddev=0.01), name='weights')
        bh_ = tf.Variable(tf.zeros([self.num_hidden]), name='hidden-bias')
        bv_ = tf.Variable(tf.zeros([self.num_visible]), name='visible-bias')

        return W, bh_, bv_

    def gibbs_sampling_step(self, visible):

        """ Performs one step of gibbs sampling.
        """

        #Sample hidden value, given visible value
        hprobs, hstates = self.sample_hidden_from_visible(visible)
        #Sample visible value, given hidden value
        vprobs = self.sample_visible_from_hidden(hprobs)
        #Sample hidden value, given visible value
        hprobs1, hstates1 = self.sample_hidden_from_visible(vprobs)

        return hprobs, hstates, vprobs, hprobs1, hstates1

    def sample_hidden_from_visible(self, visible):

        """ Sample the hidden units from the visible units. 
        This is the Positive phase of the Contrastive Divergence algorithm.
        """

        hprobs = tf.nn.sigmoid(tf.matmul(visible, self.W) + self.bh_)

        # 0 or 1
        hstates = tf.nn.relu(tf.sign(hprobs - self.hrand))
       
        return hprobs, hstates

    def sample_visible_from_hidden(self, hidden):

        """ Sample the visible units from the hidden units.
        This is the Negative phase of the Contrastive Divergence algorithm.
        """

        visible_activation = tf.matmul(hidden, tf.transpose(self.W)) + self.bv_

        #binary or gaussian
        if self.visible_unit_type == 'bin':
            vprobs = tf.nn.sigmoid(visible_activation)

        elif self.visible_unit_type == 'gauss':
            vprobs = tf.truncated_normal((1, self.num_visible), mean=visible_activation, stddev=self.stddev)

        else:
            vprobs = None

        return vprobs

    def compute_positive_association(self, visible, hidden_probs, hidden_states):

        """ Compute positive associations between visible and hidden units. """

        if self.visible_unit_type == 'bin':
            positive = tf.matmul(tf.transpose(visible), hidden_states)

        elif self.visible_unit_type == 'gauss':
            positive = tf.matmul(tf.transpose(visible), hidden_probs)

        else:
            positive = None

        return positive

    def _create_data_directories(self):

        """ Create the three directories for storing respectively the models,
        the data generated by training and the TensorFlow's summaries.
        """

        self.main_dir = self.main_dir + '/' if self.main_dir[-1] != '/' else self.main_dir
        
        models_dir = 'models/'  # dir to save/restore models
        data_dir = 'data/'  # directory to store algorithm data
        summary_dir = 'logs/'  # directory to store tensorflow summaries
        
        models_dir  =   self.main_dir   +   'models/'
        data_dir = self.main_dir + 'data/'
        summary_dir = self.main_dir + 'logs'
        
        for d in [models_dir, data_dir, summary_dir]:
            if not os.path.isdir(d):
                try:
                    os.makedirs(d)
                except OSError as exc:  # Python >2.5
                    if exc.errno == errno.EEXIST and os.path.isdir(d):
                        pass
                    else:
                        raise
        return models_dir, data_dir, summary_dir

