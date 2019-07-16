from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import Dense, Embedding, GRU, Input, LSTM, TimeDistributed
from keras.models import Model
from keras.optimizers import RMSprop, Adam
import numpy as np
import os


#TODO: Rethink design of class. Making Input, Embedding, Recurrent, Output properties may make more sense.
#TODO: Write code for running in generate mode.
class WordRNNLM:
    """
        Base Class for Word-Level Recurrent Language Model
    """
    def __init__(self,
                 max_seq_len,
                 word_vocab_size,
                 save_dir,
                 config):
        ### Protected Attributes

        # Input Attributes
        self._max_seq_len = max_seq_len
        self._word_vocab_size = word_vocab_size
        self._word_embedding_dim = config['embedding']['word']['dim']

        # Recurrent Attributes
        self._recurrent_type = config['recurrent']['type']
        self._recurrent_cell_size = config['recurrent']['cell_size']
        self._num_recurrent_layers = config['recurrent']['num_layers']
        self._recurrent_layer_input_dropout = config['recurrent']['dropout']
        self._recurrent_layer_recurrent_dropout = config['recurrent']['recurrent_dropout']

        # Optimization Attributes
        self._optimizer_type = config['training']['optimizer_type']
        self._lr = config['training']['lr']
        self._lr_decay = config['training']['lr_decay']
        self._clipnorm = config['training']['clipnorm']
        self._epochs = config['training']['epochs']
        self._batch_size = config['training']['batch_size']
        self._optimizer = self._get_optimizer()

        # Meta-data Attributes
        self._path = save_dir
        if not os.path.exists(self._path):
            os.makedirs(self._path)
        self._name = 'word_{}_rnnlm'.format(self._recurrent_type)

        # Internal LM object
        self._model = None

    # TODO: Support Learning Rate Schedules
    def _get_optimizer(self):
        """ Returns optimizer object.

        Instantiates and returns a Keras Optimizer. Currently only RMSPRop and Adam are supported. Currently
        initialization with given learning rate, learning rate decay and norm clipping values supported.

        Variable learning rate schedules NOT supported and can be done through keras callbacks 'LearningRateScheduler'
        and 'EarlyStopping'.
        """
        if self._optimizer_type == 'rmsprop':
            return RMSprop(lr=self._lr, decay=self._lr_decay, clipnorm=self._clipnorm)
        else:
            return Adam(lr=self._lr, decay=self._lr_decay, clipnorm=self._clipnorm)

    # TODO: Support Tied Input and Output Weights
    def _get_word_representation(self):
        """ Returns word input variables and word embedding layer.

        Tied Input and Output embeddings have been shown to improve LM performance (https://arxiv.org/abs/1608.05859)
        but is not trivially supported in Keras. https://stackoverflow.com/a/50892512 seems the most plausible solution
        but will require updates to model saving to also save custom layers).
        """
        word_input = Input(shape=(self._max_seq_len,), dtype='int32', name='word_input')
        word_embedding_layer = Embedding(output_dim=self._word_embedding_dim,
                                         input_dim=self._word_vocab_size,
                                         input_length=self._max_seq_len,
                                         trainable=True,
                                         mask_zero=True,
                                         name='word_embedding'
                                         )(word_input)
        inputs = [word_input]
        return inputs, word_embedding_layer

    def _get_embedding_layer(self):
        """ Returns list of input variables and final pre-recurrent layer.
        """
        return self._get_word_representation()

    # TODO: Support layer normalization after input / recurrent layers
    # TODO: Support Sampled Softmax / Noise-Contrastive Estimation
    # TODO: Support Recurrent Layer Weight Initialization
    def _build_model(self):
        """Builds and compiles Language Model.

        Layer normalization is not trivially supported in Keras but 3rd party solutions like
        https://pypi.org/project/keras-layer-normalization/ may be worth trying.

        Similary NCE is only available in TF and PyTorch. For Keras, https://github.com/eggie5/NCE-loss maybe viable.
        """
        inputs, embedding_layer = self._get_embedding_layer()

        recurrent_layer = None
        if self._recurrent_type == 'lstm':
            recurrent_layer = LSTM(units=self._recurrent_cell_size,
                                   dropout=self._recurrent_layer_input_dropout,
                                   recurrent_dropout=self._recurrent_layer_recurrent_dropout,
                                   return_sequences=True,
                                   name='lstm0')(embedding_layer)

            for i in range(1, self._num_recurrent_layers):
                recurrent_layer = LSTM(units=self._recurrent_cell_size,
                                       dropout=self._recurrent_layer_input_dropout,
                                       recurrent_dropout=self._recurrent_layer_recurrent_dropout,
                                       return_sequences=True, name='lstm' + str(i))(recurrent_layer)

        elif self._recurrent_type == 'gru':
            recurrent_layer = GRU(units=self._recurrent_cell_size,
                                  dropout=self._recurrent_layer_input_dropout,
                                  recurrent_dropout=self._recurrent_layer_recurrent_dropout,
                                  return_sequences=True, name='gru0')(embedding_layer)

            for i in range(1, self._num_recurrent_layers):
                recurrent_layer = GRU(units=self._recurrent_cell_size,
                                      dropout=self._recurrent_layer_input_dropout,
                                      recurrent_dropout=self._recurrent_layer_recurrent_dropout,
                                      return_sequences=True, name='gru' + str(i))(recurrent_layer)

        output_layer = TimeDistributed(Dense(self._word_vocab_size,
                                             activation='softmax'),
                                       name='output')(recurrent_layer)

        self._model = Model(inputs=inputs, output=output_layer)
        self._model.compile(optimizer=self._optimizer, loss='sparse_categorical_crossentropy')
        print(self._model.summary())

    # TODO: Move to fit_generator
    def train(self, train_x, train_y, val_x, val_y):
        csv_logger = CSVLogger('{0}/{1}_training.log'.format(self._path,
                                                             self._name))

        checkpointer = ModelCheckpoint(filepath='{0}/{1}.hdf5'.format(self._path,
                                                                      self._name),
                                       monitor='val_loss', verbose=1, save_best_only=True)
        self._build_model()
        history = self._model.fit(x=train_x,
                                  y=train_y,
                                  batch_size=self._batch_size,
                                  epochs=self._epochs,
                                  validation_data=(val_x, val_y),
                                  callbacks=[csv_logger, checkpointer])
        print(history.history)
        return history

    def evaluate_perplexity(self, x, y_true):
        loss = self._model.evaluate(x=x,
                                    y=y_true)
        return np.exp(loss)

    def predict(self, x, true_y):
        pred_y = self._model.predict(x=x)
        label_probabilities = []
        for sent_id, sent in enumerate(true_y):
            sent_prob_list = []
            for word_pos, word  in enumerate(sent):
                word_idx = word[0]

                if word_idx == 0:
                    # EOS
                    label_probabilities.append(sent_prob_list.copy())
                    break
                else:
                    prob = pred_y[sent_id][word_pos][word_idx]
                    sent_prob_list.append((word_idx, prob))
            sent_prob_list.clear()

        return label_probabilities
