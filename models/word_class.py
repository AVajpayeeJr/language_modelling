from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import Activation, Dense, Embedding, GRU, Input, LSTM, Multiply, TimeDistributed
from keras.models import Model
from models.word import WordRNNLM


class WordClassRNNLM(WordRNNLM):
    def __init__(self,
                 max_seq_len,
                 word_vocab_size,
                 class_vocab_size,
                 class_embedding_weights,
                 save_dir,
                 config):
        super().__init__(max_seq_len=max_seq_len, word_vocab_size=word_vocab_size,
                         save_dir=save_dir, config=config)
        self._class_vocab_size = class_vocab_size
        self._class_embedding_weights = class_embedding_weights

    def _build_model(self):
        """
        """
        word_inputs, word_embedding_layer = self._get_embedding_layer()

        recurrent_layer = None
        if self._recurrent_type == 'lstm':
            recurrent_layer = LSTM(units=self._recurrent_cell_size,
                                   dropout=self._recurrent_layer_input_dropout,
                                   recurrent_dropout=self._recurrent_layer_recurrent_dropout,
                                   return_sequences=True,
                                   name='lstm0')(word_embedding_layer)

            for i in range(1, self._num_recurrent_layers):
                recurrent_layer = LSTM(units=self._recurrent_cell_size,
                                       dropout=self._recurrent_layer_input_dropout,
                                       recurrent_dropout=self._recurrent_layer_recurrent_dropout,
                                       return_sequences=True, name='lstm' + str(i))(recurrent_layer)

        elif self._recurrent_type == 'gru':
            recurrent_layer = GRU(units=self._recurrent_cell_size,
                                  dropout=self._recurrent_layer_input_dropout,
                                  recurrent_dropout=self._recurrent_layer_recurrent_dropout,
                                  return_sequences=True, name='gru0')(word_embedding_layer)

            for i in range(1, self._num_recurrent_layers):
                recurrent_layer = GRU(units=self._recurrent_cell_size,
                                      dropout=self._recurrent_layer_input_dropout,
                                      recurrent_dropout=self._recurrent_layer_recurrent_dropout,
                                      return_sequences=True, name='gru' + str(i))(recurrent_layer)

        class_prob_output_layer = TimeDistributed(Dense(self._class_vocab_size,
                                                        activation='softmax'),
                                                  name='class_probability')(recurrent_layer)

        class_inputs = Input(shape=(self._max_seq_len,), dtype='int32', name='class_input')

        # Simulating softmax over restricted vocab (based on input class)
        class_embedding_layer = Embedding(output_dim=self._word_vocab_size,
                                          input_dim=self._class_vocab_size,
                                          input_length=self._max_seq_len,
                                          weights=[self._class_embedding_weights],
                                          trainable=False,
                                          mask_zero=True,
                                          name='class_word_membership'
                                         )(class_inputs)
        projection_layer = TimeDistributed(Dense(self._word_vocab_size,
                                                 activation='linear'),
                                           name='projection')(recurrent_layer)
        word_class_membership_layer = Multiply(name='class_word_gate')([class_embedding_layer,
                                                                        projection_layer])
        class_cond_word_prob_layer = TimeDistributed(Activation(activation='softmax'),
                                                     name='class_cond_word_prob')(word_class_membership_layer)

        self._model = Model(inputs=[class_inputs] +  word_inputs,
                            outputs=[class_prob_output_layer, class_cond_word_prob_layer])
        self._model.compile(loss={'class_probability': 'sparse_categorical_crossentropy',
                                  'class_cond_word_prob': 'sparse_categorical_crossentropy'},
                            metrics={'class_probability': 'sparse_categorical_crossentropy',
                                     'class_cond_word_prob': 'sparse_categorical_crossentropy'},
                            optimizer=self._optimizer,
                            loss_weights={'class_probability': 5, 'class_cond_word_prob': 1})
        print(self._model.summary())

    def train(self, train_x, train_word_y, train_class_y, val_x, val_word_y, val_class_y):
        csv_logger = CSVLogger('{0}/{1}_training.log'.format(self._path,
                                                             self._name))

        checkpointer = ModelCheckpoint(filepath='{0}/{1}.hdf5'.format(self._path,
                                                                      self._name),
                                       monitor='val_loss', verbose=1, save_best_only=True)

        self._build_model()
        history = self._model.fit(x=train_x,
                                  y={'class_probability': train_class_y,
                                     'class_cond_word_prob': train_word_y},
                                  batch_size=self._batch_size,
                                  epochs=self._epochs,
                                  validation_data=(val_x,
                                                   {'class_probability': val_class_y,
                                                    'class_cond_word_prob': val_word_y}),
                                  callbacks=[csv_logger, checkpointer])
        print(history.history)
        return history

    # TODO: Add Perplexity Prediction
