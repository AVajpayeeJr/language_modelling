import keras.backend as K
from keras.layers import Activation, Dense, Dot, Embedding, GRU, Input, Lambda, LSTM, Multiply, TimeDistributed
from keras.models import Model
from models.word import WordRNNLM


class WordClassRNNLM(WordRNNLM):
    def __init__(self,
                 max_seq_len,
                 word_vocab_size,
                 class_vocab_size,
                 class_membership_weights,
                 class_one_hot_weights,
                 save_dir,
                 config):
        super().__init__(max_seq_len=max_seq_len, word_vocab_size=word_vocab_size,
                         save_dir=save_dir, config=config)
        self._class_vocab_size = class_vocab_size
        self._class_membership_weights = class_membership_weights
        self._class_one_hot_weights = class_one_hot_weights

    # TODO: Use Lambda Layers
    def _build_model(self):
        """
        """
        word_inputs, word_embedding_layer = self._get_embedding_layer()
        class_inputs = Input(shape=(self._max_seq_len,), dtype='int32', name='class_input')

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

        class_prob_vector_layer = TimeDistributed(Dense(self._class_vocab_size,
                                                        activation='softmax'),
                                                  name='class_probability')(recurrent_layer)

        # Getting Probability of actual class
        class_one_hot_layer = Embedding(output_dim=self._class_vocab_size,
                                        input_dim=self._class_vocab_size,
                                        input_length=self._max_seq_len,
                                        weights=[self._class_one_hot_weights],
                                        trainable=False,
                                        mask_zero=True,
                                        name='class_one_hot_layer')(class_inputs)
        class_prob_output_layer = Dot(axes=-1, name='class_gate')([class_one_hot_layer, class_prob_vector_layer])

        # Simulating softmax over restricted vocab (based on input class)
        class_membership_layer = Embedding(output_dim=self._word_vocab_size,
                                           input_dim=self._class_vocab_size,
                                           input_length=self._max_seq_len,
                                           weights=[self._class_membership_weights],
                                           trainable=False,
                                           mask_zero=True,
                                           name='class_word_membership'
                                           )(class_inputs)
        projection_layer = TimeDistributed(Dense(self._word_vocab_size,
                                                 activation='linear'),
                                           name='projection')(recurrent_layer)
        word_class_membership_layer = Multiply(name='class_word_gate')([class_membership_layer,
                                                                        projection_layer])
        class_cond_word_prob_layer = TimeDistributed(Activation(activation='softmax'),
                                                     name='class_cond_word_prob')(word_class_membership_layer)

        # Final Output Layer
        reshape_layer = Lambda(lambda x: K.repeat_elements(x, self._word_vocab_size, -1),
                               name='reshape')(class_prob_output_layer)
        output_layer = Multiply(name='output_layer')([class_cond_word_prob_layer, reshape_layer])


        self._model = Model(inputs=[class_inputs] +  word_inputs,
                            output=output_layer)
        self._model.compile(loss='sparse_categorical_crossentropy',
                            optimizer=self._optimizer)
        print(self._model.summary())

