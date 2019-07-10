from keras.layers import Bidirectional, Concatenate, Embedding, GRU, Input, LSTM, TimeDistributed
from models.word import WordRNNLM

class RNNLM(WordRNNLM):
    def __init__(self,
                 type,
                 max_seq_len,
                 max_word_len,
                 word_vocab_size,
                 char_vocab_size,
                 save_dir,
                 config):
        super().__init__(max_seq_len=max_seq_len, word_vocab_size=word_vocab_size,
                         save_dir=save_dir, config=config)
        self._type = type
        self._max_word_len = max_word_len
        self._char_vocab_size = char_vocab_size
        self._char_embedding_dim = config['embedding']['char']['embedding_dim']
        self._char_hidden_type = config['embedding']['char']['hidden_type']
        if self._char_hidden_type in {'lstm', 'gru'}:
            self._char_recurrent_num_hidden_layers = config['embedding']['char'][self._char_hidden_type]['num_hidden_layers']
            self._char_recurrent_hidden_dim = config['embedding']['char'][self._char_hidden_type]['hidden_dim']
        else:
            raise ValueError('{} character level model not implemented.'.format(self._char_hidden_type))

    def _get_char_representation(self):
        char_input_layer = Input(shape=(self._max_seq_len, self._max_word_len),
                                 name='char_input'
                                 )

        char_embedding_layer = TimeDistributed(Embedding(input_dim=self._char_vocab_size,
                                                         output_dim=self._char_embedding_dim,
                                                         mask_zero=True,
                                                         trainable=True,
                                                         ),
                                               name='char_embedding'
                                               )(char_input_layer)

        char_rep_layer = None
        if self._char_hidden_type == 'lstm':
            char_rep_layer = TimeDistributed(Bidirectional(LSTM(units=self._char_recurrent_hidden_dim,
                                                                return_sequences=False,
                                                                return_state=False)),
                                             name='char_lstm0')(char_embedding_layer)

            for i in range(1, self._char_recurrent_num_hidden_layers):
                char_rep_layer = TimeDistributed(Bidirectional(LSTM(units=self._char_recurrent_hidden_dim,
                                                                    return_sequences=False,
                                                                    return_state=False)),
                                                 name='char_lstm' + str(i)
                                                 )(char_rep_layer)
        elif self._char_hidden_type == 'gru':
            char_rep_layer = TimeDistributed(Bidirectional(GRU(units=self._char_recurrent_hidden_dim,
                                                               return_sequences=False,
                                                               return_state=False)),
                                             name='char_gru0'
                                             )(char_embedding_layer)

            for i in range(1, self._char_recurrent_num_hidden_layers):
                char_rep_layer = TimeDistributed(Bidirectional(GRU(units=self._char_recurrent_hidden_dim,
                                                                   return_sequences=False,
                                                                   return_state=False)),
                                                 name='char_gru' + str(i)
                                                 )(char_rep_layer)

        return [char_input_layer], char_rep_layer

    def _get_embedding_layer(self):
        if self._type == 'char':
            return self._get_char_representation()
        elif self._type == 'word':
            return self._get_word_representation()
        else:
            char_inputs, char_rep_layer = self._get_char_representation()
            word_inputs, word_rep_layer = self._get_word_representation()

            inputs = char_inputs + word_inputs
            merged = Concatenate(name='combined_rep')([char_rep_layer, word_rep_layer])
            return inputs, merged
