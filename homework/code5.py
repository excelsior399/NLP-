import re
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation, GlobalAveragePooling1D
from tensorflow.keras.models import Model

from seq2seq import Seq2Seq
from transformer import TokenAndPositionEmbedding, TransformerBlock


def make_transformer(num_outputs=1, word_size=16, reg_param=0.0001, final_activation='sigmoid'):
    maxlen = 20000
    vocab_size = 200
    embed_dim = 32
    num_heads = 2
    ff_dim = 32
    inp = Input(shape=(int(word_size * 4),))
    # rs = Reshape((s_groups, int(num_blocks * word_size * 4)))(inp)
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inp)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = GlobalAveragePooling1D()(x)
    flat1 = Dense(20, activation="relu")(x)

    dense1 = Dense(20)(flat1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense2 = Dense(20)(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2)
    model = Model(inputs=inp, outputs=out)
    model.summary()
    return (model)

def is_uchar(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    if uchar in ('，', '。', '：', '？', '“', '”', '！', '；', '、', '《', '》', '——'):
        return True
    return False

def data_generator(data, batch_size, time_steps):
    num_batches = len(data) // (batch_size * time_steps)
    data = data[:num_batches * batch_size * time_steps]
    data = np.array(data).reshape((batch_size, -1))
    while True:
        for i in range(0, data.shape[1], time_steps):
            x = data[:, i:i + time_steps]
            y = np.roll(x, -1, axis=1)
            yield x, y


if __name__ == '__main__':
    book_name = '笑傲江湖'
    content = ''
    f = open("../中文语料" + "/" + book_name + ".txt", "r", encoding='gbk', errors='ignore')
    book = f.readlines()

    pattern = re.compile(r'\(.*\)')
    data = [pattern.sub('', lines) for lines in book]
    data = [line.replace('……', '。') for line in data if len(line) > 1]
    data = ''.join(data)
    data = [char for char in data if is_uchar(char)]
    data = ''.join(data)

    # 构建词汇表
    vocab = list(set(data))
    char2id = {c: i for i, c in enumerate(vocab)}
    id2char = {i: c for i, c in enumerate(vocab)}
    numdata = [char2id[char] for char in data]


    hidden_size = 256  # 减少隐藏单元数量
    hidden_layers = 3  # 减少隐藏层数量
    vocab_size = len(vocab)
    batch_size = 32  # 增加批次大小
    time_steps = 50  # 减少时间步长
    epochs = 100  # 减少训练轮次
    maxlen = 20
    embedding_dim = 10
    hidden_units = 100
    learning_rate = 0.01  # 降低学习率

    seq2seq_model = Seq2Seq(maxlen, embedding_dim, hidden_units, vocab_size)
    seq2seq_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='categorical_crossentropy')
    transformer_model = make_transformer()
    transformer_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                          loss='categorical_crossentropy')
    train_data = data_generator(numdata, batch_size, time_steps)

    # 训练模型
    seq2seq_model.fit(train_data, epochs=epochs, steps_per_epoch=len(numdata) // (batch_size * time_steps))

    # 文本生成函数
    def generate_text(model, start_string, num_generate=100):
        input_eval = [char2id[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        text_generated = []
        states = None

        for i in range(num_generate):
            predictions, states = model(input_eval, states=states, return_state=True)
            predictions = tf.squeeze(predictions, 0)
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
            input_eval = tf.expand_dims([predicted_id], 0)
            text_generated.append(id2char[predicted_id])

        return start_string + ''.join(text_generated)


    # 生成文本示例
    print(generate_text(seq2seq_model, start_string=""))