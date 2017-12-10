import mxnet as mx
from mxnet import nd, autograd
import numpy as np

out_file = open("./lnn.output", "w")
out_file.write("test: {}".format(5))

mx.random.seed(1)
ctx = mx.cpu()

def one_hots(numerical_list, vocab_magnitude):
    total = len(numerical_list)
    result = nd.zeros((len(numerical_list), vocab_magnitude), ctx=ctx)
    for i, idx in enumerate(numerical_list):
        result[i, idx] = 1.0
    return result


def textify(embedding, given_character_list):
    result = ""
    total = len(embedding)
    indices = nd.argmax(embedding, axis=1).asnumpy()
    for i, idx in enumerate(indices):
        result += given_character_list[int(idx)]
    return result



def softmax(y_linear, temperature=0.1):
    lin = (y_linear - nd.max(y_linear)) / temperature
    exp = nd.exp(lin)
    partition = nd.sum(exp, axis=0, exclude=True).reshape((-1, 1))
    return exp / partition


# define the lstm model
def lstm_rnn(inputs, h, c, temperature=1.0):
    outputs = []
    for X in inputs:
        g = nd.tanh(nd.dot(X, Wxg) + nd.dot(h, Whg) + bg)
        i = nd.sigmoid(nd.dot(X, Wxi) + nd.dot(h, Whi) + bi)
        f = nd.sigmoid(nd.dot(X, Wxf) + nd.dot(h, Whf) + bf)
        o = nd.sigmoid(nd.dot(X, Wxo) + nd.dot(h, Who) + bo)
        #######################
        #
        #######################
        c = f * c + i * g
        h = o * nd.tanh(c)
        #######################
        #
        #######################
        yhat_linear = nd.dot(h, Why) + by
        yhat = softmax(yhat_linear, temperature=temperature)
        outputs.append(yhat)
    return (outputs, h, c)

def cross_entropy(yhat, y):
    return - nd.mean(nd.sum(y * nd.log(yhat), axis=0, exclude=True))


def average_ce_loss(outputs, labels):
    assert(len(outputs) == len(labels))
    total_loss = 0
    for (output, label) in zip(outputs, labels):
        total_loss = total_loss + cross_entropy(output, label)
    return total_loss / len(outputs)


""" Optimizer """
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


def sample(prefix, num_chars, temperature=1.0):
    #####################################
    # Initialize the string that we"ll return to the supplied prefix
    #####################################
    string = prefix

    #####################################
    # Prepare the prefix as a sequence of one-hots for ingestion by RNN
    #####################################
    prefix_numerical = [character_dict[char] for char in prefix]
    input = one_hots(prefix_numerical, vocab_size)

    #####################################
    # Set the initial state of the hidden representation ($h_0$) to the zero vector
    #####################################
    h = nd.zeros(shape=(1, num_hidden), ctx=ctx)
    c = nd.zeros(shape=(1, num_hidden), ctx=ctx)

    #####################################
    # For num_chars iterations,
    #     1) feed in the current input
    #     2) sample next character from from output distribution
    #     3) add sampled character to the decoded string
    #     4) prepare the sampled character as a one_hot (to be the next input)
    #####################################
    for i in range(num_chars):
        outputs, h, c = lstm_rnn(input, h, c, temperature=temperature)
        choice = np.random.choice(vocab_size, p=outputs[-1][0].asnumpy())
        string += character_list[choice]
        input = one_hots([choice], vocab_size)
    return string


# data set is from the time machine
with open("./time-machine.txt") as f:
    time_machine = f.read()
    # cut out the legalese
    time_machine = time_machine[:-18600]
    # out_file.write(time_machine[:-500])

    # numerical representation of characters
    character_list = list(set(time_machine))
    vocab_size = len(character_list)

    # index lookup for each character
    character_dict = {}
    for e, char in enumerate(character_list):
        character_dict[char] = e

    # get the index for every char in the time_machine
    time_numerical = [character_dict[char] for char in time_machine]

    seq_length = 64 # text sequences of 64 chars
    # -1 here so we have enough characters for labels later (there is no label for the last char)
    num_samples = (len(time_numerical) - 1) // seq_length
    dataset = one_hots(
        time_numerical[:seq_length * num_samples],
        vocab_size
        # reshape set of onehots to be organized by sequence
        ).reshape((num_samples, seq_length, vocab_size))
    # out_file.write(textify(dataset[0], character_list))

    # feed batches all at once to take advantage of gpu
    batch_size = 32
    out_file.write("\n# of sequnces in dataset: {}".format(len(dataset)))
    num_batches = len(dataset) // batch_size
    out_file.write("\n# batches: {}".format(num_batches))
    data = dataset[:num_batches * batch_size]
    train_data = data.reshape((num_batches, batch_size, seq_length, vocab_size))
    train_data = nd.swapaxes(train_data, 1, 2)
    out_file.write("\nShape of data: {}".format(train_data.shape))
    # set up train labels
    labels = one_hots(time_numerical[1:seq_length * num_samples + 1], vocab_size)
    train_labels = labels.reshape((num_batches, batch_size, seq_length, vocab_size))
    train_labels = nd.swapaxes(train_labels, 1, 2)
    out_file.write("\nShape of data: {}".format(train_labels.shape))

    num_inputs = vocab_size
    num_hidden = 256
    num_outputs = vocab_size

    ########################
    #  Weights connecting the inputs to the hidden layer
    ########################
    Wxg = nd.random_normal(shape=(num_inputs, num_hidden), ctx=ctx) * .01;
    Wxi = nd.random_normal(shape=(num_inputs, num_hidden), ctx=ctx) * .01;
    Wxf = nd.random_normal(shape=(num_inputs, num_hidden), ctx=ctx) * .01;
    Wxo = nd.random_normal(shape=(num_inputs, num_hidden), ctx=ctx) * .01;

    ########################
    #  Recurrent weights connecting the hidden layer across time steps
    ########################
    Whg = nd.random_normal(shape=(num_hidden, num_hidden), ctx=ctx) * .01;
    Whi = nd.random_normal(shape=(num_hidden, num_hidden), ctx=ctx) * .01;
    Whf = nd.random_normal(shape=(num_hidden, num_hidden), ctx=ctx) * .01;
    Who = nd.random_normal(shape=(num_hidden, num_hidden), ctx=ctx) * .01;

    ########################
    #  Bias vector for hidden layer
    ########################
    bg = nd.random_normal(shape=num_hidden, ctx=ctx) * .01
    bi = nd.random_normal(shape=num_hidden, ctx=ctx) * .01
    bf = nd.random_normal(shape=num_hidden, ctx=ctx) * .01
    bo = nd.random_normal(shape=num_hidden, ctx=ctx) * .01

    ########################
    # Weights to the output nodes
    ########################
    Why = nd.random_normal(shape=(num_hidden, num_outputs), ctx=ctx) * 0.01
    by = nd.random_normal(shape=(num_outputs), ctx=ctx) * 0.01

    # attach the gradients
    params = [Wxg, Wxi, Wxf, Wxo, Whg, Whi, Whf, Who, bg, bi, bf, bo, Why, by]

    for param in params:
        param.attach_grad()

    epochs = 2000
    moving_loss = 0.

    learning_rate = 2.0

    for e in range(epochs):
        ############################
        # Attenuate the learning rate by a factor of 2 every 100 epochs.
        ############################
        if ((e+1) % 100 == 0):
            learning_rate = learning_rate / 2.0

        h = nd.zeros(shape=(batch_size, num_hidden), ctx=ctx)
        c = nd.zeros(shape=(batch_size, num_hidden), ctx=ctx)

        for i in range(num_batches):
            if i % 10 == 0: out_file.write("\nbatch {}".format(i))
            data_one_hot = train_data[i]
            label_one_hot = train_labels[i]
            with autograd.record():
                outputs, h, c = lstm_rnn(data_one_hot, h, c)
                loss = average_ce_loss(outputs, label_one_hot)
                loss.backward()
            SGD(params, learning_rate)

            ##########################
            #  Keep a moving average of the losses
            ##########################
            if (i == 0) and (e == 0):
                moving_loss = nd.mean(loss).asscalar()
            else:
                moving_loss = .99 * moving_loss + .01 * nd.mean(loss).asscalar()


        out_file.write("\nEpoch {}. Loss: {}".format(e, moving_loss))
        out_file.write(sample("\nThe Time Ma", 1024, temperature=.1))
        out_file.write(sample("\nThe Medical Man rose, came to the lamp,", 1024, temperature=.1))


