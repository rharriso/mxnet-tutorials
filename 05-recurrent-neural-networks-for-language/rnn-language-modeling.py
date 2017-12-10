from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
import numpy as np

mx.random.seed(1)
# ctx = mx.gpu(0)
ctx = mx.cpu()

""" produces the one hot vectors for a an entire text,
represented by a numerical list of vocab indices """


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
        if i % 1000 == 0: print("textify %s of %s" % (i, total))
        result += given_character_list[int(idx)]
    return result


def softmax(y_linear, temperature=0.1):
    lin = (y_linear - nd.max(y_linear)) / temperature
    exp = nd.exp(lin)
    partition = nd.sum(exp, axis=0, exclude=True).reshape((-1, 1))
    return exp / partition


def simple_rnn(inputs, state, temperature=1.0):
    outputs = []
    h = state
    for X in inputs:
        h_linear = nd.dot(X, Wxh) + nd.dot(h, Whh) + bh
        h = nd.tanh(h_linear)
        yhat_linear = nd.dot(h, Why) + by
        yhat = softmax(yhat_linear, temperature=temperature)
        outputs.append(yhat)
    return (outputs, h)


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
    # Initialize the string that we'll return to the supplied prefix
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
    sample_state = nd.zeros(shape=(1, num_hidden), ctx=ctx)

    #####################################
    # For num_chars iterations,
    #     1) feed in the current input
    #     2) sample next character from from output distribution
    #     3) add sampled character to the decoded string
    #     4) prepare the sampled character as a one_hot (to be the next input)
    #####################################
    for i in range(num_chars):
        outputs, sample_state = simple_rnn(input, sample_state, temperature=temperature)
        choice = np.random.choice(vocab_size, p=outputs[-1][0].asnumpy())
        string += character_list[choice]
        input = one_hots([choice], vocab_size)
    return string


# data set is from the time machine
with open("./time-machine.txt") as f:
    time_machine = f.read()
    # cut out the legalese
    time_machine = time_machine[:-18600]
    # print(time_machine[:-500])

    # numerical representation of characters
    character_list = list(set(time_machine))
    vocab_size = len(character_list)
    # print(character_list)
    # print("Length of vocab %s" % vocab_size)

    # index lookup for each character
    character_dict = {}
    for e, char in enumerate(character_list):
        character_dict[char] = e

    # get the index for every char in the time_machine
    time_numerical = [character_dict[char] for char in time_machine]

    # print(len(time_numerical))
    # print(time_numerical[:20])
    # print as text
    # print("".join([character_list[idx] for idx in time_numerical[10000:20]]))

    # one hot representations
    # print(one_hots(time_numerical[:2], vocab_size))

    # convert onehots back to text
    # print(textify(
    #     one_hots(time_numerical[10000:10500], vocab_size)
    #     , character_list))

    # preparing data for training
    # split into smaller datasets
    seq_length = 64 # text sequences of 64 chars
    # -1 here so we have enough characters for labels later (there is no label for the last char)
    num_samples = (len(time_numerical) - 1) // seq_length
    dataset = one_hots(
        time_numerical[:seq_length * num_samples],
        vocab_size
        # reshape set of onehots to be organized by sequence
        ).reshape((num_samples, seq_length, vocab_size))
    # print(textify(dataset[0], character_list))

    # feed batches all at once to take advantage of gpu
    batch_size = 32
    print('# of sequnces in dataset', len(dataset))
    num_batches = len(dataset) // batch_size
    print('# batches', num_batches)
    train_data = dataset[:num_batches * batch_size].reshape(
        (num_batches, batch_size, seq_length, vocab_size))
    # swap batch_size and seq_length axis to make later acces easier
    train_data = nd.swapaxes(train_data, 1, 2)
    print('Shape of data: ', train_data.shape)

    # print out the first three batches
    # for i in range(3):
    #     print("***Batch %s:" % i)
    #     print("%s \n\n" %
    #         ( textify(train_data[i, :, 0], character_list)
    #         + textify(train_data[i, :, 1], character_list)
    #         ))

    labels = one_hots(time_numerical[1:seq_length * num_samples + 1], vocab_size)
    train_labels = labels.reshape((num_batches, batch_size, seq_length, vocab_size))
    # same axis swap as earlier
    train_labels = nd.swapaxes(train_labels, 1, 2)
    print(train_labels.shape)

    #print(textify(train_data[0, :, 0], character_list))
    #print(textify(train_label[0, :, 0], character_list))

    num_inputs = vocab_size
    num_hidden = 256
    num_outputs = vocab_size

    ########################
    #  Weights connecting the inputs to the hidden layer
    ########################
    Wxh = nd.random_normal(shape=(num_inputs, num_hidden), ctx=ctx) * 0.01

    ########################
    #  Recurrent weights connecting the hidden layer across time steps
    ########################
    Whh = nd.random_normal(shape=(num_hidden, num_hidden), ctx=ctx) * 0.01

    ########################
    #  Bias vector for hidden layer
    ########################
    bh = nd.random_normal(shape=(num_hidden), ctx=ctx) * 0.01

    ########################
    # Weights to the output nodes
    ########################
    Why = nd.random_normal(shape=(num_hidden, num_outputs), ctx=ctx)  * 0.01
    by = nd.random_normal(shape=(num_outputs), ctx=ctx)  * 0.01

    # attache the gradients
    params = [Wxh, Whh, bh, Why, by]

    for param in params:
        param.attach_grad()

    ####################
    # With a temperature of 1 (always 1 during training),
    # we get back some set of probabilities
    ####################
    print(softmax(nd.array([[1, -1], [-1, 1]]), temperature=1.0))

    ####################
    # If we set a high temperature,
    # we can get more entropic (*noisier*) probabilities
    ####################
    print(softmax(nd.array([[1, -1], [-1, 1]]), temperature=1000.0))

    ####################
    # Often we want to sample with low temperatures
    # to produce sharp probabilities
    ####################
    print(softmax(nd.array([[10, -10], [-10, 10]]), temperature=0.1))

    print(cross_entropy(nd.array([.2,.5,.3]), nd.array([1.,0,0])))


    epochs = 2000
    moving_loss = 0

    learning_rate = 0.5

    # state = nd.zeros(shape=(batch_size, num_hidden), ctx=ctx)
    for e in range(epochs):
        ############################
        # Attenuate the learning rate by a factor of 2 every 100 epochs.
        ############################
        if ((e + 1) % 100 == 0):
            learning_rate = learning_rate / 2.0

        state = nd.zeros(shape=(batch_size, num_hidden), ctx=ctx)
        for i in range(num_batches):
            data_one_hot = train_data[i]
            label_one_hot = train_labels[i]
            with autograd.record():
                outputs, state = simple_rnn(data_one_hot, state)
                loss = average_ce_loss(outputs, label_one_hot)
                loss.backward()
            SGD(params, learning_rate)


            ##########################
            #  Keep a moving average of the losses
            ##########################
            if (i == 0) and (e == 0):
                moving_loss = np.mean(loss.asnumpy()[0])
            else:
                moving_loss = .99 * moving_loss + .01 * np.mean(loss.asnumpy()[0])


        print("Epoch %s. Loss: %s" % (e, moving_loss))
        print(sample("The Time Ma", 1024, temperature=.1))
        print(sample("The Medical Man rose, came to the lamp,", 1024, temperature=.1))

