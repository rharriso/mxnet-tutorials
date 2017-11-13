from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
import numpy as numpy

mx.random.seed(1)
ctx = mx.gpu(0)

""" produces the one hot vectors for a an entire text,
represented by a numerical list of vocab indices """


def one_hots(numerical_list, vocab_magnitude):
    total = len(numerical_list)
    result = nd.zeros((len(numerical_list), vocab_magnitude), ctx=ctx)
    for i, idx in enumerate(numerical_list):
        if i % 1000 == 0: print("onehot %s of %s" % (i, total))
        result[i, idx] = 1.0
    return result


def textify(embedding, given_character_list):
    result = ""
    indices = mx.nd.argmax(embedding, axis=1).asnumpy()
    for idx in indices:
        result += given_character_list[int(idx)]
    return result


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
    train_labels = mx.nd.swapaxes(train_labels, 1, 2)
    print(train_labels.shape)

    print(textify(train_data[0, :, 0], character_list))
    print(textify(train_label[0, :, 0], character_list))

