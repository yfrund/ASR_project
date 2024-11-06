#Yuliia Frund

import torch


def predict(input, labels, model):
    """
    :param input: features for single input
    :param labels: list of true labels
    :param model: model for predictions
    :return: predicted label
    """
    model.eval()
    pred = torch.argmax(model(torch.tensor(input)))
    decode = labels[pred]

    return decode
