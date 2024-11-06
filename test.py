#Yuliia Frund

from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import predict_tensor

def roc_auc(test_data, labels, model):
    """
    :param test_data: test data file
    :param labels: list of true lables
    :param model: model to assess
    :return: AUROC, accuracy
    also creates a confusion matrix and saves it in images
    need to adapt its name for different assessments
    """
    model.eval()
    num_labels = len(labels)
    y_true = []
    y_pred = []
    y_t = []
    y_p = []
    count = 0
    test_loader = DataLoader(test_data)
    for item, target in test_loader:
        temp = [0] * num_labels
        temp[labels.index(target[0])] = 1
        y_true.append(temp)
        temp = [0] * num_labels
        temp[labels.index(predict_tensor.predict(item, labels, model))] = 1
        y_pred.append(temp)
        y_t.append(labels.index(target[0]))
        y_p.append(labels.index(predict_tensor.predict(item, labels, model)))

        if predict_tensor.predict(item, labels, model) == target[0]:
            count += 1

    score = roc_auc_score(y_true, y_pred)


    conf_matrix = confusion_matrix(y_t, y_p)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=labels)
    disp.plot(xticks_rotation='vertical')
    plt.savefig('images/conf_matr_new_recs_synth_hp.jpeg')

    return score, count/len(test_data)
