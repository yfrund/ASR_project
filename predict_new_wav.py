#Yuliia Frund

import torch
import torchaudio
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path
import model_conv
import test


def predict(data, model, labels):
    """
    :param data: input data - list of tuples [(input, true label)]
    :param model: trained model to make predictions
    :param labels: list of true labels
    prints number of correct guesses out of total + accuracy
    """
    model.eval()
    data_loader = DataLoader(data)

    correct = 0
    num_files = 0
    for item, target in data_loader:
        pred = torch.argmax(model(item))
        decode = labels[pred]
        print(f'Prediction: {decode}, true label: {target[0]}')
        if decode == target[0]:
            correct += 1
        num_files += 1
    print(f'{correct}/{num_files} correct guesses. Accuracy: {correct/num_files}')



def main():
    with open('labels.txt', 'r', encoding='utf-8') as f:
        labels = f.read().splitlines()


    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    asr_model = model_conv.ASR_Model(len(labels), device)
    asr_model.load_state_dict(torch.load('models/asr_conv_synth_200.pth', map_location=torch.device('cpu')))
    new_recs = Path('new_recordings_high_pass/').glob('*')

    data = []
    for file in new_recs:
        waveform , sample_rate = torchaudio.load(file)
        if waveform.size(1) < 16000:
            padding = 16000 - waveform.size(1)
            waveform = F.pad(waveform, (0, padding))
        elif waveform.size(1) > 16000:
            waveform = waveform[:, :16000]
        data.append((waveform, file.stem))
    predict(data, asr_model, labels)
    print(test.roc_auc(data, labels, asr_model))


if __name__ == '__main__':
    main()