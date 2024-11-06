#Yuliia Frund

import torchaudio
import random
import torch.nn.functional as F

def balance_data(directory):
    """
    :param directory: directory with subdirectories with data files
    :return: minimal number of files in a subdirectory
    """
    counts = []
    for dir in directory.iterdir():
        files = [file for file in dir.glob('*.wav')]
        counts.append(len(files))
    return min(counts)

def get_features(directory, maximum):
    """
    :param directory: directory with subdirectories with data files
    :param maximum: maximum number of files to include
    :return: list of tuples [(wavform, tru label)]
    """
    labels = []
    data = []
    for dir in directory.iterdir():
        count = 0
        if dir.is_dir():
            labels.append(dir.name)

            for file in dir.glob('*.wav'):
                if count < maximum:

                    waveform, sample_rate = torchaudio.load(file)
                    if waveform.size(1) < 16000:
                        padding = 16000 - waveform.size(1)
                        waveform = F.pad(waveform, (0, padding))
                    elif waveform.size(1) > 16000:
                        waveform = waveform[:, :16000]
                    data.append((waveform, dir.name))
                    count += 1
    labels.sort()
    with open('labels.txt', 'w', encoding='utf-8') as out:
        for label in labels:
            out.write(label + '\n')
    random.shuffle(data)
    return data
