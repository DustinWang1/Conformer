import math
import preprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
from Conformer import Conformer
from ConformerLightning import LitAuto
import lightning as L
from config import get_config
from torch.utils.data import DataLoader


def data_processing(batch):
    config = get_config()
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for item in batch:
        spectrograms.append(item["spec"].transpose(0, 1))
        labels.append(torch.tensor(item["label"]))
        input_lengths.append(item["spec"].shape[1])
        label_lengths.append(len(item["label"]))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True, padding_value=0.0).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=config["PAD"])

    return spectrograms, labels, torch.LongTensor(input_lengths), torch.LongTensor(label_lengths)


def main(config):
    torch.set_float32_matmul_precision('high')

    # spec: (mel_banks, time)
    train_ds, valid_ds, test_ds, vocab_dict = preprocess.get_datasets()

    # Create a data loader
    train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=7, collate_fn=lambda x: data_processing(x))
    valid_dl = DataLoader(valid_ds, batch_size=config["batch_size"], shuffle=False, num_workers=7, collate_fn=lambda x: data_processing(x))
    test_dl = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, num_workers=7, collate_fn=lambda x: data_processing(x))

    model = LitAuto(config, vocab_dict)

    trainer = L.Trainer(max_epochs=config["max_epochs"], gradient_clip_val=0.5)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)
    trainer.test(dataloaders=test_dl)


if __name__ == '__main__':
    config = get_config()
    main(config)
