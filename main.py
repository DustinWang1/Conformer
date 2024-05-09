import math

import preprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
from Conformer import Conformer
import lightning as L
from config import get_config
from torch.utils.data import DataLoader


class LitAuto(L.LightningModule):
    def __init__(self, Conformer, config):
        super().__init__()
        self.conformer = Conformer
        self.config = config
        self.ctc = nn.CTCLoss(blank=config["BLANK"], zero_infinity=True)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        spec, label, input_lengths, label_lengths = batch
        output, output_lengths = self.conformer(spec, input_lengths)
        output, output_lengths = pad_outputs(output, output_lengths, label_lengths, self.config["BLANK"])
        val_loss = F.ctc_loss(output.transpose(0, 1), label, output_lengths, label_lengths, zero_infinity=True)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        spec, label, input_lengths, label_lengths = batch
        output, output_lengths = self.conformer(spec, input_lengths)
        output, output_lengths = pad_outputs(output, output_lengths, label_lengths, self.config["BLANK"])
        test_loss = F.ctc_loss(output.transpose(0, 1), label, output_lengths, label_lengths, zero_infinity=True)
        self.log("test_loss", test_loss, prog_bar=True)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        spec, label, input_lengths, label_lengths = batch
        output, output_lengths = self.conformer(spec, input_lengths)

        output, output_lengths = pad_outputs(output, output_lengths, label_lengths, self.config["BLANK"])

        if torch.isnan(output).any():
            raise ValueError("Nan in output")
        if torch.isinf(output).any():
            raise ValueError("Inf in output")

        loss = self.ctc(output.transpose(0, 1), label, output_lengths, label_lengths)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=4e-5)
        return optimizer


def pad_outputs(output, output_lengths, label_lengths, blank_token):
    padding_amount = max(label_lengths - output_lengths)+1

    if padding_amount <= 0:
        return output, output_lengths

    output_lengths += padding_amount
    blank_vector = F.one_hot(torch.tensor(blank_token), num_classes=output.size(2)).to("cuda")
    blank_vector = F.log_softmax(blank_vector.float(), dim=-1)
    batches = []
    for batch in output:
        batches.append(torch.cat([batch, blank_vector.repeat(padding_amount, 1)], dim=0))
    output = torch.stack(batches, dim=0)
    return output, output_lengths


def data_processing(batch, data_type="train"):
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
    train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=7, collate_fn=lambda x: data_processing(x, 'train'))
    valid_dl = DataLoader(valid_ds, batch_size=config["batch_size"], shuffle=False, num_workers=7, collate_fn=lambda x: data_processing(x, 'train'))
    test_dl = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, num_workers=7, collate_fn=lambda x: data_processing(x, 'train'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    conformer = Conformer(config["input_dim"], config["model_dim"], len(vocab_dict), device=device)
    model = LitAuto(conformer, config)

    trainer = L.Trainer(max_epochs=config["max_epochs"], gradient_clip_val=0.5)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)
    trainer.test(dataloaders=test_dl)


if __name__ == '__main__':
    config = get_config()
    main(config)
