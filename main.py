import preprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
from Conformer import Conformer
import lightning as L
from config import get_config
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelSummary


class LitAuto(L.LightningModule):
    def __init__(self, Conformer):
        super().__init__()
        self.conformer = Conformer

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        spec, label, input_lengths, label_lengths = batch
        output, output_lengths = self.conformer(spec, input_lengths)
        val_loss = F.ctc_loss(output.transpose(0, 1), label, output_lengths, label_lengths)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        spec, label, input_lengths, label_lengths = batch
        output, output_lengths = self.conformer(spec, input_lengths)
        test_loss = F.ctc_loss(output.transpose(0, 1), label, output_lengths, label_lengths)
        self.log("test_loss", test_loss, prog_bar=True)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        spec, label, input_lengths, label_lengths = batch
        output, output_lengths = self.conformer(spec, input_lengths)
        loss = F.ctc_loss(output.transpose(0, 1), label, output_lengths, label_lengths)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=4e-5)
        return optimizer

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

    # spec: (mel_banks, time)
    train_ds, valid_ds, test_ds, vocab_dict = preprocess.get_datasets()

    # Create a data loader
    train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=lambda x: data_processing(x, 'train'))
    valid_dl = DataLoader(valid_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=lambda x: data_processing(x, 'train'))
    test_dl = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=lambda x: data_processing(x, 'train'))

    conformer = Conformer(config["input_dim"], config["model_dim"], len(vocab_dict))
    model = LitAuto(conformer)

    trainer = L.Trainer(max_epochs=20, log_every_n_steps=5, callbacks=[ModelSummary(max_depth=3)])
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)




if __name__ == '__main__':
    config = get_config()
    main(config)
