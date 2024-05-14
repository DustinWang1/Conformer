import lightning as L
from Conformer import Conformer
import torch
import torch.nn as nn
import torch.nn.functional as F

class LitAuto(L.LightningModule):
    def __init__(self, config, vocab_dict):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conformer = Conformer(config["input_dim"], config["model_dim"], len(vocab_dict), device=device)
        self.config = config
        self.ctc = nn.CTCLoss(blank=config["BLANK"], zero_infinity=True)

    def pad_outputs(self, output, output_lengths, label_lengths, blank_token):
        padding_amount = max(label_lengths - output_lengths) + 1

        if padding_amount <= 0:
            return output, output_lengths

        output_lengths += padding_amount
        blank_vector = F.one_hot(torch.tensor(blank_token), num_classes=output.size(2)).to(self.device)
        blank_vector = F.log_softmax(blank_vector.float(), dim=-1)
        batches = []
        for batch in output:
            batches.append(torch.cat([batch, blank_vector.repeat(padding_amount, 1)], dim=0))
        output = torch.stack(batches, dim=0)
        return output, output_lengths

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        spec, label, input_lengths, label_lengths = batch
        output, output_lengths = self.conformer(spec, input_lengths)
        output, output_lengths = self.pad_outputs(output, output_lengths, label_lengths, self.config["BLANK"])
        val_loss = F.ctc_loss(output.transpose(0, 1), label, output_lengths, label_lengths, zero_infinity=True)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        spec, label, input_lengths, label_lengths = batch
        output, output_lengths = self.conformer(spec, input_lengths)
        output, output_lengths = self.pad_outputs(output, output_lengths, label_lengths, self.config["BLANK"])
        test_loss = F.ctc_loss(output.transpose(0, 1), label, output_lengths, label_lengths, zero_infinity=True)
        self.log("test_loss", test_loss, prog_bar=True)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        spec, label, input_lengths, label_lengths = batch
        output, output_lengths = self.conformer(spec, input_lengths)

        output, output_lengths = self.pad_outputs(output, output_lengths, label_lengths, self.config["BLANK"])

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

    def forward(self, spec, input_lengths):
        output, output_lengths = self.conformer(spec, input_lengths)
        return output, output_lengths

