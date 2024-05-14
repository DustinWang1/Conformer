import torch
import preprocess
from config import get_config
from main import LitAuto, data_processing
from torch.utils.data import DataLoader
from torchmetrics.text import CharErrorRate


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0, pad=28):
        super().__init__()
        self.labels = {value: key for key, value in labels.items()}
        self.blank = blank
        self.pad = pad

    def forward(self, emission: torch.Tensor, int_to_text: bool = False):
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        if not int_to_text:
            indices = torch.argmax(emission, dim=-1)  # [num_seq,]
            indices = torch.unique_consecutive(indices, dim=-1)
            indices = [i for i in indices if i != self.blank]
        else:
            indices = [i for i in emission if i != self.blank and i != self.pad]
        joined = "".join([self.labels[i.item()] for i in indices])
        return joined.replace("|", " ").strip()


def main():
    config = get_config()

    # create and load data
    train_ds, valid_ds, test_ds, vocab_dict = preprocess.get_datasets()
    test_dl = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, num_workers=7,
                         collate_fn=lambda x: data_processing(x, 'train'))

    decoder = GreedyCTCDecoder(vocab_dict, blank=config["BLANK"], pad=config["PAD"])
    cer = CharErrorRate()

    # load model
    checkpoint_path = "./lightning_logs/version_0/checkpoints/epoch=0-step=5708.ckpt"
    model = LitAuto.load_from_checkpoint(checkpoint_path, config=config, vocab_dict=vocab_dict)
    print(model)
    model.eval()
    print("model loaded")
    print("Sampling model")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Get sample
    spec, label, input_lengths, label_lengths = next(iter(test_dl))
    spec = spec[0]
    label = decoder(label[0], int_to_text=True)
    print("LABEL\n", label)


    with torch.no_grad():
        output, output_lengths = model(spec.unsqueeze(0).to(device), input_lengths[0].to(device))
        pred = decoder(output.squeeze(0))
        error = cer(pred, label)
        print(pred)
        print("Character Error Rate: ", error)




main()
