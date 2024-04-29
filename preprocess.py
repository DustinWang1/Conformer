import torch.nn as nn
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import random
from datasets import load_dataset, load_metric, Audio
import config
import simpleaudio as sa
import numpy as np
import matplotlib.pyplot as plt
import librosa
import json
import os
import re

# Data Preprocessing
config = config.get_config()


def text_to_int(batch, vocab_dict):
    translation = []
    for c in batch["text"]:
        if c == ' ':
            ch = vocab_dict["|"]
        else:
            ch = vocab_dict[c]
        translation.append(ch)
    batch["text"] = torch.tensor(translation)
    return batch

def int_to_text(labels, vocab_dict):
    string = []
    for i in labels:
        string.append(vocab_dict[i])
    # return joins all values in the array with no separation.
    return ''.join(string).replace('', ' ')

def normalizeText(batch):
    chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\'\»\«]'
    batch["text"] = re.sub(chars_to_remove_regex, '', batch["text"]).lower()
    return batch

def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    # since set does not allow duplicate values, it will automatically remove all duplicate characters in all_text
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

def get_save_unique_vocab(train, test):
    # Get unique character list
    vocab_train = train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True,
                            remove_columns=train.column_names)
    vocab_test = test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True,
                          remove_columns=test.column_names)

    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}

    # replace space with | for ease
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]

    # Add blank and padding tokens
    vocab_dict["[BLNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    return vocab_dict

def show_random_sample(dataset):
    idx = random.randint(0, len(dataset))
    print(dataset[idx]["text"])
    audio_array = dataset[idx]["audio"]["array"]
    max_value = np.max(np.abs(audio_array))
    if max_value > 0:
        audio_array = audio_array / max_value
    audio_array = (audio_array * 32767).astype(np.int16)
    print("playing audio")
    audio_obj = sa.play_buffer(audio_array, 1, 2, dataset[idx]["audio"]["sampling_rate"])
    audio_obj.wait_done()
    print("finished")

def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)

def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")

def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")

def plotWaveandSpec(SPEECH_WAVEFORM, SAMPLE_RATE, spec):
    fig, axs = plt.subplots(2, 1)
    plot_waveform(SPEECH_WAVEFORM, SAMPLE_RATE, title="Original waveform", ax=axs[0])
    plot_spectrogram(spec, title="spectrogram", ax=axs[1])
    fig.tight_layout()

def visualize_sample(dataset):
    idx = random.randint(0, len(dataset))
    spec = dataset[idx]["spec"]
    audio_array = dataset[idx]["array"].unsqueeze(0)
    plotWaveandSpec(audio_array, 16000, spec)

def get_vocab_dict(train, valid):
    if os.path.exists("./vocab.json"):
        print("Using existing vocab dictionary")
        with open("./vocab.json", 'r') as file:
            vocab_dict = json.load(file)
            print(vocab_dict)
            return vocab_dict
    else:
        print("No vocab dictionary found. Generating.")
        vocab_dict = get_save_unique_vocab(train, valid)
        print(vocab_dict)
        return vocab_dict

def wrap_token(batch, token):
    batch["text"] = torch.cat([torch.tensor(token).unsqueeze(0), torch.tensor(batch["text"]), torch.tensor(token).unsqueeze(0)])
    return batch

# Create custom pytorch dataset
class CustomDataset(Dataset):
    def __init__(self, dataset, transform):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        audio_array = torch.from_numpy(self.dataset[idx]["audio"]["array"]).to(torch.float32)
        label = self.dataset[idx]["text"]
        return {"spec": self.transform(audio_array), "label": label, "array": audio_array}

def get_datasets():
    # Load and prepare dataset
    librispeech_train = load_dataset(config["dataset-link"], name=config["dataset-name"], split="train.100",
                                     token=config["HF_TOKEN"], trust_remote_code=True).select(range(100))
    librispeech_valid = load_dataset(config["dataset-link"], name=config["dataset-name"], split="validation",
                                     token=config["HF_TOKEN"], trust_remote_code=True).select(range(100))
    librispeech_test = load_dataset(config["dataset-link"], name=config["dataset-name"], split="test",
                                    token=config["HF_TOKEN"], trust_remote_code=True).select(range(100))

    # Remove unused data
    librispeech_train = librispeech_train.remove_columns(config["unused_columns"])
    librispeech_valid = librispeech_valid.remove_columns(config["unused_columns"])
    librispeech_test = librispeech_test.remove_columns(config["unused_columns"])

    librispeech_train = librispeech_train.map(normalizeText)
    librispeech_test = librispeech_test.map(normalizeText)
    librispeech_valid = librispeech_valid.map(normalizeText)

    vocab_dict = get_vocab_dict(librispeech_train, librispeech_valid)

    # Convert text to labels
    librispeech_train = librispeech_train.map(lambda x: text_to_int(x, vocab_dict))
    librispeech_test = librispeech_test.map(lambda x: text_to_int(x, vocab_dict))
    librispeech_valid = librispeech_valid.map(lambda x: text_to_int(x, vocab_dict))

    # Wrap labels with blank
    librispeech_train = librispeech_train.map(lambda x: wrap_token(x, config["BLANK"]))
    librispeech_test = librispeech_test.map(lambda x: wrap_token(x, config["BLANK"]))
    librispeech_valid = librispeech_valid.map(lambda x: wrap_token(x, config["BLANK"]))


    train_audio_transforms = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=config["input_dim"]),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
        torchaudio.transforms.TimeMasking(time_mask_param=35)
    )

    valid_audio_transforms = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64)

    train_dataset = CustomDataset(librispeech_train, train_audio_transforms)
    valid_dataset = CustomDataset(librispeech_valid, valid_audio_transforms)
    test_dataset = CustomDataset(librispeech_test, valid_audio_transforms)

    return train_dataset, valid_dataset, test_dataset, vocab_dict

