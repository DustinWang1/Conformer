
def get_config():
    return {
        "dataset-title": "librispeech_asr",
        "dataset-link": "librispeech_asr",
        "dataset-name": "clean",
        "HF_TOKEN": "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",
        "unused_columns": ['speaker_id', 'chapter_id', 'id', 'file'],
        "batch_size": 5,
        "input_dim": 64,
        "model_dim": 512,
        "n_heads": 16,
        "dropout": 0.1,
        "kernel_size": 31,
        "PAD": 28,
        "BLANK": 27,
        "learning_rate": 5e-4
    }

