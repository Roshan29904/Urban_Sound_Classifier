import numpy as np
import librosa

from config import SAMPLE_RATE, SAMPLE, N_MELS


def preprocess_audio(file_path):
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    if len(signal) > SAMPLE:
        signal = signal[:SAMPLE]
    elif len(signal) < SAMPLE:
        signal = np.pad(signal, (0, SAMPLE - len(signal)), "constant")

    signal = signal.astype(np.float32)
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    sample_mean = mel_db.mean()
    sample_std  = mel_db.std()
    mel_db_norm = (mel_db - sample_mean) / (sample_std + 1e-8)

    if mel_db_norm.shape[1] > 173:
        mel_db_norm = mel_db_norm[:, :173]
    elif mel_db_norm.shape[1] < 173:
        pad_width = 173 - mel_db_norm.shape[1]
        mel_db_norm = np.pad(mel_db_norm, ((0, 0), (0, pad_width)), "constant")

    return mel_db_norm[np.newaxis, ..., np.newaxis]