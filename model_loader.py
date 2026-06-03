import json
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow as tf


class SpecAugment(Layer):
   
    def __init__(self, freq_mask_max=20, time_mask_max=30,
                 n_freq_masks=2, n_time_masks=2, **kwargs):
        super().__init__(**kwargs)
        self.freq_mask_max = freq_mask_max
        self.time_mask_max = time_mask_max
        self.n_freq_masks  = n_freq_masks
        self.n_time_masks  = n_time_masks

    def call(self, x, training=False):
        return x   

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(freq_mask_max=self.freq_mask_max,
                        time_mask_max=self.time_mask_max,
                        n_freq_masks=self.n_freq_masks,
                        n_time_masks=self.n_time_masks))
        return cfg


def load_assets():
    model = load_model("Urban_Sound_Model.keras",
                       custom_objects={"SpecAugment": SpecAugment})
    with open("norm_stats.json") as f:
        stats = json.load(f)
    return model, stats["mean"], stats["std"]