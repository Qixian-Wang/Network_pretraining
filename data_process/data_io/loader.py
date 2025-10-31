import os

from dataclasses import dataclass
import pickle
from typing import Optional, Sequence


@dataclass
class Dataset:
    session_name: str
    spike_train: object
    stimulation_init_time: Optional[Sequence[float]]


class DataLoader:
    @staticmethod
    def load_spike_train(path) -> object:
        dataset_list = []
        for file in os.listdir(path):
            if file.startswith("spike_train"):
                dataset = Dataset(session_name=None, spike_train=None, stimulation_init_time=None)
                spike_train_path = os.path.join(path, file)

                with open(spike_train_path, "rb") as f:
                    spike_train = pickle.load(f)
                    session_name = file.split("_train")[1].split(".pkl")[0]
                    dataset.session_name = session_name
                    dataset.spike_train = spike_train
                    if session_name.startswith("train") or session_name.startswith("pretrain"):
                        stimulation_init_time_path = os.path.join(path, f"{session_name}_stimulation_init_time.pkl")
                        with open(stimulation_init_time_path, "rb") as f:
                            stim_times = pickle.load(f)
                            dataset.stimulation_init_time = stim_times
                
                dataset_list.append(dataset)
        return dataset_list


    def load(self, data_path):
        return self.load_spike_train(data_path)