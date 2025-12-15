import os

from mpi4py import MPI
import numpy as np
import pickle

from miv.io.intan.data import DataIntan
from miv.io.openephys import Data, DataManager

from miv.signal.filter import ButterBandpass
from miv.signal.spike import ThresholdCutoff
from miv.core.operator.policy import StrictMPIRunner
from miv.core.pipeline import Pipeline

def _bank_encode(channel_name):
    bank = channel_name[0].upper()
    idx = int(channel_name[-3:])
    base = {"A": 0, "B": 32, "C": 64, "D": 96}[bank]
    return base + idx

def extract_stimulation_init(digital_in_stamps):
        data = digital_in_stamps.data
        assert len(data) == 4, "digit in channel number is wrong"

        flatten_data = np.array(sorted(sum(data, [])))
        timestamps = np.array([])

        timestamps = np.append(timestamps, flatten_data[0])
        
        time_pre = np.array(flatten_data[:-1])
        time_next = np.array(flatten_data[1:])
        time_interval = time_next - time_pre
        timestamps = np.append(timestamps, time_next[np.where(time_interval > 0.5)])
                    
        return timestamps



if __name__ == "__main__":
    device = "OpenEphys"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    data_folder = "/scratch2/10197/qxwang/recording_data/Experiment_11_28"
    session_names = ["spontaneous1", "spontaneous2", "spontaneous3", "pretrain1", "pretrain2", "pretrain3", "pretrain4", "pretrain5", "train0", "train1", "train2", "train3", "train4", "train5", "train6"]
    data_paths = [os.path.join(data_folder, session_name) for session_name in session_names]

    excluded_channels = ['A-005', 'A-031', 'B-002', 'B-018', 'B-028', 'B-029', 'B-030', 'C-031', 'D-016', 'D-017', 'D-018', 'D-020']
    excluded_channels = [_bank_encode(channel) for channel in excluded_channels]

    data_path = data_paths[rank]
    session_name = session_names[rank]

    spike_train_save_path = f"spike_train/"
    pipline_data_save_path = f"results/{session_name}"

    if device == "Intan":
        data = DataIntan(data_path)
    else:
        data = Data(data_path)

    bandpass_filter = ButterBandpass(lowcut=200, highcut=1600, order=4)
    spike_detection = ThresholdCutoff(exclude_channels=excluded_channels, cutoff=5.0) 
    spike_detection.runner = StrictMPIRunner(comm=comm)

    data >> bandpass_filter >> spike_detection
    pipeline = Pipeline(spike_detection)
    pipeline.run(working_directory=pipline_data_save_path)
    
    spike_train = spike_detection.output()

    os.makedirs(spike_train_save_path, exist_ok=True)
    with open(os.path.join(spike_train_save_path, f"spike_train{session_name}.pkl"), "wb") as f:
        pickle.dump(spike_train, f)

    if session_name.startswith("train") or session_name.startswith("pretrain"):
        digital_in_data = data.load_digital_in_event(progress_bar=False)
        stimulation_init_time = extract_stimulation_init(digital_in_data)
        with open(os.path.join(spike_train_save_path, f"{session_name}_stimulation_init_time.pkl"), "wb") as f:
            pickle.dump(stimulation_init_time, f)
