import matplotlib as plt
import os
from pathlib import Path
os.chdir(Path(os.path.abspath("")).parent)
from mros_data.datamodule import SleepEventDataModule
from mros_data.datamodule.transforms import STFTTransform
import matplotlib.pyplot as plt
import torch
import numpy as np
import ast
import json

def event_dist():
    data_dir = "/scratch/aneol/detr-mros/"
    #data_dir = "D:/10channel"
    params = dict(
        data_dir=data_dir,
        batch_size=1,
        n_eval=0,
        n_test=0,
        num_workers=0,
        seed=1337,
        events={"ar": "arousals", "sdb": "Sleep-disordered breathing", "lm": "Leg Movement"},
        window_duration=600,  # seconds
        cache_data=False,
        default_event_window_duration=[15],
        event_buffer_duration=3,
        factor_overlap=2,
        fs=128,
        matching_overlap=0.5,
        n_jobs=-1,
        n_records=2831 if data_dir == "/scratch/aneol/detr-mros/" else 3,
        # picks=["c3", "c4"],
        picks=["c3", "c4", "eogl", 'eogr', 'chin', 'legl', 'legr', 'nasal', 'abdo', 'thor'],
        # transform=MultitaperTransform(128, 0.5, 35.0, tw=8.0, normalize=True),
        #transform=STFTTransform(
        #    fs=128, segment_size=int(4.0 * 64), step_size=int(0.5 * 64), nfft=1024, normalize=True
        #)
        transform = None,
        scaling="robust",
    )
    dm = SleepEventDataModule(**params)

    # The datamodule will split the dataset into train/eval partitions by calling the setup() method.
    dm.setup('fit')
    train_dl, eval_dl = dm.train_dataloader(), dm.val_dataloader()
    train_ds = dm.train
    event_durs = {0: [], 1: [], 2: []}
    # print(train_ds[0]['signal'].shape)
    for idx, batch in enumerate(train_ds):
        events = batch['events']
        durs = (events[:, 1] - events[:, 0]) * 600
        for event_label in np.unique(events[:, -1]):
            event_durs[event_label].append(list(durs[events[:, -1] == event_label]))

    with open("/scratch/s194277/event_count.json", "w+") as f:
        print('saving')
        json.dump(event_durs, f)

event_dist()