import os
import random
import time
import warnings

import numpy as np
import soundfile
from audiomentations import (
    AddBackgroundNoise,
)
# from audiomentations.augmentations.seven_band_parametric_eq import SevenBandParametricEQ
from audiomentations.core.audio_loading_utils import load_sound_file
from audiomentations.core.transforms_interface import (
    MultichannelAudioNotSupportedException,
)
from scipy.io import wavfile
from tqdm import tqdm

warnings.filterwarnings("ignore")
DEMO_DIR = os.path.abspath('')


class timer(object):

    def __init__(self, description="Execution time", verbose=False):
        self.description = description
        self.verbose = verbose
        self.execution_time = None

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.execution_time = time.time() - self.t
        if self.verbose:
            print("{}: {:.3f} s".format(self.description, self.execution_time))


class BackgroundNoise:

    def add_background_noise(self, op_tts, main_audio, noise, count, path):
        print("Background noise addition started")
        output_dir = os.path.join(DEMO_DIR, "Audio_Data/audio")
        os.makedirs(output_dir, exist_ok=True)

        # provide the folder path where audios are stored
        input_file = os.path.join(DEMO_DIR, main_audio)

        response_dict = {
            'Utterance': [],
            'Response': [],
            'Intent': [],
            'Entity': [],
            'Levels': [],
            'Audio_file': []
        }

        np.random.seed(420)
        random.seed(420)

        sound_file_paths = [os.path.join(DEMO_DIR, input_file)]
        audio_noise_path = os.path.join(DEMO_DIR, "audio_noises\\" + noise)
        # print(wav_files)
        # sound_file_paths = [
        # Path(os.path.join(DEMO_DIR, "op.wav")),
        # hashtag#Path(os.path.join(DEMO_DIR, "perfect-alley1.ogg")),
        # hashtag#Path(os.path.join(DEMO_DIR, "p286_011.wav")),
        # ]

        transforms = [
            # {
            # "instance": AddBackgroundNoise(
            # sounds_path=os.path.join(DEMO_DIR, audio_noise_path), p=1.0
            # ),
            # "num_runs": 1,
            # "name": str(count) + "_" + noise,
            # },
            {
                "instance": AddBackgroundNoise(
                    sounds_path=os.path.join(DEMO_DIR, audio_noise_path),
                    noise_rms="absolute",
                    min_absolute_rms_in_db=-30,
                    max_absolute_rms_in_db=-10,
                    p=1.0,
                ),
                "num_runs": 1,
                "name": str(count) + "_" + noise,
            },
            # {
            # "instance": AddBackgroundNoise(
            # sounds_path=os.path.join(DEMO_DIR, audio_noise_path),
            # min_snr_in_db=2,
            # max_snr_in_db=4,
            # noise_transform=Reverse(p=1.0),
            # p=1.0,
            # ),
            # "num_runs": 1,
            # "name": str(count) + "_" + noise,
            # },
        ]

        for sound_file_path in sound_file_paths:
            samples, sample_rate = load_sound_file(
                sound_file_path, sample_rate=None, mono=False
            )

            if len(samples.shape) == 2 and samples.shape[0] > samples.shape[1]:
                samples = samples.transpose()

            execution_times = {}

            for transform in tqdm(transforms):
                augmenter = transform["instance"]
                run_name = (
                    transform.get("name")
                    if transform.get("name")
                    else transform["instance"].__class__.__name__
                )
                execution_times[run_name] = []
                for i in range(transform["num_runs"]):
                    filename = "{}_{}.wav".format(sound_file_path.split('.')[0], run_name)
                    # split_name = sound_file_path.split('.')[0].split('\\')[8]
                    # print(split_name)
                    response_dict['Utterance'].append(op_tts['Utterance'][count])
                    response_dict['Response'].append(op_tts['Response'][count])
                    response_dict['Intent'].append(op_tts['Intent'][count])
                    response_dict['Entity'].append(op_tts['Entity'][count])
                    response_dict['Levels'].append(op_tts['Levels'][count])
                    response_dict['Audio_file'].append(filename.split("\\")[-1])

                    output_file_path = os.path.join(
                        output_dir,
                        filename
                    )
                    try:
                        with timer() as t:
                            augmented_samples = augmenter(
                                samples=samples, sample_rate=sample_rate
                            )
                        execution_times[run_name].append(t.execution_time)
                        if len(augmented_samples.shape) == 2:
                            augmented_samples = augmented_samples.transpose()

                        wavfile.write(
                            output_file_path, rate=sample_rate, data=augmented_samples
                        )
                    except MultichannelAudioNotSupportedException as e:
                        print(e)
                # j = j+1
            for run_name in execution_times:
                if len(execution_times[run_name]) > 1:
                    print(
                        "{:<32} {:.3f} s (std: {:.3f} s)".format(
                            run_name,
                            np.mean(execution_times[run_name]),
                            np.std(execution_times[run_name]),
                        )
                    )
                else:
                    print(
                        "{:<32} {:.3f} s".format(
                            run_name, np.mean(execution_times[run_name])
                        )
                    )
        for audio_file_path in list(response_dict['Audio_file']):
            data, samplerate = soundfile.read(path + "\\" + audio_file_path)
            soundfile.write(path + "\\" + audio_file_path, data, samplerate, subtype='PCM_16')
        return response_dict