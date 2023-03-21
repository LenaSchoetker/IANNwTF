import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# set the path to the folder containing the audio files
audio_folder = "processed_data\\segments\\dev" 

# set the path to save the mel spectrograms
output_folder = "processed_data\\test" 

# set the parameters for the mel spectrograms
n_fft = 2048
hop_length = 512
n_mels = 128

# iterate over the audio files in the folder
for file_name in os.listdir(audio_folder):
    if file_name.endswith('.wav'):
        # load the audio file
        audio_file = os.path.join(audio_folder, file_name)
        y, sr = librosa.load(audio_file, sr=22050)

        # compute the mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        
        # convert to log scale (dB)
        log_S = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(128, 32))

        # plot the mel spectrogram
        librosa.display.specshow(log_S, sr=sr, hop_length=hop_length)

        # save the mel spectrogram
        output_file = os.path.join(output_folder, file_name[:-4] + '.png')
        plt.tight_layout()
        plt.savefig(output_file, dpi =1)
        plt.close()
