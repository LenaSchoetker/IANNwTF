import os
import pandas as pd
import numpy as np
import librosa
import math

import matplotlib.pyplot as plt #nur zum testen, kann später raus

annotations_path = "bbdc_2021_public_data\\final_pre_dataset\\dev-labels.csv" #set path where the give labels are stored
df = pd.read_csv(annotations_path)
labels = ["Shatter", "Doorbell", "Cough", "Church_bell", "Fireworks", "Meow", "Bark", "Shout", "Camera", "Scratching_(performance_technique)", "Burping_and_eructation", "Cheering"]
audio_folder = "processed_data\\segments\\dev"                                #set path where the audio segments are stored
output_dir = "processed_data\\targets\\dev"   

# Define constants for signal processing
n_fft = 2048  # FFT window size
hop_length = 512  # Hop length for mel spectrogram calculation
n_mels = 128  # Number of mel bands

# Group rows by filename
#groups = df.groupby("filename")
groups = df.head(5).groupby("filename")

# Iterate over each group consisting of one audio file with 1-4 sound events
for filename, group in groups:

    # iterating over all  previously created segments from the audio file
    for i in range(1, 11):

        #load the ith segment as a mel-spectogram
        audio_file = os.path.join(audio_folder, f"{filename[:9]}_{i}.wav")
        audio_data, sample_rate = librosa.load(audio_file, sr=None, duration=1.0)
        Spec = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

        # Create empty binary masks for each event label
        masks = {}
        for label in labels:
            masks[label] = np.zeros_like(Spec, dtype=int)
            
        # Iterate over each row in the group (every entry for the audio file)
        for _, row in group.iterrows():

            ################  Frage ###########
            #wir könnten einbauen:
            #duration  = librosa.get_duration(path = audio_file)

            #falls wir irgendwie beweise müssen, dass die wirklich eine Sekunde dauern, aber ist halt mehr rechnen und dann nicht ganz so effizent
            #ALternative nennen wir es nochmal explizit beim Erstellen der Segments
            #################################################################################

            #calculate the segment where the event is active by deviding the onset time by the duration of the audio segments. 
            #Since all segments are 1 second long, we can apply floor function on the onset and the offset 
            start_segment = math.ceil(row["onset"])
            end_segment = math.ceil(row["offset"])

            start_index  = None
            end_index = None

            #the event lies entirly within the segment
            if i == start_segment == end_segment:
                #############################################
                #Jetzt kommt eine sehr merkwürdige Erklärung, ich hoffe, man kann mir iiiirgendwie folgen
                ################################

                #Since we already know the segment of the event and that its duration is 1.0, we do not need to normalize it further.
                #(Subtracting it by the number of segments and multiplying it by the the duration (1.0) would be an unnecessary calculation)
                #we can find the index by look at the fraction of the onset and offset multiplied by the length of the spectogram
                start_index= int((row["onset"] - int(row["onset"]))*len(Spec))
                end_index = int((row["offset"] - int(row["offset"]))*len(Spec))

            #the event starts in the segment 
            if i  == start_segment:
                start_index= int((row["onset"] - int(row["onset"]))*len(Spec))
                end_index = len(Spec)

            #the event ends in the segment
            if i == end_segment:
                start_index= 0
                end_index = int((row["offset"] - int(row["offset"]))*len(Spec))

            # Set the corresponding region in the binary mask to 1 for the event label
            if start_index or end_index:
                mask = masks[row["event_label"]]
                mask[start_index:end_index, :] = 1

                ################### NUR TEST, kann später gelöscht werden ##########################
                # #Plottet die  Maske, die verändert wurde.
                # event = row["event_label"]
                # plt.plot(mask)
                # plt.title(f"Segment {i} of {filename}: Occurance of {event}")
                # plt.show()
                #################################

        # Iterate over each event label and save its binary mask as an image
        for label, mask in masks.items():


            # Create the file name for the binary mask image
            file_name = f"{filename[:9]}_{i}_{label}.png"
            # Save the binary mask as an image in the output directory with the same shape as the spectrogram
            file_path = os.path.join(output_dir, file_name)
            plt.imsave(file_path, mask.transpose(), cmap = "gray")
