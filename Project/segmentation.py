import librosa
import os
import soundfile as sf

"Seperates audio files into num_segments=10 segments and saves them into end_path"

dataset_path = "bbdc_2021_public_data\\final_pre_dataset\\eval"  #set path where dataset is stored
end_path = "processed_data\\segments\\eval" #set path where the segments are about to be stored

#create segments for each file in the dataset
for file_name in os.listdir(dataset_path):
    if file_name.endswith('.wav'):
        file_path = os.path.join(dataset_path, file_name) #get the path of the current file
        audio_data, sr = librosa.load(file_path, sr=None) #load the audio file with the native sampling rate

        #set the number and length of the segments
        num_segments = 10
        segment_length = len(audio_data) // num_segments

        for i in range(num_segments):
            segment_start = i * segment_length  #start point of the ith segment
            segment_end = (i + 1) * segment_length #end point of the ith segment
            segment_data = audio_data[segment_start:segment_end] #slice the audio to the segment

            #save the segment at the end path 
            file_name = file_name.replace(".wav", "")
            segment_file_name = f'{file_name}_{i+1}.wav'
            segment_file_path = os.path.join(end_path, segment_file_name)

            sf.write(segment_file_path, segment_data, sr)
