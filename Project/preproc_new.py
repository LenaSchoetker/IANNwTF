import tensorflow as tf
import os

# Define the paths to your data
mel_spectrogram_dir = "processed_data\\test"
binary_image_dir = "processed_data\\targets\\dev"

# Load the file paths of the mel spectrograms and binary images
mel_spectrogram_paths = tf.io.gfile.glob(os.path.join(mel_spectrogram_dir, "*.png"))[10]
binary_image_paths = tf.io.gfile.glob(os.path.join(binary_image_dir, "*.png"))[10]


# Define the preprocessing function
def preprocess_fn(mel_path, binary_path):
    # Load the mel spectrogram and binary image
    mel = tf.io.decode_png(tf.io.read_file(mel_path))
    binary = tf.io.decode_png(tf.io.read_file(binary_path))

    # # Print the shape of the input images
    # print('Mel spectrogram shape:', tf.shape(mel))
    # print('Binary image shape:', tf.shape(binary))
    
    # Normalize the mel spectrogram to [0, 1]
    mel = mel/255.0
    
    # Convert the binary image to a float and reshape to (height, width, 1)
    binary = tf.cast(binary, tf.float32)
    binary = tf.reshape(binary, (binary.shape[0], binary.shape[1], 1))
    
    return mel, binary


# Create a dataset from the file paths
ds = tf.data.Dataset.from_tensor_slices((mel_spectrogram_paths, binary_image_paths))

# Map the preprocessing function to the dataset
ds = ds.map(lambda x, y: preprocess_fn(x, y), num_parallel_calls=tf.data.AUTOTUNE)

# Shuffle, batch, prefetch the dataset
ds = ds.shuffle(5000).batch(32).prefetch(tf.data.AUTOTUNE)

# for mel_path, binary_path in zip(mel_spectrogram_paths, binary_image_paths):
#     preprocess_fn(mel_path, binary_path)




