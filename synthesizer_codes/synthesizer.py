from collections import defaultdict
from pathlib import Path

import numpy as np
import os
import pandas as pd
import sdv
import shutil
import matplotlib.pyplot as plt
import heapq
import time
import pickle
import math

from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

data_directory = '../data'
metadata_directory = '../metadata'
model_directory = '../models'

def generate_metadata(df, name):
    # Create metadata object
    metadata = SingleTableMetadata()
    
    # Detect metadata from the DataFrame
    metadata.detect_from_dataframe(df)
    
    # Handle unknown columns if they exist
    unknown_columns = metadata.get_column_names(sdtype='unknown')
    if unknown_columns:
        metadata.update_columns(
            column_names=unknown_columns,
            sdtype='categorical'
        )
    
    # Optionally, set a primary key if needed
    # metadata.set_primary_key(column_name='preset_name')  # Uncomment and modify if you have a primary key
    
    # Validate the metadata
    metadata.validate()
    metadata.validate_data(data=df)
    
    # Construct the metadata filename using the DataFrame variable name
    metadata_filename = f'{name}_metadata.json'
    
    # Construct the full file path using the global metadata_directory
    full_file_path = os.path.join(metadata_directory, metadata_filename)
    
    # Save metadata to JSON in the specified directory
    metadata.save_to_json(filepath=full_file_path)

    # Return the metadata object
    return metadata


def train_and_save_synthesizer(df, metadata, name):
    
    # Step 1: Create the synthesizer
    synthesizer = GaussianCopulaSynthesizer(metadata)
    
    # Step 2: Train the synthesizer
    start_time = time.time()
    synthesizer.fit(df)
    end_time = time.time()
    
    # Calculate the time taken
    training_time = end_time - start_time
    
    # Convert the time to hours, minutes, and seconds
    hours, rem = divmod(training_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    # Display the formatted time
    print(f"Model training took {int(hours):02}:{int(minutes):02}:{int(seconds):02} (hh:mm:ss)")
    
    # Step 3: Save the trained model locally
    # Ensure the model directory exists
    os.makedirs(model_directory, exist_ok=True)
    
    # Construct the model filename using the DataFrame variable name
    model_filename = f'{name}_synthesizer.pkl'
    model_file_path = os.path.join(model_directory, model_filename)
    
    with open(model_file_path, 'wb') as file:
        pickle.dump(synthesizer, file)
    
    print(f"Model saved to {model_file_path}")
    return synthesizer

# Example usage:
# Assuming df_bass is your DataFrame
# metadata = generate_metadata(df_bass)  # Generate or load your metadata
# train_and_save_synthesizer(df_bass, metadata)


def generate_synthesizer(df, name):
    metadata = generate_metadata(df, name)
    model = train_and_save_synthesizer(df, metadata, name)
    return model

