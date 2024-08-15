from collections import defaultdict
from pathlib import Path

import numpy as np
import zipfile
import os
import json
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import heapq
import time
import pickle
import math
import re
import inspect

model_directory = '../models'
samples_wavetables_path = '../misc/full_samples_and_wavetables.json'
default_values_path = '../misc/default_values.json'
gen_num = 10
output_directory = '../vital_output'

# preset synthesis

def preset_synthesis(synthesizer, gen_num):
    gen_df = synthesizer.sample(num_rows=gen_num)
    return gen_df

def create_output_folder(name):    
    # Full path to the new folder
    output_preset_dir = os.path.join(output_directory, name)
    # Create the new folder
    os.makedirs(output_preset_dir, exist_ok=True)
    return output_preset_dir

def post_processing(gen_dir, new_df):
    # Function to process each group
    def process_group(group):
        # Separate X and Y pairs
        x_cols = group.columns[::2]
        y_cols = group.columns[1::2]

        new_data = []
        for idx, row in group.iterrows():
            # Extract X and Y values
            xy_pairs = [(row[x], row[y]) for x, y in zip(x_cols, y_cols)]

            # Remove pairs where either X or Y is NaN
            valid_pairs = [(x, y) for x, y in xy_pairs if not (np.isnan(x) or np.isnan(y))]

            # Compare X values and remove the right one if it's smaller than the left one
            filtered_pairs = []
            for i in range(len(valid_pairs)):
                if i == 0 or valid_pairs[i][0] >= valid_pairs[i - 1][0]:
                    filtered_pairs.append(valid_pairs[i])

            # Fill the new row data
            new_row = [val for pair in filtered_pairs for val in pair]
            new_row.extend([np.nan] * (len(xy_pairs) * 2 - len(new_row)))  # Pad with NaNs if needed

            new_data.append(new_row)

        # Reconstruct the group with the new filtered data
        new_df_group = pd.DataFrame(new_data, columns=group.columns, index=group.index)
        return new_df_group

    # Apply the processing function to each group in new_df
    grouped = new_df.columns.str.extract(r'(.*)-points-')[0]
    for group_name in grouped.unique():
        group_cols = new_df.columns[grouped == group_name]
        new_df[group_cols] = process_group(new_df[group_cols])
    def nested_dict():
        """Helper function to create a nested defaultdict of defaultdicts"""
        return defaultdict(nested_dict)

    def unflatten_and_save_json(df, directory, base_filename="output"):
        """Function to unflatten DataFrame and save JSON files to a directory"""

        # Ensure the directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)

        for idx, row in df.iterrows():
            # Create a nested dictionary
            root = nested_dict()

            for col, val in row.items():
                keys = col.split('-')
                d = root
                for key in keys[:-1]:
                    d = d[key]
                d[keys[-1]] = val

            # Convert defaultdict to regular dict
            json_obj = json.loads(json.dumps(root))

            # Define filename
            filename = os.path.join(directory, f"{base_filename}_{idx + 1}.json")

            # Save JSON object to file
            with open(filename, 'w') as f:
                json.dump(json_obj, f, indent=4)

            # Confirmation print
            print(f"Saved {filename}")

    # Example usage
    # Assuming you have `new_df` DataFrame ready
    unflatten_and_save_json(new_df, gen_dir, base_filename="AIU_Preset")

    # Fields to add at the beginning of the JSON
    starting_fields = {
        "author": "AIU",
        "comments": "FLY AI Challenger",
        "macro1": "MACRO 1",
        "macro2": "MACRO 2",
        "macro3": "MACRO 3",
        "macro4": "MACRO 4"
    }

    # Field to add at the end of the JSON
    ending_field = {"synth_version": "1.5.5"}

    # Iterate over all files in the directory
    for filename in os.listdir(gen_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(gen_dir, filename)

            # Read the current content of the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)

            # Insert the starting fields at the beginning of the JSON content
            updated_data = {**starting_fields, **data}

            # Add the ending field at the end of the JSON content
            updated_data.update(ending_field)

            # Write the updated content back to the JSON file
            with open(file_path, 'w') as file:
                json.dump(updated_data, file, indent=4)

    def unflatten_json(data):
        """
        Recursively unflatten the JSON object by converting numeric keys into list elements.
        """
        if isinstance(data, dict):
            numeric_keys = [key for key in data.keys() if key.isdigit()]
            if numeric_keys:
                # Sort numeric keys to preserve order in the list
                numeric_keys.sort(key=int)
                return [unflatten_json(data[key]) for key in numeric_keys]
            else:
                # Recursively apply unflattening to nested dictionaries
                return {key: unflatten_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            # Recursively apply unflattening to lists
            return [unflatten_json(item) for item in data]
        else:
            return data

    def process_json_file(file_path):
        """
        Load a JSON file, unflatten the data if necessary, and save the corrected data.
        """
        with open(file_path, 'r') as file:
            data = json.load(file)

        unflattened_data = unflatten_json(data)

        with open(file_path, 'w') as file:
            json.dump(unflattened_data, file, indent=4)

    def process_json_files_in_directory(directory):
        """
        Process all JSON files in the specified directory.
        """
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                file_path = os.path.join(directory, filename)
                process_json_file(file_path)



    # Process all JSON files in the directory
    process_json_files_in_directory(gen_dir)

    def clean_nans_in_arrays(data):
        if isinstance(data, dict):
            for key, value in data.items():
                if key in ["points", "powers"] and isinstance(value, list):
                    data[key] = [x for x in value if not (isinstance(x, float) and math.isnan(x))]
                else:
                    clean_nans_in_arrays(value)
        elif isinstance(data, list):
            for item in data:
                clean_nans_in_arrays(item)

    def process_json_files(directory):
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r') as file:
                    data = json.load(file)

                clean_nans_in_arrays(data)

                with open(file_path, 'w') as file:
                    json.dump(data, file, indent=4)


    process_json_files(gen_dir)

    def clean_points_and_remove_empty_parents(data):
        if isinstance(data, dict):
            keys_to_delete = []
            for key, value in data.items():
                if isinstance(value, dict):
                    clean_points_and_remove_empty_parents(value)
                    # Special handling for "points" key within the nested dictionary
                    if "points" in value and isinstance(value["points"], list) and len(value["points"]) <= 1:
                        # Mark the entire parent key for deletion
                        keys_to_delete.append(key)

                elif isinstance(value, list):
                    # Clean nested lists
                    clean_points_and_remove_empty_parents(value)

            # Remove the identified parent keys if "points" was empty or had one item
            for key in keys_to_delete:
                del data[key]

        elif isinstance(data, list):
            items_to_delete = []
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    # Avoid deletion if the parent key is "lfos"
                    if "points" in item and isinstance(item["points"], list) and len(item["points"]) <= 1 and not math.isnan(item.get("name", 0)):
                        # Mark the entire object for deletion
                        items_to_delete.append(i)
                    else:
                        clean_points_and_remove_empty_parents(item)

            # Remove the identified items from the list in reverse order to avoid index shifting
            for i in reversed(items_to_delete):
                del data[i]

    def process_json_files(directory):
        for root, dirs, files in os.walk(directory):
            # Skip the lfos directory nested inside the settings directory
            if 'settings/lfos' in root or 'settings\\lfos' in root:  # Account for different OS path separators
                continue

            for filename in files:
                if filename.endswith('.json'):
                    file_path = os.path.join(root, filename)
                    with open(file_path, 'r') as file:
                        data = json.load(file)

                    clean_points_and_remove_empty_parents(data)

                    with open(file_path, 'w') as file:
                        json.dump(data, file, indent=4)

    # Replace 'gen_dir' with the actual directory you want to process
    process_json_files(gen_dir)

    # Function to replace NaN with empty string in JSON data
    def replace_nan_with_empty_string(data):
        if isinstance(data, dict):
            return {key: replace_nan_with_empty_string(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [replace_nan_with_empty_string(item) for item in data]
        elif isinstance(data, float) and np.isnan(data):
            return ""
        else:
            return data

    # Iterate through all files in the gen_dir
    for filename in os.listdir(gen_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(gen_dir, filename)

            # Read the JSON file
            with open(file_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)

            # Replace NaN values
            updated_data = replace_nan_with_empty_string(json_data)

            # Write the updated JSON data back to the file
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(updated_data, file, ensure_ascii=False, indent=4)

    # Define the keys to check
    keys_to_check = ["osc_1_spectral_morph_phase",
                    "osc_2_spectral_morph_phase",
                    "osc_3_spectral_morph_phase"]

    # Iterate over all files in the directory
    for filename in os.listdir(gen_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(gen_dir, filename)

            # Open the JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Check if 'settings' is in the JSON and it's a dictionary
            if 'settings' in data and isinstance(data['settings'], dict):
                # Check each key and delete if the value is an empty string
                for key in keys_to_check:
                    if key in data['settings'] and data['settings'][key] == "":
                        del data['settings'][key]

            # Save the modified JSON back to the file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)

    def adjust_points_and_powers(data):
        if isinstance(data, dict):
            for key, value in data.items():
                if key == "points" and isinstance(value, list):
                    # Ensure the number of points is even
                    if len(value) % 2 != 0:
                        value.pop()  # Remove the last element if odd
                    data['num_points'] = len(value) // 2

                if key == "powers" and isinstance(value, list) and 'num_points' in data:
                    # Trim powers to match num_points
                    data[key] = value[:data['num_points']]

                adjust_points_and_powers(value)
        elif isinstance(data, list):
            for item in data:
                adjust_points_and_powers(item)

    def process_json_files(gen_dir):
        for filename in os.listdir(gen_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(gen_dir, filename)
                with open(file_path, 'r') as file:
                    data = json.load(file)

                adjust_points_and_powers(data)

                with open(file_path, 'w') as file:
                    json.dump(data, file, indent=4)


    process_json_files(gen_dir)
    # Iterate over all files in the directory
    for filename in os.listdir(gen_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(gen_dir, filename)

            # Open and load the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)

            # Process the modulations array
            if 'modulations' in data.get('settings', {}):
                data['settings']['modulations'] = [
                    modulation for modulation in data['settings']['modulations']
                    if modulation.get('destination') and modulation.get('source')
                ]

            # Save the modified JSON back to the file
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)
    # Iterate over all files in the directory
    for filename in os.listdir(gen_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(gen_dir, filename)

            # Open and load the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)

            # Modify the 'random_values' list by removing items with an empty 'seed'
            data['settings']['random_values'] = [rv for rv in data['settings']['random_values'] if rv.get('seed') != ""]

            # Save the modified JSON back to the file
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)
    def replace_smooth_key(obj):
        """Recursively replace 'smooth' key's empty string values with False."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "smooth" and value == "":
                    obj[key] = False
                else:
                    replace_smooth_key(value)
        elif isinstance(obj, list):
            for item in obj:
                replace_smooth_key(item)

    def process_json_files(directory):
        """Process all JSON files in the given directory."""
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r') as file:
                    data = json.load(file)

                replace_smooth_key(data)

                with open(file_path, 'w') as file:
                    json.dump(data, file, indent=4)


    process_json_files(gen_dir)

    valid_wavetables_names = [
        "Init", "Basic Shapes", "Brown Noise", "Classic Blend", "Classic Fade", "Harmonic Series", "Pink Noise",
        "Pulse Width", "Quad Saw", "Three Graces", "White Noise", "Drink the Juice", "Low High Fold",
        "Post Modern Pulse", "Saws for Days", "Stabbed", "Vital Sine 2", "Vital Sine", "Didg", "Jaw Harp",
        "Acid Rock ft Maynix", "Additive Squish 2", "Alternating Harmonics", "Clicky Robot", "Corpusbode Phaser",
        "Crappy Toilet", "Creepy Solar", "Flange Sqrowl", "Granular Upgrade", "Hollow Distorted FM", "Solar Powered",
        "Squish Flange", "Thank u False Noise", "Water Razor",
        "Analog_BD_Sin" #외부 파형 중 개수 그나마 많은거 1개 추가
    ]

    valid_sample_names = [
        'BART', 'Box Fan', 'Grinder', 'HVAC Unit', 'Jack Hammer',
        'River', 'Waves', 'Brown Noise', 'Pink Noise', 'White Noise',
        "BrightWhite", "Cassette Bell C Low", "3osc Noise" #외부 샘플 중 개수 그나마 많은거 3개 추가
    ]


    def modify_json_structure(obj):
        """Recursively search and modify the 'name' keys in 'wavetables' and 'sample-name'."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "wavetables" and isinstance(value, list):
                    for wavetable in value:
                        if isinstance(wavetable, dict) and "name" in wavetable:
                            if wavetable["name"] not in valid_wavetables_names:
                                wavetable["name"] = "Init"
                elif key == "sample" and isinstance(value, dict) and "name" in value:
                    if value["name"] not in valid_sample_names:
                        value["name"] = "White Noise"
                else:
                    modify_json_structure(value)
        elif isinstance(obj, list):
            for item in obj:
                modify_json_structure(item)

    def process_json_files(directory):
        """Process all JSON files in the given directory."""
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r') as file:
                    data = json.load(file)

                modify_json_structure(data)

                with open(file_path, 'w') as file:
                    json.dump(data, file, indent=4)


    # Replace 'gen_dir' with the actual directory containing the JSON files
    process_json_files(gen_dir)

    # Load the JSON file that contains the full samples and wavetables data
    with open(samples_wavetables_path, 'r') as file:
        full_data = json.load(file)

    full_sample_dict = full_data.get("full samples", {})
    full_wavetables_dict = full_data.get("full wavetable", {})

    def replace_with_full_data(obj):
        """Recursively replace sample and wavetable names with full dictionaries."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "sample" and isinstance(value, dict) and "name" in value:
                    sample_name = value["name"]
                    if sample_name in full_sample_dict:
                        obj["sample"] = full_sample_dict[sample_name]
                elif key == "wavetables" and isinstance(value, list):
                    for i in range(len(value)):
                        if "name" in value[i]:
                            wavetable_name = value[i]["name"]
                            if wavetable_name in full_wavetables_dict:
                                value[i] = full_wavetables_dict[wavetable_name]
                else:
                    replace_with_full_data(value)
        elif isinstance(obj, list):
            for item in obj:
                replace_with_full_data(item)

    def process_json_files(directory):
        """Process all JSON files in the given directory."""
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r') as file:
                    data = json.load(file)

                replace_with_full_data(data)

                with open(file_path, 'w') as file:
                    json.dump(data, file, indent=4)

    # Set the directory path to your JSON files
    directory_path = gen_dir
    process_json_files(directory_path)

    # Replacement structure for empty "points" arrays in "lfos"
    replacement_lfo = {
        "name": "Triangle",
        "num_points": 3,
        "points": [
            0.0,
            1.0,
            0.5,
            0.0,
            1.0,
            1.0
        ],
        "powers": [
            0.0,
            0.0,
            0.0
        ],
        "smooth": False
    }

    # Iterate over all files in the specified directory
    for filename in os.listdir(gen_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(gen_dir, filename)

            # Load the JSON data
            with open(file_path, 'r') as file:
                data = json.load(file)

            # Check and replace the LFOs with empty "points"
            if "settings" in data and "lfos" in data["settings"]:
                updated = False
                for i, lfo in enumerate(data["settings"]["lfos"]):
                    if "points" in lfo and not lfo["points"]:
                        data["settings"]["lfos"][i] = replacement_lfo
                        updated = True

                # If any LFO was updated, save the changes
                if updated:
                    with open(file_path, 'w') as file:
                        json.dump(data, file, indent=4)



    #        print(f"Processed {filename}")

    # Load default values from the provided file
    with open(default_values_path, 'r') as default_file:
        default_values = json.load(default_file)


    # Function to update JSON with default values
    def update_json_with_defaults(json_data, defaults):
        for key, value in json_data.items():
            if isinstance(value, dict):  # Recursively update nested dictionaries
                json_data[key] = update_json_with_defaults(value, defaults.get(key, {}))
            elif value == "" and key in defaults:
                json_data[key] = defaults[key]
        return json_data

    # Iterate over all JSON files in gen_dir
    for filename in os.listdir(gen_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(gen_dir, filename)

            # Load the JSON file
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)

            # Update the JSON data with default values
            updated_data = update_json_with_defaults(data, default_values)

            # Save the updated JSON back to the file
            with open(file_path, 'w') as json_file:
                json.dump(updated_data, json_file, indent=4)

    # Change the extension from .json to .vital for all files in gen_dir
    for filename in os.listdir(gen_dir):
        if filename.endswith('.json'):
            # Construct the full file path
            json_path = os.path.join(gen_dir, filename)
            # Replace .json with .vital
            vital_filename = filename.replace('.json', '.vital')
            vital_path = os.path.join(gen_dir, vital_filename)
            # Rename the file
            os.rename(json_path, vital_path)



