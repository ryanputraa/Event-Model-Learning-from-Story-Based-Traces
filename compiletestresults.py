import os

def print_existing_file_extensions(file_names, directory):
    """
    Prints each existing file name with .json and .txt extensions, including the directory path,
    and displays their contents if they exist.

    Parameters:
    file_names (list of str): List of file names with .json extensions.
    directory (str): Path to the directory containing the files. Use an empty string "" for the current directory.
    """
    for name in file_names:
        # Construct the full path for the .json file
        json_path = os.path.join(directory, f"tests/{name}")
        
        # Derive the corresponding .txt file name by removing the last 5 characters ('.json') and adding '.txt'
        if name.lower().endswith('.json'):
            base_name = name[:-5]
        else:
            base_name = name  # If the file doesn't end with '.json', use the full name
        
        txt_path = os.path.join(directory, f"output/{base_name}.txt")
        
        # Check and print the .json file
        if os.path.exists(json_path):
            print(f"{json_path}:")
            try:
                with open(json_path, 'r', encoding='utf-8') as json_file:
                    content = json_file.read()
                    print(content)
            except Exception as e:
                print(f"Error reading {json_path}: {e}")
        else:
            print(f"{json_path}: File does not exist.")
        
        # Check and print the .txt file
        if os.path.exists(txt_path):
            print(f"{txt_path}:")
            try:
                with open(txt_path, 'r', encoding='utf-8') as txt_file:
                    content = txt_file.read()
                    print(content)
            except Exception as e:
                print(f"Error reading {txt_path}: {e}")
        else:
            print(f"{txt_path}: File does not exist.")
        
        # Separator for readability
        print("-" * 40)

# Example usage:
if __name__ == "__main__":
    json_files = [
        'simple1.json',
        'non_linear1.json',
        'non_linear2.json',
        'non_linear3.json',
        'multiple_fsms1.json',
        'hole_filling1.json',
        'complex1.json',
        'single1.json',
        'cyclic1.json',
        'shared_actions1.json',
        'test_non_linear.json',
        'test_fsms.json',
    ]
    
    # json_files = [
    #     'multiple_fsms1.json',
    #     'complex1.json',
    #     'cyclic1.json',
    #     'test_non_linear.json',
    # ]

    folder = ""  # Specify your directory path here. Use "" for the current directory.
    print_existing_file_extensions(json_files, folder)
