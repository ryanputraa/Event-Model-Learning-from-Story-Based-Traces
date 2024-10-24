import json

try:
    # Load file and manually fix any concatenated objects by adding commas and brackets
    with open('traces.json', 'r') as f:
        content = f.read()
        # Add comma between objects if missing and wrap in brackets
        fixed_content = '[{}]'.format(content.replace('}{', '},{'))
        
    # Try to load the fixed JSON
    data_list = json.loads(fixed_content)

    # Write the fixed content back to the file (optional)
    with open('traces_fixed.json', 'w') as fixed_file:
        fixed_file.write(fixed_content)

    print("Successfully fixed and loaded the JSON.")
except json.JSONDecodeError as e:
    print(f"Still encountering JSONDecodeError: {str(e)}")
