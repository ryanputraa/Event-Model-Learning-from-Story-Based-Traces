import re
def count_string_in_file(filename, search_string, encoding='cp1252'):
    """
    Counts the number of times `search_string` appears in `filename`.

    Parameters:
    - filename (str): Path to the text file.
    - search_string (str): The string to search for.
    - encoding (str): The encoding of the file.

    Returns:
    - int: The count of occurrences.
    """
    try:
        with open(filename, 'r', encoding=encoding) as file:
            contents = file.read()
            count = contents.count(search_string)
        return count
    except FileNotFoundError:
        print(f"Error: The file '{filename}' does not exist.")
        return None
    except UnicodeDecodeError as e:
        print(f"Encoding error: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def extract_max_state_number(filename, encoding='cp1252'):
    """
    Extracts the highest state number +1 from each FSM section in the file.

    Parameters:
    - filename (str): Path to the text file.
    - encoding (str): The encoding of the file.

    Returns:
    - int: Sum of the maximum state numbers (plus 1) from each FSM.
    """
    try:
        with open(filename, 'r', encoding=encoding) as file:
            max_states_sum = 0
            current_max = -1  # Start with -1 to handle empty sections correctly

            for line in file:
                # Detect new FSM section by finding lines starting with 'FSM'
                if line.startswith('FSM for object'):
                    if current_max >= 0:
                        max_states_sum += current_max + 1  # Add 1 to the max state number
                    current_max = -1  # Reset for the new FSM section

                # Find all occurrences of 'State_<number>'
                states = re.findall(r'State_(\d+)', line)
                if states:
                    # Convert to integers and find the maximum in the line
                    max_in_line = max(int(state) for state in states)
                    current_max = max(current_max, max_in_line)

            # Add the last FSM section's max state + 1
            if current_max >= 0:
                max_states_sum += current_max + 1

        return max_states_sum

    except FileNotFoundError:
        print(f"Error: The file '{filename}' does not exist.")
        return None
    except UnicodeDecodeError as e:
        print(f"Encoding error: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():
    # Define the filename and the string to search for
    filename = 'output/traces_fixed.txt'
    search_string = '--['

    # Count the occurrences with specified encoding
    count = count_string_in_file(filename, search_string, encoding='cp1252')

    # Output the result
    if count is not None:
        print(f"The string '{search_string}' appears {count} time(s) in '{filename}'.")

    # Extract the sum of the maximum state numbers
    max_states_sum = extract_max_state_number(filename, encoding='cp1252')

    # Output the result
    if max_states_sum is not None:
        print(f"The sum of the highest state numbers across all FSMs is: {max_states_sum}")

if __name__ == "__main__":
    main()
