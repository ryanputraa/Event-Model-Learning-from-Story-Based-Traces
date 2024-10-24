# Adapted LOCM2 Algorithm

This repository contains the implementation of the Adapted LOCM2 algorithm for learning Finite State Machines (FSMs) from story-based data. The algorithm processes narrative event traces to generate FSMs that model the behaviors of objects within the narratives.

## Files and Descriptions

### 1. `AdaptedLOCM2.py`

- **Description:** This is the main code file containing the implementation of the Adapted LOCM2 algorithm. It processes input data in JSON format, representing events, objects, and their relationships, and generates FSMs for the objects involved.
- **Usage:** Run this script to process the input data and generate FSM outputs.

### 2. `AdaptedLOCM2_Testing.py`

- **Description:** This script is used for testing and evaluating the Adapted LOCM2 algorithm. It tracks various statistics such as processing time, memory usage, and accuracy metrics.
- **Usage:** Use this script to perform tests on different datasets and observe the performance of the algorithm.

### 3. `fixjson.py`

- **Description:** A utility script that fixes broken JSON formats. It is particularly useful when the original `traces.json` file has formatting issues, such as misplaced brackets or commas.
- **Usage:** Run this script to repair malformed JSON files before processing them with the main algorithm.

### 4. `testing_utilities.py`

- **Description:** This script contains miscellaneous functions that assist with testing. These include functions for data generation, result comparison, and other helper methods used during the testing phase.
- **Usage:** This script is imported by `AdaptedLOCM2_Testing.py` and can also be used independently for custom testing purposes.

## Getting Started

### Prerequisites

- **Python 3.x**: Ensure you have Python 3 installed on your system.
- **Required Libraries**: The scripts use standard Python libraries such as `json`, `logging`, and `collections`. No additional external libraries are required.

### Running the Main Algorithm

1. **Prepare the Input Data**: Ensure that your input data is in the correct JSON format, similar to the examples provided.

2. **Fixing JSON Files (If Necessary)**: If your JSON files have formatting issues, use `fixjson.py` to correct them.

   ```bash
   python fixjson.py input.json fixed_input.json
   ```

3. **Run the Adapted LOCM2 Algorithm**:

   ```bash
   python AdaptedLOCM2.py
   ```

   By default, the script will process the JSON files specified within it and generate FSM outputs in the `output/` directory.

### Testing and Evaluation

Use `AdaptedLOCM2_Testing.py` to run tests and track statistics.

```bash
python AdaptedLOCM2_Testing.py
```

This script will execute the algorithm on test datasets and provide detailed statistics on its performance.

## Directory Structure

- `AdaptedLOCM2.py`: Main algorithm implementation.
- `AdaptedLOCM2_Testing.py`: Testing and evaluation script.
- `fixjson.py`: JSON fixing utility.
- `testing_utilities.py`: Helper functions for testing.
- `tests/`: Directory containing test JSON files.
- `output/`: Directory where FSM outputs are saved.

## Example Usage

**Processing a Single JSON File:**

```bash
python AdaptedLOCM2.py --input tests/example1.json --output output/example1_output.txt
```

**Fixing a JSON File:**

- edit manually, then
```bash
python fixjson.py 
```

## Notes

- Ensure that your input JSON files are properly formatted before running the main algorithm.
- The `AdaptedLOCM2.py` script may need to be adjusted if you have specific input or output file requirements.
- The testing scripts are useful for evaluating the algorithm's performance on different datasets and configurations.

## Contact

For any questions or issues, please contact [Your Name] at [Your Email Address].

---

Feel free to adjust the descriptions and usage instructions to better fit your specific implementation and to include any additional information that might be helpful for users of your code.