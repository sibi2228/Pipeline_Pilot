import os

def load_instructions_from(filename: str) -> str:
    """
    Loads instruction text from a file within the 'instructions' subdirectory.

    Args:
        filename: The name of the text file (e.g., "agent_instructions.txt").

    Returns:
        The content of the file as a string.

    Raises:
        FileNotFoundError: If the specified file does not exist in the
                           'instructions' subdirectory.
    """
    # Construct the full path relative to this util.py file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    instructions_dir = os.path.join(dir_path, "instructions")
    file_path = os.path.join(instructions_dir, filename)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"ERROR: Instruction file not found at {file_path}")
        raise # Re-raise the error so the program stops if instructions are missing
    except Exception as e:
        print(f"ERROR: Failed to read instruction file {file_path}: {e}")
        raise # Re-raise other potential errors

