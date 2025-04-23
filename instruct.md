### Step-by-Step Instructions

1. **Remove the existing virtual environment (if any)**:

    ```bash
    rm -rf venv310
    ```

    This command deletes the existing `venv310` directory (if it exists), ensuring a fresh environment for Python 3.10.

2. **Install Python 3.10 using Homebrew**:

    ```bash
    brew install python@3.10
    ```

    This command installs Python 3.10 on your system using Homebrew. Make sure Homebrew is installed on your macOS before running this command.

3. **Create a virtual environment using Python 3.10**:

    ```bash
    python3.10 -m venv venv310
    ```

    After installing Python 3.10, create a new virtual environment named `venv310`.

4. **Activate the virtual environment**:

    ```bash
    source venv310/bin/activate
    ```

    This command activates the `venv310` virtual environment. Once activated, any Python packages you install will be contained within this environment.

5. **Deactivate the virtual environment** (optional, if you're done working in the virtual environment):

    ```bash
    deactivate
    ```

    This command deactivates the virtual environment and returns you to the system's global Python environment.

6. **Install the required Python packages**:

    ```bash
    pip install streamlit tensorflow-macos tensorflow-metal pillow numpy pandas
    ```

    This command installs the necessary packages for your project, including `streamlit`, `tensorflow-macos`, `tensorflow-metal`, `pillow`, `numpy`, and `pandas`.

7. **Run the Streamlit app**:
    ```bash
    streamlit run plant_disease_app.py
    ```
    Finally, this command runs the `plant_disease_app.py` script using Streamlit. Make sure this script is in your working directory or provide the full path to the script.
