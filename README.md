
# ALMA Cold Region Tracker Dynamic Event Detection and Analysis

## External Libraries

This project uses an external library file `salat.py`, which is located in the `src/external_libs/` directory. This file is not part of the standard Python libraries and is required for ALMA data processing.

## Setup

1. Ensure you have Python 3.9 or later installed.
2. Clone this repository.
3. Install the required packages:                                                                                                                                  
   pip install -r requirements.txt
   ```
4. Add the `src` directory to your PYTHONPATH:
   ```
   export PYTHONPATH=$PYTHONPATH:/path/to/your/project/src
   ```

## Running Tests

To run the tests, use the following command from the project root:

```
pytest
```

## CI/CD

The CI/CD pipeline is set up to automatically add the `src` directory to the PYTHONPATH before running tests and linting.

