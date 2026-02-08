# let_flownet

## Setup
This project uses `uv` for dependency management and package building.

### 1. Install uv
If you haven't installed `uv` yet:
```bash
pip install uv
```

### 2. Install Dependencies
Run the following command to create a virtual environment and install all required packages (including compiling the `Forward_Warp` CUDA extension):
```bash
uv sync
```

### 3. Activate Environment
```bash
source .venv/bin/activate
```

## Running Tests

To verify that the environment is set up correctly and the custom CUDA modules are compiled:

```bash
python -c "import torch; import Forward_Warp; print('Forward_Warp imported successfully')"
```
