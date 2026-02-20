# Code: The State of Scientific Poster Sharing and Reuse Paper

## About
Scientific posters are one of the most common forms of scholarly communication, with millions presented at conferences each year. They contain early-stage insights that, if shared beyond the conference, could accelerate scientific discovery. This repository contains the code related to our analysis of poster sharing and reuse. See this [inventory](https://github.com/fairdataihub/poster-sharing-reuse-paper-inventory) for all related resources, including the paper.

## Standards followed
The overall code is structured according to the [FAIR-BioRS guidelines](https://fair-biors.org/). The Python code in the Jupyter notebook [main.ipynb](main.ipynb) follows the [PEP8 guidelines](https://peps.python.org/pep-0008). All the dependencies are documented in the [environment.yml](environment.yml) file.

## Getting Started

### Prerequisites

This project contains both Jupyter notebooks (`.ipynb`) and Python scripts (`.py`). You can set up your environment using **either** `venv` or `conda`.

> **Note:** `venv` is lightweight and works well for running Python scripts. `conda` may be more convenient for managing Jupyter notebook environments. Both options install the same dependencies.

- **Python 3.12+** (for venv) or **[Anaconda (Python 3 version)](https://www.anaconda.com/products/individual) / Miniconda** (for conda)
- **Git** (to clone the repository)

### Clone the repository

Clone the repo or download as a zip and extract it.

### Navigate to the code directory

Open Anaconda prompt (Windows) or the system Command line interface then navigate to the code
```bash
cd poster-metadata-quality-code
```

### Setup environment

#### Option A: Using venv

##### 1. Create and activate the virtual environment
```bash
python -m venv venv
```

Activate it:
- **Windows:** `venv\Scripts\activate`
- **macOS / Linux:** `source venv/bin/activate`

##### 2. Install dependencies and register Jupyter kernel
```bash
pip install -r requirements.txt
```
##### 3. Set up Jupyter kernel (for notebooks only)

```bash
python -m ipykernel install --user --name=<any_name_for_kernel>
```

##### 4. Deactivate when done
```bash
deactivate
```

---

#### Option B: Using conda

##### 1. Create the environment
```bash
conda env create -f environment.yml
```

##### 2. Activate the environment
```bash
conda activate poster-metadata-quality-code
```

##### 3. Set up Jupyter kernel (for notebooks only)
```bash
python -m ipykernel install --user --name=<any_name_for_kernel>
```

##### 4. Deactivate when done
```bash
conda deactivate
```

---

### Launch JupyterLab

Launch JupyterLab and navigate to open the main.ipynb file. Make sure to change the kernel to the one created above (e.g., see [here](https://doc.cocalc.com/howto/jupyter-kernel-selection.html#cocalc-s-jupyter-notebook)). We recommend to use the [JupyterLab code formatter](https://github.com/jupyterlab-contrib/jupyterlab_code_formatter) along with the [Black](https://github.com/psf/black) and [isort](https://github.com/PyCQA/isort) formatters to facilitate compliance with PEP8 if you are editing the notebook.

## Inputs/outputs
The Jupyter notebook makes use of files in the dataset associated with the paper. You will need to download the dataset at add it in the input folder (call the dataset folder 'dataset').

Outputs of the code include plots displayed in the notebook but also saved as files. These saved plot files are included in the [output](https://github.com/fairdataihub/poster-metadata-quality-code/tree/main/outputs) folder.

## License
This work is licensed under
[MIT](https://opensource.org/licenses/mit) License. See [LICENSE](https://github.com/fairdataihub/poster-metadata-quality-code/blob/main/LICENSE) for more information.

## Feedback and contribution
Use the [GitHub issues](https://github.com/fairdataihub/poster-metadata-quality-code/issues) for submitting feedback or making suggestions. You can also work the repository and submit a pull request with suggestions.

## How to cite
If you use this code, please follow the citation instructions from the [CITATION.cff](CITATION.cff) file.
