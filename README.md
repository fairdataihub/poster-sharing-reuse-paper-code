# Trends, FAIR Practices, and Opportunities in Scientific Poster Sharing Paper

## About
Scientific posters are the most common forms of scholarly communication, with millions presented at conferences each year. They contain early-stage insights that, if shared beyond the conference, could accelerate scientific discovery. In this study, digital sharing of posters and alignment with FAIR principles were investigated. See this [inventory](https://github.com/fairdataihub/poster-sharing-reuse-paper-inventory) for all related resources, including the paper.

## Standards followed
The overall code is structured according to the [FAIR-BioRS guidelines](https://fair-biors.org/). The Python code in the Jupyter notebook [main.ipynb](main.ipynb) follows the [PEP8 guidelines](https://peps.python.org/pep-0008). Functions are documented with docstring formatted following [Google's style guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings). All the dependencies are documented in the [environment.yml](environment.yml) file.

## Using the Jupyter notebook

### Prerequisites

We recommend using Anaconda to create and manage your development environment and using JupyterLab to run the notebook. All the subsequent instructions are provided assuming you are using [Anaconda (Python 3 version)](https://www.anaconda.com/products/individual) and JupyterLab.

### Clone the repository

Clone the repo or download as a zip and extract it.

### Navigate to the code directory

Open Anaconda prompt (Windows) or the system Command line interface then naviguate to the code

```bash
cd .poster-metadata-quality-code 
```

### Setup conda env

```bash
$ conda env create -f environment.yml
```

### Set up Jupyter kernel

```sh
$ conda activate poster-metadata-quality-code
$ conda install ipykernel
$ ipython kernel install --user --name=<any_name_for_kernel>
$ conda deactivate
```

### Launch Jupyter lab
Launch Jupyter lab and naviguate to open the main.ipynb file. Make sure to change the kernel to the one created above (e.g., see [here](https://doc.cocalc.com/howto/jupyter-kernel-selection.html#cocalc-s-jupyter-notebook)). We recommend to use the [JupyterLab code formatter](https://github.com/jupyterlab-contrib/jupyterlab_code_formatter) along with the [Black](https://github.com/psf/black) and [isort](https://github.com/PyCQA/isort) formatters to facilitate compliance with PEP8 if you are editing the notebook.

## Inputs/outputs
The Jupyter notebook makes use of files in the dataset associated with the paper. You will need to download the dataset at add it in the input folder (call the dataset folder 'dataset').

Outputs of the code include plots displayed in the notebook but also saved as files. These saved plot files are included in the [output](https://github.com/fairdataihub/poster-metadata-quality-code/tree/main/outputs) folder.

## License
This work is licensed under
[MIT](https://opensource.org/licenses/mit) License. See [LICENSE](https://github.com/fairdataihub/poster-metadata-quality-code/blob/main/LICENSE) for more information.

## Feedback and contribution
Use the [GitHub issues](https://github.com/fairdataihub/poster-metadata-quality-code/issues) for submitting feedback or making suggestions. You can also work the repository and submit a pull request with suggestions.

## How to cite
If you use this code, please cite the related paper (it will be listed [here](https://github.com/fairdataihub/poster-sharing-reuse-paper-inventory) when available)..