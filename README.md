# Trends, FAIR Practices, and Opportunities in Scientific Poster Sharing Paper

## About
Scientific posters are the most common forms of scholarly communication, with millions presented at conferences each year. They contain early-stage insights that, if shared beyond the conference, could accelerate scientific discovery. In this study, digital sharing of posters and alignment with FAIR principles were investigated.

## Standards followed
The overall code is structured according to the FAIR-BioRS guidelines. The Python code in the Jupyter notebook main.ipynb follows the PEP8 guidelines. Functions are documented with docstring formatted following Google's style guide. All the dependencies are documented in the environment.yml file.

## Using the Jupyter notebook

### Prerequisites

### Clone repo
Clone the repo or download as a zip and extract.

### cd into the code folder
Open Anaconda prompt (Windows) or the system Command line interface then naviguate to the code

`cd .poster-metadata-quality-code `

### Setup conda env

`$ conda env create -f environment.yml`

### Setup kernell for Jupyter lab
```
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
Citation information will be added upon manuscript submission or publication.
