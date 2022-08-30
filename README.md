MLBD Project - Team 🦖
==============================

Machine Learning for behavioral data - Project repository

All the code relevant to this project can be found inside the `notebooks` directory. There, all notebooks and additional helper python code needed to reproduce our results are present.

We recommend running the notebooks using Google Colab due to dataset sizes and execution speed. To mount a personal Drive to load data, run:
```
from google.colab import drive
drive.mount('/content/drive')
```
and accordingly set a `DATA_DIR` variable to specify the data directory. To import code from a python file `filename.py`, run:
```
from google.colab import files
src = list(files.upload().values())[0]
open('filename.py','wb').write(src)
import filename
```

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── weights        <- Trained and serialized models, model predictions, or model summaries
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks.
    │   └── modules        <- Python code for use in the project
    │       ├── __init__.py    <- Makes modules a Python module
    │       ├── archive        <- Old notebooks and scripts from previous milestones
    │       │
    │       ├── preparation.py <- Scripts to download or generate data
    │       ├── sesh.py        <- Helper scripts for session happiness predictions
    │       └── models        <- Scripts to train models and then use trained models to make predictions
    │
    ├── report_m4.pdf      <- Milestone 4 report
    │
    ├── poster.pptx        <- Presentation poster
    │
    └── final-report.pdf   <- Final report


--------