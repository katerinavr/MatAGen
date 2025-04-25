```Repository with the code of MMatAGen for multimodal data mining from materials science literature```

## Installation
conda create --name mmatagen python=3.11 -y
conda activate mmatagen
pip install -r requirements.txt
pip install -e .


You can your project and its dependencies using commands like:
```
pip install . (Install core package)
pip install .[ui] (Install core + UI dependencies)
pip install -e .[dev,ui] (Install in editable mode with dev and UI dependencies - recommended for development)
```

## Run the app
```
panel serve app.py --autoreload --show
```

## Project Structure
```text
.
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
├── data              # Example input data
│   ├── absorption_spectra_plots
│   ├── html_folder
│   ├── pdf_examples
│   ├── test_files
│   └── xml_examples
├── notebooks         # Demo notebooks
│   └── ...
├── pyproject.toml
├── requirements.txt
├── src
│   └── matagen       # Core library package
│       ├── __init__.py
│       ├── agents        # AutoGen agent definitions
│       │   └── ...
│       ├── analysis      # Data analysis/extraction tools
│       │   └── ...
│       ├── config        # Configuration loading
│       │   └── ...
│       ├── custom_tools_drop # Contains modified external tools? Review placement.
│       │   ├── __init__.py
│       │   └── external_tools
│       ├── scraping      # Data scraping tools
│       │   ├── __init__.py
│       │   └── html_scraper.py # ...
│       └── utils         # Generic utility functions
│           └── __init__.py
├── tests             # Tests for matagen
│   └── ...
└── ui                # Panel web application
    ├── __init__.py
    └── app.py
```

**Key Directories:**

* `src/matagen/`: Core Python library code for scraping, analysis, and agent logic.
* `ui/`: Contains the Panel web application code (`app.py`).
* `data/`: Example input data files (HTML, PDF).
* `notebooks/`: Jupyter notebooks for demonstrations and experiments.
* `outputs/`: Default location for generated files (logs, JSON results, images). (Ignored in the tree above by default).
* `tests/`: Unit and integration tests for the `matagen` library.
* `requirements.txt`: Project dependencies.
* `.env.example`: Template for required environment variables (API keys).
* `matagen/agents`: for defining the agents
* `matagen/analysis`: for data analysis tools
* `matagen/scraping`: for scraping tools
 matagen/utils: for generic helper functions



## Notebooks

| Notebook Name        | Description                          | Link                                  |
|----------------------|--------------------------------------|---------------------------------------|
| workshop_tutorial   | data mining + RAG tutorial          | [Open](notebooks/workshop_tutorial.ipynb) |
| literature_data_mining    | Multimodal data mining        | [Open](notebooks/literature_data_mining.ipynb) |
| plot_digitalization      | Digitalizing absorption spectra    | [Open](notebooks/model_training.ipynb)   |

