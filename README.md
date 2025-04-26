```Repository with the code of MMatAGen for multimodal data mining from materials science literature```

## Installation
conda create --name mmatagen python=3.11 -y
conda activate mmatagen
pip install -r requirements.txt
pip install -e .

## Add your LLM credentials
Create a file: src/matagen/config/settings.py
and add your keys:
```
OPENAI_API_KEY = ""
anthropic_api_key = ""
```

## Load pretrained weights for the Computer Vision models
By running the following scripts a checkpoints folder will be created in the main directory with the pretrained weights for the vision model.
```
python src/matagen/utils/load_pretrained_yolo.py
```
If you want to perform absorption spectra digitalization you also need to also download the following weights.
```
python src/matagen/utils/load_pretrained_plot2spectra.py
```

## Notebooks

| Notebook Name        | Description                          | Link                                  |
|----------------------|--------------------------------------|---------------------------------------|
| workshop_tutorial   | data mining + RAG tutorial          | [Open](notebooks/workshop_tutorial.ipynb) |
| literature_data_mining    | Multimodal data mining        | [Open](notebooks/literature_data_mining.ipynb) |
| plot_digitalization      | Digitalizing absorption spectra    | [Open](notebooks/model_training.ipynb)   |

