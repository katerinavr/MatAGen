```Repository with the code of MMatAGen for multimodal data mining from materials science literature```

## Installation
```
# Clone the repository
!git clone https://github.com/katerinavr/MatAGen.git
%cd MatAGen

# Create a conda environment
conda create --name mmatagen python=3.11 -y
conda activate mmatagen
pip install -r requirements.txt
pip install -e .
```

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
Detailed instructions on how to use the pipeline for various materials data mining use cases are provided in the 
following links notebooks.

| Notebook Name        | Description                          | Link                                  |
|----------------------|--------------------------------------|---------------------------------------|
| workshop_tutorial   | data mining + RAG tutorial          | [Open](notebooks/tutorial_1_data_mining.ipynb) |
| literature_data_mining    | Multimodal data mining        | [Open](notebooks/tutorial_2_image_mining&tagging.ipynb) |
| plot_digitalization      | Digitalizing absorption spectra    | [Open](notebooks/tutorial_3_image_segmentation.ipynb)   |

