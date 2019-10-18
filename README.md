# ChestXRay
 Chest X-Ray Pneumonia Classifiers Using fastai

# Key Notebooks:
- Classification of Pneumonia/Normal 
  - File: 'RESNET101 Classify Pneumonia.ipynb' [GitHub](https://github.com/williamsdoug/ChestXRay/blob/master/RESNET101%20Classify%20Pneumonia.ipynb)  [nbviewer](https://nbviewer.jupyter.org/github/williamsdoug/ChestXRay/blob/master/RESNET101%20Classify%20Pneumonia.ipynb)
  - Results Summary:
    - 94.0% accuracy tuning head only
    - 98.2% accuracy after fine tuning
    - 99.2% accuracy, 3.3% unknown with using threshold=0.80

- Classification of Viral/Bacterial Pneumonia
  - File: 'RESNET50 Classify Viral or Bacterial.ipynb' [GitHub](https://github.com/williamsdoug/ChestXRay/blob/master/RESNET50%20Classify%20Viral%20or%20Bacterial.ipynb)  [nbviewer](https://nbviewer.jupyter.org/github/williamsdoug/ChestXRay/blob/master/RESNET50%20Classify%20Viral%20or%20Bacterial.ipynb)
  - Results Summary:
    - 76.6% accuracy tuning head only
    - 77.8% accuracy after fine tuning
    - 82.7% accuracy, 25.26% Unknown with threshold = 0.70
    - 89.7% accuracy, 50.45% Unknown with threshold = 0.85

# Kaggle Pneumonia Chest X-Ray Images Dataset
- [dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- [paper](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)


# Other Files of interest
- fastai_addons.py -- Various analysis functions for ClassificationInterpretation

# Credits
- The basic Pneumonia Classifer notebook was modeled after https://www.kaggle.com/natevegh/pneumonia-detection-98-acc-fastai-2019-update