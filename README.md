# TokenViz

## Contents of this repository
The repository includes the code used for our experiments, including implementations of the three tasks described in the paper.

.license.txt: The license of our code
├── clone_detection/code_classification/defect_detection: Implementation of this task
│  └── dataset: Dataset folder for experiment
│  └── nets: Contains the implementation of nerual network for this task
└── shared: Contains modules shared by the code implementation of each task
   └── utils: Practical modules for completing experiments
   └── visualize: Implementation of our code visualisation approach

## Running the code
### Prerequisites
Computer with a graphics card that support CUDA and Linux installation.



### Dataset
We saved the dataset on Google Drive for download:
[https://drive.google.com/drive/folders/18SPu6DwUv2158k3zxGsrsuhMgFuspw0Y](https://drive.google.com/drive/folders/18SPu6DwUv2158k3zxGsrsuhMgFuspw0Y)

For Clone Detection/Code Classification:
   Download the dataset files in the corresponding folder on Google Drive and place the dataset files in the "dataset" folder of the corresponding task.


For Defect Detection:
1. Download the "dataset.zip" file from Google Drive in the defect_detection folder.
2. Run the "mkcsvdataset.py" file in the "txt" folder to generate the dataset for WPDP/CPDP.
3. Place the generated dataset in the "dataset" folder.

### Running the experiment
Run the experiment.py file in the corresponding task folder

Note: The corresponding parameters can be adjusted by modifying experiment.py.






