# Capstone 2026 - Breast Lesion MRI Analysis Pipeline and Interface

The goal of our capstone project was to create a lightweight preliminary tool to preprocess, segment and classify breast MRI data, and generate a comphrensive report detailing lesion statistics. Our target demographic was unique as our client wanted our solution to be very lightweight, not utilise cloud computing, and be able to be run on a standard laptop without an external graphics card. 

It is accompanied by a simple UI as well as CLI commands, detailed in ```/pipeline.v2/README.md```

Our main pipeline is build on Python and is self-hosting, and our frontend utilises React and FastAPI.

Please note that installing nnUnet for segmentation can differ heavily per system, please go to <nnUnetv2 GITHUB>

# Training

Training was done on UBC's SOCKEYE cluster, over the ODELIA Challenge Dataset, FASTMRI Dataset, and AMBL.

# Model Testing and selection

## Classification
Our pipeline considered Resnet50, Efficientnet, and Densenet models for our classifiers, but found that EfficientNet had superior results for lesion detection and for malignancy detection. As such, only EfficientNet is used for classification, with two sets of weights for each classification step.

## Segmentation
nnUnet was our main contender for our segmentation model as it was highly praised from our client as well as had historically strong results. We found that the 3DFullRes configuration was required for our algorithm as our segmentation performed poorly with 2D configurations.




