# DFPC Pruned-Models
PyTorch Code for loading and analysing the DFPC pruned models of ResNet-50 trained on the imagenet dataset.

## Step 1: Set up environment
- OS: Linux (Tested on Ubuntu 20.04. It should be all right for most linux platforms. Did not test on Windows and MacOS.)
- python=3.9.7 (conda is *strongly* suggested to manage environment)
- All the dependant libraries are summarized in `requirements.txt`.
- We use CUDA 10.2
```
conda create -n dfpc python=3.9.7
conda activate dfpc
conda install pip
pip install -r requirements.txt
```

## Step 2: Set up dataset
- We use the ImageNet dataset.
- For ImageNet,
    - Download the dataset. [Link to academic torrent](https://academictorrents.com/details/943977d8c96892d24237638335e481f3ccd54cfb)
    - Extract the dataset using `tar -xvzf ILSVRC2017_CLS-LOC.tar.gz`.
    - Prepare the validation set
        - Copy [`valprep.sh`](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) to the `val` folder of the dataset. Location: `<path-to-extracted-dataset>/ILSVRC/Data/CLS-LOC/val/`
        - `cd` into the `/val` folder and execute `bash valprep.sh`
        - Remove the valprep.sh file once execution finishes.

## Step 3: Download pruned models
- Download pruned models from [google drive](https://drive.google.com/drive/folders/1xeb9kAP28MqIG8cK4Cm_NsFVLilz61PO?usp=sharing) and put them in a directory named `checkpoints` within the current directory.

## Step 4: Loading pruned models
To evaluate the pruned models, use the following command
```
python load_models.py -a <model_name> <dataset_path>
```
Here, we can have the following options for arguments under the angled brackets.
- `<model_name>` can be replaced with `dfpc30` or `dfpc54`.
- `<dataset_path>` is the path of the ImageNet dataset until the `CLS-LOC` folder. For example, `/home/tanayn/ImageNet/ILSVRC/Data/CLS-LOC`
