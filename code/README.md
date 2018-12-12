## Modified from https://github.com/manzilzaheer/DeepSets

## Requirements
* Python 3
* PyTorch
* tqdm
* h5py
* NumPy

## Data generate

This needs preprocessed ModelNet40 dataset (http://modelnet.cs.princeton.edu/). From the CAD models, 10,000 points have to be uniformly sampled to produce the point cloud. For both train and test set, store the point cloud generated as a (N, M, 3) dimensional numpy array and labels as (N,) numpy array where N is number of objects, M=10,000 points per object, and each point is in 3D space. We save the resulting arrays in a hdf5 file `ModelNet40_cloud.h5` with keys `tr_cloud`, `tr_label`, `test_cloud`, and `test_label` and the current code assumes availability of this file. Please contact us if you are having difficulty in generating this file.

Since the data is too big to uploaded to github, so we do not provide our data here. Email me: shiyue@cs.unc.edu if you want the formated data to run the code successfully.

## Models

Network in `classifier.py`

Change `#Setup network` part in `run.py` to change the model

## Baseline Experiments

Once you have the data file prepared, you may simply run

    python3 run.py

to replicate the results. There are following parameters which can be set in `run.py`:

```python
# Number of epochs to run
num_epochs = 1000  

# Batch size for training
batch_size = 64

# Number of points the classifier will use
# Since we have in total 10,000 points, if we 
# want to use 1,000 points we have to downsample 
# by 10. Similarly if we want to use 100 points, 
# the downsample parameter will be 100.
downsample = 10

# Size of the permutation equivariant layers
# For 5000 points use 512,
# For 1000 points use 256
# For 100 points use 256
network_dim = 256  

#Number of times to repeat the experiment so as to get a variance estimate
num_repeats = 5    
```
