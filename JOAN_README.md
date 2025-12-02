# Info on using this model

This model now works (i think).

To run on your own device:

### 1. Data Format
The files should be putting as the following structure.
```
files
└── <dataset>
    ├── all_images
    |   ├── dataset_001_extra.png
    │   ├── dataset_001_extra.png
    │   ├── dataset_001_extra.png
    │   ├── ...
    |
    └── all_masks
        ├── dataset_001.png
        ├── dataset_002.png
        ├── dataset_003.png
        ├── ...        
        
```

Dataset: Name of where you got the data from ex: ISPY1, Duke. The extra is not explicitly necessary.. I think


### 2. Lines to change
```
{
  output_dir = "/Users/joannwokeforo/Documents/BMEG457/Data/ISPY1-TestFormat"
}
``` 

Change this to match the path of your desired output director.

### 3. Packages you may need to install

You may need to install torch and tensorboard. Your IDE should flag the ones you need

### 4. Using Tensorboard

Tensor board lets you check how the metrics (DCE, accuracy, sensitivity, specificity, loss) change as testing continues. 

To see this in your terminal run:
```
{
  tensorboard --logdir=runs
}
``` 

**Note:** Tensor board must be installed for this to work, and should be running

### 5. Optional changes

You can change the number of epochs by changing the default val in this line:
```
{
  parser.add_argument('--epochs', default=60, type=int,
                    help='number of total epochs to run')
}
``` 

5-10 is good for init testing. 


