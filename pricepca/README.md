# PCA


Price data -> PCA for dim reduction -> Kmeans clustering (only 1 cluster atm, so it's just the mean) -> plot

python3 pca.py --data 206423.csv -p


## Results

> What the heck am I looking at?

The graphs shows are the 32 dimensional price data sets projected onto the 1st, 3rd, and 4th eigen vectors (feel free to play with which eigen vectors to project onto). This will lose information, but should keep as much variation information as possible. The axises are therefore a bit difficult to comprehend. First all price data is normalized to remove scale. This means that each axis is a combination of normalized prices multiplied with some coeffecient. If you think of each opr as a matrix of 1x32, each eigen vector is a matrix of 32x1. By using 3 eigen vectors, the result is 3 dimensions, and if you print the eigen vectors, you can see which prices most affect the value for the axis. Jitter is added to all the datapoints to prevent many data points from appearing as a single pt.

**Legend**
- Blue dots == top 50
- Brown dots == OPRs outside top 50
- Red dot == mean of the dataset

**Data**
- [206423](./206423_pca.png) 
- [207002](./207002_pca.png) 
- [207291](./207291_pca.png) -- [Data](https://docs.google.com/spreadsheets/d/1plXV2wQaGM3gO338BWfi6AZWPKesYE1VgtpAPtNxI9Y/edit#gid=2087312555) is all pretty much the same. So jitter is what you are seeing.
- [207296](./207296_pca.png) 