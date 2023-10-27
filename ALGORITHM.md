# Plugin usage

## Tutorial
- A detailed tutorial can be found at [Demo](https://youtu.be/7Yjd-Z3zJtk?si=_AksSBUJuEXbvIFM)

## Plugin Steps
- Open a 3D + time raw image and optionally its segmentation image, mask image in the viewer or select the provided sample data from the Napari menu.
- Launch the plugin and select the raw and segmentation image, provide the paths for xml and csv files.
- If you want to create a master xml file, select a pre-trained autoencoder model or load a .ckpt file if you have trained one using our lightning library.
- Click the compute tracks button.
- Wait for computation to be over, the progress of different steps is displayed in the progress bar.
- Once done, you can view the histogram, mean-variance, phenotype plots, and a track table.
- You can interactively select the track analysis type to be dividing or non-dividing, which will update the plots too.
- You can select a TrackMate track id to visualize the tracks or you can select to visualize all tracks.
- Tracks can be colored using different track properties or if the segmentation image is also input into the plugin, you can color that image with chosen track and spot attributes.

![plugin design](images/plugin_look.jpg)

## Segmentation and Tracking
VollSeg has multiple model combinations for obtaining segmentations; some of its combinations are more suited for certain image and microscopy types, the details of which can be found in the [VollSeg repository](https://github.com/Kapoorlabs-CAPED/VollSeg). The output of segmentation is instance labels for 2D, 3D, 2D + time, and 3D + time datasets. The aim of obtaining these labels is accurate quantification of cell shapes, as that is usually a biological aim for researchers interested in cell fate quantification. This tool was made as a [Napari grant action](https://chanzuckerberg.com/science/programs-resources/imaging/napari/vollseg-extensions-and-workflow-development-with-user-support/) to achieve that purpose in cell tracking.

In our algorithm, we train an autoencoder model to obtain a point cloud representation of the segmentation labels (figure below).

In our algorithm, the segmentation labels coming from VollSeg along with the raw image of the cells and the tissue boundary mask of those cells serve as an input to this plugin. The tracking is performed prior to using this plugin in Fiji using TrackMate, which is the most popular tracking solution with track editing tools in Fiji. The output of that plugin is XML and [tracks, spots, and edges CSV files] that serve as an input to this plugin. We also provide pre-trained autoencoder models for nuclei and membrane that can be used to obtain the point cloud representation of the segmented cells. Users can also provide their autoencoder models if they have trained them on their data.
![comparison](images/point_clouds_compared.png)

## Point Clouds

As a first step, users apply the trained autoencoder models to the input timelapse of segmentation, and in the plugin, point cloud representation of all the cells in the tracks is computed. We provide a [script](examples/visualize_point_clouds.py) to visualize the point cloud representation for the input segmentation image (binary) using classical and autoencoder model predictions.

## Autoencoder

This is an algorithm developed by [Sentinal](https://www.sentinal4d.com/) AI startup of the UK, and they created a [PyTorch](https://github.com/Sentinal4D) based program to train autoencoder models that generate point cloud representations. KapoorLabs created a [Lightning version](https://github.com/Kapoorlabs-CAPED/KapoorLabs-Lightning) of their software that allows for multi-GPU training. In this plugin, the autoencoder model is used to convert the instances to point clouds. Users can select our pre-trained models or choose their own before applying the model. The computation is then performed on their GPU (recommended) before further analysis is carried out. As this is an expensive computation, we also provide a [script](examples/apply_autoencoder.py) to do the same that can be submitted to the HPC to obtain a master XML file that appends additional shape and dynamic features to the cell feature vectors, thereby enhancing the basic XML that comes out of TrackMate.

## Auto Track Correction

We use a mitosis and apoptosis detection network to find the locations of cells in mitosis, which is then used as prior information to solve a local Jaqman linker for linking mitotic cell trajectories and terminating the apoptotic cell trajectories. To read more about this approach, please read more about [oneat](MITOSIS.md).

## Shape Features
The shape features computed in the plugin use the point cloud representations produced by the autoencoder model. We compute the following shape features:

- Eccentricity
- Eigenvectors and Eigenvalues of the covariance matrix
- Surface area and Volume

## Dynamic Features
The dynamic features computed in the plugin are the following:

- Radial Angle: Angle between the center of the tissue and the cell centroid, taking the origin as the top left coordinate of the image. [demonstration](RADIAL_ANGLE.md)
- Motion Angle: Angle between the center of the tissue and the difference between the cell locations in successive coordinates.
- Cell Axis Angle: Angle between the center of the tissue and the largest eigenvector of the cell.
- Speed: Cell speed at a given time instance.
- Acceleration: Cell acceleration at a given time instance.
- Distance of cell-mask: Distance between the cell and the tissue mask at a given time instance.
