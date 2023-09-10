# Mitosis Detection
## DenseNet

We use densenet architecture to compute the plane of focus of cells and its action class. Cell mitosis, apoptosis can be considered as action events with -T 
time frames before the occurance of the event and +T timeframe after the occurance of the event. We use a fully convolutional network to train our models and this
ensures that the model can be trained on image volume patches but the trained model can be applied to time-lapse volumes of arbitary size. The prediction function maps 
location of the action event and also predicts height, width and depth of the detected cell in the central time frame.

![image](images/mitosis_detection.gif)

## Training Data Details

To create the training data for Xenopus we used [script](examples/annotate_oneat.py) You will need to point to a directory containing raw timelapse images that will open in a custom Napari widget showing the classes to be annotated as point layers of different colors. Using the Napari points layer you can add/remove points at the location of the chosen events. Save points button will be dosplayed on the lower right corner and has to be clicked to save the csv file for each chosen tif file. On the lower left/middle of the widget is a drop down menu showing the raw image present in the directory that has to be chosen for the proces sof annotation. Once the annotation has been completed csv files bearing the filename, event name and an added name of 'ONEAT' appears in the directory. After this step please create segmentation images for the raw timelapse images using your custom VollSeg mode. 

Once these steps are done you are ready to create training patches and labels using this [script](examples/create_oneat_patches.py). In this script you can specify a certain size (Z,Y,X) which for our example use case was (8,64,64). This script uses the event locations marked in the previous step and creates a voxel of volume around it to create a training patch. It also uses the segmentation image to obtain the height, width and depth of the cell that has been clicked and in our patch making process the cell clicked is always at the center of the frame in the TZYX generated patch. Users can specify the timeframe before and after the event location, we used 1 time frame before and one time frame after the event (mitosis) to make the training patch, hence the size of the generated patch in temporal dimension is 3. The output of this script is a directory containing training patches as .tif file, their corresponding .csv file labels and .npz files to be used in the training of the oneat model.

Once the npz file has been created we use the generated npz file in this [script](examples/train_oneat.py) to train a volumetric action classification network of oneat using Densenet as underlying network. For the Xenopus dataset these are the training dataset details:

| Volume shape (TZYX)   |      Number of patches      |  Size in MB |
|----------|:-------------:|------:|
| (3,8,64,64) |  20446 Normal events | 8200 MB |
| (3,8,64,64) |  4336 Division events | 1700 MB |
