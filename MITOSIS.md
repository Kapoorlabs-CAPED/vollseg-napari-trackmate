# Mitosis Detection
## DenseNet

We use densenet architecture to compute the plane of focus of cells and its action class. Cell mitosis, apoptosis can be considered as action events with -T 
time frames before the occurance of the event and +T timeframe after the occurance of the event. We use a fully convolutional network to train our models and this
ensures that the model can be trained on image volume patches but the trained model can be applied to time-lapse volumes of arbitary size. The prediction function maps 
location of the action event and also predicts height, width and depth of the detected cell in the central time frame.

![image](images/mitosis_detection.gif)
