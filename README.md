# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this project, I developed a CNN model in Keras to clone driving behavior. Based on the camera view, the model outputs a steering angle to autonomously drive the vehicle.

These are the main steps of the project:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around the track without leaving the road


[//]: # (Image References)

[image1]: ./Images/Train.png "Training"
[image2]: ./Images/NVIDIA.png "Model"




### Data Collection and Augmentation

The first step was to drive the car in the simulator in manual mode to record the driving behavior, as shown below.

![alt text][image1]

The vehicle is equipped with 3 cameras, installed on the left, center and right of the vehicle.
I then processed the collected data through normalization and augmentation.
I applied techniques such as flipping the images horizontally and reversing the steering angle to create a more balanced dataset.
I also incorporated the left and right camera views with adjusted steering angles.
Finally, I cropped the images to the portion of interest in order to improve the learning of the model.


### CNN Model

I implemented the End-to-End learning model from NVIDIA's paper.
Here is the model architecture:
![alt text][image2]

The network has about 27 million connections and 250 thousand parameters.

The convolutional layers are designed to perform feature extraction. It uses strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel, and a non-strided convolution with a 3×3 kernel size in the final two convolutional layers.

The five convolutional layers are followed by three fully connected layers, leading to a final output control value. 

### Final Result

The final model is able to successfully drive around the track without leaving the road.

Here's a [link to my video result](./final-model-results-recording.mp4)

