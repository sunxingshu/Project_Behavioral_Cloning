# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[image1]: ./images/sharp_angle.png
[image2]: ./images/crop.png
[image3]: ./images/reduce_pixel2.png
[image4]: ./images/hist.png
[image5]: ./images/network_model.png



Objective
---
We will develop a deep neural network based autonomous driving system. The framework runs training and validation in a game simulator.

The model shall ensure that no tire may leave the drivable portion of the track surface. The car may not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe (if humans were in the vehicle).

Approach
---
We follow the below steps to establish the framework:
* Data gathering: Gather training data including images and steering angle using the game simulator 
* Data processing: Preprocess the image and balance data based on steering angle
* Network Building: Build a deep neural network based on NVIDIA architecture
* Training and Validation: Train and validate the data in Track#1

### Data gathering

#### Running Simulator
We divide our training into three categories:
* We drive clockwise along the track and keep the car in the center
* We drive counter clockwise along the track and keep the car in the center 
* We deliberately drive the car to the edge and then recenter the car

The three categories shall be sufficient to teach the neural network to 1) avoid driving car to the edge 2) self-center if the car detours from the center

#### Leverage left and right cameras
The car is equipped with three cameras: left, center, and right.  The center camera will serve to facilitate training car to drive along the center. Instead of directly using the left and right cameras to calculate steering angle, we treat them as the center camera while adding a correction steering so the car will self-center. We choose 0.2 in radiances as the correction factor. So in short, left_angle = steering + 0.2 and right_angle = steering - 0.2.

![alt text][image1]

### Data preprocessing:

#### Flip Image
Though we have collected the data by driving the track both clockwise and counterclockwise, we still flip the image from left to right to increase our sample at the turns. Accordingly, we will also flip the sign of the angle of each image. 

#### Crop the image
Not all the pixels in the image captured are useful to calculate steering angle; however, they do take up memory space. Thus, we crop the image by from top by 70x pixels and from bottom by 20x pixels as shown below.

![alt text][image2]

#### Reduce # of pixels
The resolution of the image can be comprised while retaining the key feature for neural network to extract. We reduce the # of pixel by four time while keeping the same aspect ratio such that we can reduce the required memory. One can compare the below image versus the above one and it is clear that the key feature (lane) is still noticeable even with four times fewer pixels. This step is directly implemented as a step in our deep neural network through Keras.

![alt text][image3]


#### Balance steering angle
From the below image, we can tell that we have certain bins of steering angles that have much more samples than the rest. This imbalance may bias our learning towards these bins.

We identify the top 3 bins with the highest sample size and crop the samples to 60. The new distribution is also shown below.

![alt text][image4]

#### Normalization
We also normalize the color channel by applying a Keras lambda layer.

### Network Building

For the deep neural network, we adopt the NIVIDA mode which consists of a convolution neural network with 4 conv layers which have 3x3 filter sizes and depths vary between 32 and 256, and 3 fully connected layers. The model includes RELU layers to introduce nonlinearity.

We also implement a dropout layer with 0.5 dropout probability to prevent over-fitting.

![alt text][image5]

### Training and Validation: 

#### Split into training and validation
We first shuffle the data and split them into 70% training data and 30% validation data. In each epoch, we train the network with the training data and then calculate validation loss using the validation data.

#### Training Parameters
We use an adam optimizer with a preset learning rate = 0.001. The batch size is set to be 100 to make the network memory friendly and the number of epoch for learning is 5.



atch_size = 100
learning_rate = 0.001
nb_epoch = 5