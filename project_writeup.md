
# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./figures/center_1.jpg "Center Camera"
[image2]: ./figures/left_1.jpg "Left Camera"
[image3]: ./figures/recovery1.jpg "Recovery Image"
[image4]: ./figures/recovery2.jpg "Recovery Image"
[image5]: ./figures/recovery3.jpg "Recovery Image"
[image6]: ./figures/left_normal.jpg "Normal Image"
[image7]: ./figures/left_reverse.jpg "Flipped Image"
[image8]: ./figures/right_1.jpg "Right Camera"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* project_writeup.md (this report) summarizing the results
* run1.mp4 contains video of navigating track 1
* run2.mp4 contains video of navigating track 2

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 100-115) 

The model includes RELU layers to introduce nonlinearity (code line 103-110), and the data is normalized in the model using a Keras lambda layer (code line 101). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 104, 106, 108). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 90-91). The model was tested with collected data from running through the simulator in a structured manner.  This assisted in ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 118).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving the reverse of the track, and using multiple tracks.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to base the original structure off of the architecture shared by the NVIDIA team, and adjust accordingly.

My first step was to use a convolution neural network model similar to the NVIDIA architecture.  I thought this model might be appropriate because it was previously used to train their autonomous vehicle's model.  

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model with Dropout layers between the early convolution layers as to create some generalizations that the model should develop.  

Then I provided additional testing data that can possibly not be relevant to the tested track, but should still benefit the end-result.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, and to improve the driving behavior in these cases I established additional testing data and scenarios.  Additionally I included the left and right cameras with a mildly adjusted steering angle to expand the giving training set.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 100-115) consisted of a convolution neural network with the following layers and layer sizes;

Lambda - Normalizing with input shape 160x320x3
Cropping Layer - reduce 70px from top, 25 px from bottom
Convolution2D - 5x5 with 24 depth, RELU activation
Dropout @ 0.5
Convolution2D - 5x5 with 36 depth, RELU activation
Dropout @ 0.5
Convolution2D - 5x5 with 48 depth, RELU activation
Dropout @ 0.5
Convolution2D - 3x3 with 64 depth, RELU activation
Convolution2D - 3x3 with 64 depth, RELU activation
Flatten
Fully Connected - Size 100
Fully Connected - Size 50
Fully Connected - Size 10
Fully Connected - Size 1


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

And Left and Right cameras;
![alt text][image2]
![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to the behaviour of removing itself from the off-road, and onto the road. These images show what a recovery looks like starting from off-road, and leading to on-road :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Additionally I drove around the track in the reverse direction, ensuring natural driving behavior without the original turn bias (And performed recovery from left and right sides on reverse track driving).  

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would also assist with removing left/right turn bias. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had X number of data points. I then preprocessed this data by normalizing, flipping, and using the proper colour channel that is required; as opencv originally reads in BGR format.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by the training loss and validation losses beginning to diverge from beyond this point. I used an adam optimizer so that manually training the learning rate wasn't necessary.
