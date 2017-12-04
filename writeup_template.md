#**Seld Driving Car Nanodegree Project 3: Behavioral Cloning** 



---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
Please note that the model was constructed using Keras 2, I tested this model on a machine with Keras 2 installed and works fine, however Keras 1.2 may fail to load the file.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the Nvidia drive network architecture described in the course material. The network consists of a cropping and lambda layer, 5 convolutional, 3 densely connected layers, and one output layer. The model is initialized inside the 'initialize_model()' function of the 'model.py' file.

The cropping layer crops the top and bottom parts of the image that do not contain information about steering, and the lambda later normalizes the images so that the data is zero mean and unit variance.

The convolutional layers sizes are 16 * 3x3, 24 * 5x5, 36 * 5x5, 64 * 3x3, and 64 * 3x3 in that order. Between convolutional layers there are 2x2 max-pooling layers to reduce the scale and 'relu' activations to introduce non-linearity.

The final convolutional layer is flattened and connected to the incomming densely connected layer. The sizes of the densely connected layers are 100, 50, and 50 respectively. Similar to the convolutional layers, the densely connected layers have 'relu' activations between them. 


#### 2. Attempts to reduce overfitting in the model

To prevent overfitting I added dropout between all of the layers. The model was trained and validated on different data sets to ensure that the model was not overfitting. I separated 30% of the data and used this as validation set. 

I first choose the keep probability a moderate value (0.5). With this value, training accuracy was improving although validation accuracy was slightly decreasing, the network was overfitting the data. I tuned the probability, and around 0.3, validation accuracy was changing together with training accuracy.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer (model.py line 179), so I did not manually tune the parameters during the training. But I have seen the accuracies swinging even from the first few epocsh so I decreased the initial learning rate of the optimizer (0.00005).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and I added some noisy driving to improve the network performance.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was similar to the course material. I implemented a network similar to the Nvidia network described in the course with 5 convolutional layers and 4 densely connected layers. I thought this network would fit my needs because a similar network was used for a similar estimation task from camera images.

The first step of the implementation is to crop the useful parts of the image. This will both reduce the training time and increase the performance, since the irrelevant parsts would not pass through the network. I used Cropping2D layer of the Keras library for this function.

The second step is to normalize data in order to have a zero mean and unit variance input. The network will be more stable and train faster with normalization. I did this with a Lambda layer and a lambda function that divides the data by 255, and subtracts 0.5 from it.

The next step is to use the convolutional layers to learn the necessary features for understanding angles. I have used 5 convolutional layers with incresing widths (widths of 16,24,36,64,64 from imput, to the last convolutional), and used max-pooling to have the ability to higher scale features. I also added 'relu' activation in order to introduce non-linearity.

After the convolutional layers I flattened the last layer,and connected it to the densely connected layers, There are four densely connected layers with decreasing widths (100, 50, 50, 1 from after flattened layer to output layer). Again, I used 'relu' activation to introduce non-linearities to these layer.

To prevent overfitting I used dropout layers between the previous layers. I began with a moderate keep probability (0.5). I wanted to observe overfitting, and reduce it until the the training and validation accuracies moved together. 

I did not manually change the parameters during the training, instead I used ADAM optimizer. I also noticed that the accuracy would swing too much during the training. I first assumed that this was due to overfitting and early stop the training process. However, the network wasn't getting better, Later, as I played with parameters, I noticed that the default learning rate of ADAM optimizer was large, and I reduced the learning rate.

I employed a large amount of data for training, instead of loading all of the data at once I employed generator functions to load the data when the batch is training using the 'generate_data()' function defined in line 53 in model.py.

When I optimized all the model parameters, the car wasn't still able to drive complete circuit. I understood that I needed to increase the data. I detailed the collection of data in the third part of this section. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 124-180) consisted of a convolution neural network with the following layers and layer sizes 
| Layer  | Description   | Shape          | 
|:------:|:-------------:| :-------------:|
| 1      | Cropping      |    -           |
| 2      | Lambda        |  (x/255)-0.5   |
|:------:|:-------------:| :-------------:|
| 3      | Conv2D        |  16 x 5 by 5   |
| 4      | Dropout       |  0.3           |
| 5      | MaxPooling2   |  2 by 2        |
| 6      | Activation    |  relu          |
|:------:|:-------------:| :-------------:|
| 7      | Conv2D        |  24 x 5 by 5   |
| 8      | Dropout       |  0.3           |
| 9      | MaxPooling2   |  2 by 2        |
| 10     | Activation    |  relu          |
|:------:|:-------------:| :-------------:|
| 11     | Conv2D        |  36 x 5 by 5   |
| 12     | Dropout       |  0.3           |
| 13     | MaxPooling2   |  2 by 2        |
| 14     | Activation    |  relu          |
|:------:|:-------------:| :-------------:|
| 15     | Conv2D        |  64 x 3 by 3   |
| 16     | Dropout       |  0.3           |
| 17     | MaxPooling2   |  2 by 2        |
| 18     | Activation    |  relu          |
|:------:|:-------------:| :-------------:|
| 19     | Conv2D        |  64 x 3 by 3   |
| 20     | Dropout       |  0.3           |
| 21     | MaxPooling2   |  2 by 2        |
| 22     | Activation    |  relu          |
| 23     | Flatten       |  -             |
|:------:|:-------------:| :-------------:|
| 24     | Dense         |  100           |
| 25     | Dropout       |  0.3           |
| 26     | Activation    |  relu          |
|:------:|:-------------:| :-------------:|
| 27     | Dense         |  50            |
| 28     | Dropout       |  0.3           |
| 29     | Activation    |  relu          |
|:------:|:-------------:| :-------------:|
| 30     | Dense         |  50            |
| 31     | Activation    |  relu          |
|:------:|:-------------:| :-------------:|
| 32     | Dense         |  1             |


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
