#**Behavioral Cloning**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[architecture]: ./images/architecture.jpeg "Model Architecture"
[model]: ./images/model.png "Model Summary"
[anticlockwise_example]: ./images/center_2017_02_15_14_36_49_867.jpg "Anticlockwise Example"
[clockwise_example]: ./images/center_2017_02_15_14_24_44_215.jpg "Clockwise Example"
[RightSideToCenter1]: ./images/center_2017_02_15_14_47_24_146.jpg "Right Side to Center"
[RightSideToCenter2]: ./images/center_2017_02_15_14_48_45_215.jpg "Right Side to Center"
[RightSideToCenter3]: ./images/center_2017_02_15_14_49_12_595.jpg "Right Side to Center"
[RightSideToCenter4]: ./images/center_2017_02_17_13_44_36_937.jpg "Right Side to Center"
[LeftSideToCenter1]: ./images/center_2017_02_15_15_03_15_825.jpg "Left Side to Center"
[LeftSideToCenter2]: ./images/center_2017_02_15_15_03_45_654.jpg "Left Side to Center"
[LeftSideToCenter3]: ./images/center_2017_02_15_15_04_01_574.jpg "Left Side to Center"
[LeftSideToCenter4]: ./images/center_2017_02_17_13_29_57_542.jpg "Left Side to Center"
[image4]: ./images/placeholder_small.png "Recovery Image"
[image5]: ./images/placeholder_small.png "Recovery Image"
[image6]: ./images/placeholder_small.png "Normal Image"
[image7]: ./images/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of one cropping layer (code line 58), one normalizing layer (code line 60), one resizing layer (code line 62) and InceptionV3 model without output (code line 68), one flatten layer (code line 73), two dense layers and two dropout layers (code line 73-77). Here is the input and output shape of each layer.

| Layer           | Input Shape   | Output Shape  |
|:---------------:|:-------------:|:-------------:|
| input_1         | (160, 320, 3) | (160, 320, 3) |
| cropping2d_1    | (160, 320, 3) | (90, 320, 3)  |
| normalization   | (90, 320, 3)  | (90, 320, 3)  |
| resize          | (90, 320, 3)  | (224, 224, 3) |
| InceptionV3     | (224, 224, 3) | (5, 5, 2048)  |
| flatten_1       | (5, 5, 2048)  | (51200)       |
| dropout_1 (0.5) | (51200)       | (51200)       |
| dense_1         | (51600)       | (1024)        |
| dropout_2 (0.5) | (1024)        | (1024)        |
| dense_2         | (1024)        | (1)           |


During the training process, the weights of InceptionV3 was frozen so that it only need to train the weights of dense\_1 and dense\_2.

More details about the model architecture are shown in `./images/model.png`.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 74 & 76).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 36-54). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 80).
The model used the mean squared error as loss, because this project is a regression question rather than a classification question.
I also tune the `correction` parameter which is used to calibrate the steering angle of left or right images. Finally, I set `correction` parameter to 0.15.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, the clockwise and anticlockwise driving.

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to build a transfer learning model.

My first step was try to use ResNet50 to build a transfer learning model. I thought this model might be appropriate because that since it can perform very well in a very large size of dataset, the features it extractes may be very useful in my model. Then I changed the original output layer of ResNet50 to a dense layer with one dim output and used output of this layer as the prediction of angle. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Finally, both training and testing loss of this model were about 0.035. 0.035 looks pretty good, but when I apply this model to simulator, the car can hardly runing on the road. The problem may lie with the dataset，so I decide to collection more data from the track one.

After collecting enough data points, I applied the ResNet50 model into the dataset. Both training and testing loss of this model were about 0.06. But the performance in autonomous mode is not very well. The car always drives along left side of road and can't return to the center.

Then I used InceptionV3 model to replace ResNet50 model. Because the output shape of InceptionV3 is (5, 5, 2048), I add a flatten layer behind it. And then I add a output layer. After 10 epoches training, the training loss was about 0.1 while the testing loss was about 0.07. I think the model was underfitting. Then I added one fully-connected layer after the flatten layer. This time the training loss were about 0.05 but the testing loss was about 0.1. The model was overfitting! Then I added two dropout layer before and after the fully-connected layer and tried 5, 10, 15, 20 epoches to train the model. Finally I found the training and testing achieved minimum loss 0.05 when the `epoches` parameter was 9 or 10. When I applied the model to the simulator the car can run at the center of road pretty well.

There were a few spots where the vehicle fell off the track, such as the bridge, the curve after the bridge and the road with a huge granite on the left. To improve the driving behavior in these cases, I recorder more data point in these places.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 55-80) consisted of one cropping layer (code line 58), one normalizing layer (code line 60), one resizing layer (code line 62) and InceptionV3 model without output (code line 68), one flatten layer (code line 73), two dense layers and two dropout layers (code line 73-77).

Here is a visualization of the architecture.

![architecture][architecture]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 5 clockwise laps and 5 anticlockwise laps on track one using center lane driving. Here is an example image of center lane driving:

![Clockwise Example][clockwise_example]         ![Anticlockwise Example][anticlockwise_example]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to return to center. These images show what a recovery looks like starting from the right side:

![Right Side to Center][RightSideToCenter1]     ![Right Side to Center][RightSideToCenter2]
![Right Side to Center][RightSideToCenter3]     ![Right Side to Center][RightSideToCenter4]

These images show what a recovery looks like starting from the left side:

![Left Side to Center][LeftSideToCenter1]       ![Left Side to Center][LeftSideToCenter2]
![Left Side to Center][LeftSideToCenter3]       ![Left Side to Center][LeftSideToCenter4]

I also want to collect more data points on track two, but finally I found that track two were very different from the track one, when I combined both the two track data points, the performance of this model declined and both the loss of training set and validation set became very large. So finally I decided to remove the track data points.

To increase the dataset, I decided to use all of the images collected by the simulator (including left, center and right images).

After the collection process, I had 88917 number of data points. I then preprocessed this data by cropping, normalizing and resizing images.

I finally randomly shuffled the data set and put 30% of the data into a validation set (model.py line 52).

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
