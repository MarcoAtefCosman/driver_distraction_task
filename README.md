[//]: # (Image References)

[image1]: ./images/images_analysis.png "data visualization"
[image2]: ./images/example_training_set.png "training set"
[image3]: ./images/graphs_1.png "training results"
[image4]: ./images/graphs_2.png "training results 2"
[image5]: ./images/example_testing.png "testing"



# Driver Distraction using inners camera

**In this Task, I used a deep neural network (ResNet50 model) (built with [Keras](https://keras.io/)) to detect if the driver is distracted away from the road.**

**The dataset used to train the network is  from [Kaggle's State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data), and it consists of images of ten classes represent the drivers behaviors.**



## Pipeline architecture:

- **Data Loading and visualization.**
- **Train-Validation Split.**
- **Load and fine tune ResNet50.**
- **Model Training.**
- **Test our final Model.**
I'll explain each step in details below.



## Step 1: Data Loading and visualization.

As mentioned before I used the Kaggle state farm distracted driver detection for this task it includes 10 classes represent the driver behavior while monitoring the road, the classes are (safe driving,texting - right,talking on the phone - right,texting - left,talking on the phone - left,operating the radio,drinking,reaching behind,hair and makeup,talking to passenger) with almost 22424 images with nearly 2000 image for each class, distributed as below:
![alt text][image1]


## Step 2: Train-Validation Split.

After loading all the images of the ten classes in a single list, I shuffled them and split it according to 0.8:0.2 ratio for training and the validation sets and decoded the labels, the images below are examples of the used data set for training.
![alt text][image2]


## Step 3: Load and fine tune ResNet50.

For this Task it's common to use the technique of cross validation to choose the training set and use transfer learning through different models like ResNet,MobileNet,VGG,etc.. to choose the best of them.
I used the fine-tuned ResNet50 model after loading it's Convolutional part through Keras applicatons then added a sequential layers represents the ten classes in the dataset. 

## Step 4: Model Training.

through very simple technique, I used gradinet descent optimizer with learning rate=0.001 and calculated the loss using categorial crossentropy loss function as the y labels are encoded to one_hot vectors.
Trained the model for 40 epochs with batch size=8, this choice of batch size is not the optimal choice and it can be seen that the losses are not stable decreasing according to the size of the batch size, but unfortunately I had to use this size according to the Lack of GPU power mode I use, I also couldn't use an image generator with data augmentation choices due to the lack of GPU power..

But although of that I reached a good results with training accuracy = 95.5% and validation accuracy = 97.06%

the graphs below show the training and loss curves:

![alt text][image3]
![alt text][image4]


## Step 5: Test our final Model.

after saving the best set of weights, I loaded it through our model and tested it using set of the available image set:

![alt text][image5]



## Step 6: Future enhancment.
-  Use high GPU power, so can use the image generator and larger batch size.
-  Implement the same task using PyTorch.


[the google colab link](https://colab.research.google.com/drive/16Ucvq2ll-JdIYleQJi9QzIsPbbiDX6QN)
