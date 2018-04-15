# Analysis and Object Detection of Artworks with Tensorflow(GPU) on Windows 10

The following post was written by Microsoft Student Partner Alexander Gonzalez as part of final year Project in Computer Science. Thanks to Microsoft Azure, I had the ability to use sophisticated tools which I have never used before and I could not have finished my experiment without azure virtual machine hardware. This article is dedicated to all Spain Technical Community.
Nowadays, all major technology companies are committed to innovative projects using technologies such as Machine Learning, Deep Learning. Both will be the most important technologies for 2018, making a difference on the currently way of living and working. Machine learning and Deep learning are constantly making new business models, jobs, as well as, innovative and efficient industries.
Machine learning is around us more than we think. It can be found in mobile phones, cars, homes and in our own job, helping us to make better decisions, access to higher quality information and faster. According to several surveys and analytical companies, these technologies will be present in all the new software products in 2020.
From the business point of view, artificial intelligence (AI) will be used as a stepping stone to improve and increase the effectiveness and efficiency of the services of any company as well as quality for the user. The trends indicate that most sectors will see radical changes in the way they use all their products, the only risk of such change is to completely ignore it. In the future, products that we currently know will change thanks to the AI . Furthermore, and it is calculated that in the next three years, around 1.2 trillion dollars will change hands thanks to this new field in computing. Consequently, this means that every year the AI is taking a lot of strength and support; therefore, it will leave a mark setting differences between companies in the up-coming years.
The objective of these post series is to show the possibilities that we currently have to perform machine learning and computer vision projects which they will be published in 4 parts: 

1.	The first part is dedicated to the installation and explanation of the software that we will need to take any project in this case, with TensorFlow, CUDA and CuDNN.

2.	   The second part will cover step by step the necessary processes to make our dataset, in this case artworks images, followed by training. Finally, the evaluation and obtaining of relevant graphics for the preparation of documentation will be explained. Link to post

 
Image of Fast-RCNN on surface webcam with python program
 
Image of Fast-RCNN on MS Surface webcam with python program
 
Image of SSD-Mobilenet on LG mobile

3.	   The third post will explain another way of recognizing and classifying images (20 artworks) using scikit learn and python without having to use models of TensorFlow, CNTK or other technologies which offer models of convolved neural networks. Moreover, we will explain how to set up your own web app with python. For this part a fairly simple API which will collect information about the captured image of our mobile application in Xamarin will be needed, so it will inference with our model made in python, and it will return the corresponding prediction. With that methodology we can get easy classification without heavy hardware like TensorFlow or CNTK. Link to post 

4.	    Finally, the last post is dedicated to a comparison between the use of TensorFlow and CNTK, the results obtained, the management and usability, points of interest and final conclusions about the research that has been carried out for the development of all the posts. Link to post

## How to install Tensorflow 1.5 on Windows 10 with CUDA and CudaDNN

Prerequisites
•	Nvidia GPU (GTX 650 or newer. The GTX1050 is a good entry level choice)
•	Anaconda with Python 3.6(or 3.5)
•	CUDA ToolKit(versión 9)
•	CuDNN(7.1.1)

If we want to get results quickly the first thing to think about is with what hardware we will face our computer vision project, since the demands are high in terms of GPU and processing. My advice is to use Data Science Virtual Machine that Azure supports. They are complete virtual machines preconfigured for the modelling, development and implementation of science data. Below, I highlight several documents provided by Azure team that will help us understand the provisioning of these machines and prices: 

•	https://azure.microsoft.com/es-es/services/virtual-machines/data-science-virtual-machines/
•	https://docs.microsoft.com/es-es/azure/machine-learning/data-science-virtual-machine/provision-vm
•	https://docs.microsoft.com/es-es/azure/machine-learning/data-science-virtual-machine/overview

*Note: We must install a Data Science virtual machine of NC6 or NV6 model with GPU. 

Once we have our machine created, we can start to install

## Download CUDA 9.0
First of all, we will have to install CUDA Tool Kit:
•	Download version 9.0 here: https://developer.nvidia.com/cuda-downloads
•	Currently version 9.0 is supported by Tensorflow 1.5
 
Installer type exe(network) is the lighter way one. More complete one is exe(local)
Set your environment Variables:

•	Go to start and search “environment variables”
•	Click th environmen variables button
•	Add the following paths: 
  o	C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\extras\CUPTI\libx64
  o	C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\libnvvp
  o	C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin
  
In data Science machine we have CUDA install and we can found this path already install:
•	Variable name: CUDA_PATH_V9_0
•	Variable value: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0

## Download CUDNN 8.0
•	Go to https://developer.nvidia.com/rdp/cudnn-download
•	Create a user profile if needed
•	Select CUDNN 7.1.1 for CUDA tool kit 9.0
•	Extract the zip file and place them somewhere(C:// directory for example)
•	Add a in environment variable path to the bin folder: For example: C:/cuda/bin

## Update GPU Driver
•	Go to http://www.nvidia.com/Download/index.aspx
•	Select your version to download
•	Install the drive
 
## Install Anaconda 3.6
In these VM Anaconda environment has been installed. Howver, the installation steps of Anaconda were not explained in this post.

## Install Tensorflow GPU on Windows 10
Open your cmd window and type the following command:

pip install –ignore-installed –upgrade tensorflow-gpu

*Notes Tensorflow documentation:
https://www.tensorflow.org/install/

Test your tensorflow install:
1.	Open anaconda prompt
2.	Write “python –-version”
3.	Write “python”
4.	Once the interpreter opens type the following:
   	import tensorflow as tf
   	hello= tf.constant('Hello, Tensorflow!')
   	Sess = tf.session()
   	print(sess.run(hello))

In the next post, we will cover step by step the necessary processes to make our dataset. In this case, artwork images, followed by training and finally the evaluation and obtaining relevant graphics for document preparation. Index of the next post:
1.	Labelling pictures
2.	Generating training data
3.	Creating a label map and configuring training
4.	Training
5.	Exporting the inference graph
6.	Testing and using your newly trained object detection classifier

# How to train an object detector classifier for multiple objects using tensorflow(GPU) on Windows 10

Once we have installed Tensorflow, Cuda and CuDNN, we can pass to the next level! The purpose of this post is to explain how to train your own convolutional neural network object detection classifier for multiple objects. In this post we do not develop a nueronal network of zero, we will only take this github:

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

It provide a collection of detection models pre-trained datasets of differents kind of objects. They are also useful for initializing your models when training on novel datasets like our dataset of artworks. At the end of this post, you will have the idea to develope your own model that can identify and program that can draw boxes around specific ítem in that case 12 items objects of different artworks, like Gioconda, Athens School, Las Meninas, The Last Supper.
 
There are several good tutorials available for how to use TensorFlow’s Object Detection API to train a classifier for a single object, like this one:
https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0

The post is written for Windows 10. But without forgetting Linux, only emphasize that the development process is very similar and can also be used for Linux operating systems, but the file paths and package installation commands should be modified accordingly. Only tell you before start that TensorFlow-GPU allows your PC or VM to use the video card to provide extra processing power while training, so it will be used for this post, Regular Tensorflow not. In my experience doing this experiment, using TensorFlow-GPU instead of regular TensorFlow reduces training time by a factor of about 8hr using Fast-RCNN model, 3 hours to train instead of 8 hours with SSD-Mobilenet model for 3 objects, 21 hours for 12 objects. You can search differents models to tensorflow here:

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

Regular TensorFlow can also be used for this post, I have tried it in my PC, but it will take longer. If you use regular TensorFlow, you do not need to install CUDA and cuDNN like in teh first post [Link to post]. 

## Installing TensorFlow-GPU

Install Tensorflow-GPU following the instructions of the first post. Dowload CUDA 9.0 and CuDNN 7.1.1, but you will likely need to continue updating the CUDA and CuDNN versions to the latest supported version.

## Setting up the Object Detection directory structure and Anaconda Virtual Environment

The TensorFlow Object Detection API requires using the specific directory structure provided in its GitHub repository. It also requires several additional Python packages, specific additions to the PATH and PYTHONPATH variables, and a few extra setup commands to get everything set up to run or train an object detection model. Stay tuned at each step because the processes are very meticulous and the minimum failure can give us error.
### Download TensorFlow Object Detection API repository from GitHub
Create a folder directly in C: and name it “tensorflow” or whatever you want. This working directory will contain the full TensorFlow object detection framework, as well as your training images, training data, trained classifier, configuration files, and everything else needed for the object detection classifier.
Download the full TensorFlow object detection repository located at https://github.com/tensorflow/models by clicking the “Clone or Download” button and downloading the zip file. Open the downloaded zip file and extract the “models-master” folder directly into your C:\DIRECTORY you just created. Rename “models-master” to just “models”.
 
Image of the principal directory
 
Image of Tensorflow Object Detection API directory
 
Image of Tensorflow Object Detection API, Research directory

### Download the Faster-RCNN and SSD-Mobilenet models
TensorFlow provides several object detection models (pre-trained classifiers with specific neural network architectures) in its model zoo. If we look at the README of this github we can see that they tell us the importance within each model of three files.
 
In the following sections we will explain what we should do with them, in the case of this project we have used ssd_mobilenet_v1_coco and faster_rcnn_inception_v2_coco. But before I would like to explain the importance of understanding the following table of models proposed by tensorflow. Especially understand well the speed, and the mAP column.
 
*Note: What is column mAP? https://stackoverflow.com/questions/46094282/why-we-use-map-score-for-evaluate-object-detectors-in-deep-learning

As we can see in the table SSD-Mobilenet has much more speed than the other models and smaller mAP, we could conclude that it is the best but once my project has been realized I have noticed that for example in this case with art paintings, it has given good results in terms of detection but in terms of accuracy it give me low level of success. On the other hand faster rcnn has surprised me because according to the table it has a very slow detection speed with respect to SSD-Mobilenet but it has gone very well on my MS Surface webcam. At the end to have been analyzing both models decided to train with both and extract their inference graph to use it both on tablets, laptops and mobile or Iot devices. You can choose which model to train your objection detection classifier on. If you are planning on using the object detector on a device with low computational like mobile, use the SDD-MobileNet model. If you will be running your object detector on a laptop or desktop PC, use one of the RCNN models.

In this post will use the Faster-RCNN-Inception-V2 model and ssd_mobilenet_v1_coco. Download the models and open the downloaded faster_rcnn_inception_v2_coco_2018_01_28.tar.gz and ssd_mobilenet_v1_coco_2017_11_17.tar.gz file with a file archiver such as WinZip or 7-Zip and extract the faster_rcnn_inception_v2_coco_2018_01_28 and ssd_mobilenet_v1_coco_2017_11_17 folder to the C:\tensorflow1\models\research\object_detection folder. 

*Note: The models date and versions will likely change in the future, but it should still work with this tutorial.

### Download this tutorial's repository from GitHub

Download the full repository located on this page, scroll to the top and click Clone or Download and extract all the contents directly into the C:\tensorflow1\models\research\object_detection directory. This establishes a specific directory structure that will be used for the resto of the post. At this point, your \object_detection folder should look like:
 
Final directory with Object Detection API with 11 extra-files(selected)

This repository contains the images, annotation data, .csv files, and TFRecords needed to train a "Artworks" objects detector. It also contains Python scripts that are used to generate the training data. It has scripts to test out the object detection classifier on images, videos, or a webcam feed.

If you want to practice training your own "Artwork" Detector, you can leave all the files as they are. You can follow along with this tutorial to see how each of the files were generated, and then run the training. You will still need to generate the TFRecord files train.record and test.record as described in nexts steps.

You can also use the frozen inference graph for my trained Artwork model detector from the I and extract the contents to \object_detection\inference_graph. This inference graph will work "out of the box". You can test it after all the setup instructions have been completed by running the Object_detection_image.py or video or webcam script.

If you want to train your own object detector, delete the following files (do not delete the folders):

•	All files in \object_detection\images\train and \object_detection\images\test
•	The “test_labels.csv” and “train_labels.csv” files in \object_detection\images
•	All files in \object_detection\training
•	All files in \object_detection\inference_graph

Now, you are ready to start from scratch in training your own object detector. This post will assume that all the files listed above were deleted, and will go on to explain how to generate the files for your own training dataset.
Set up new Anaconda virtual environment
Next, we'll work on setting up a virtual environment in Anaconda for tensorflow-gpu. From the Start menu in Windows, search for the Anaconda Prompt utility, right click on it, and click “Run as Administrator”.

Activate the environment of tensorflow by issuing:
C:\> activate tensorflow

Install the other necessary packages by issuing the following commands:
(tensorflow) C:\> conda install -c anaconda protobuf
(tensorflow) C:\> pip install pillow
(tensorflow) C:\> pip install lxml
(tensorflow) C:\> pip install jupyter
(tensorflow) C:\> pip install matplotlib
(tensorflow) C:\> pip install pandas
(tensorflow) C:\> pip install opencv-python

*Note: The ‘pandas’ and ‘opencv-python’ packages are not needed by TensorFlow, but they are used in the Python scripts to generate TFRecords and to work with images, videos, and webcam feeds.

### Configure PATH and PYTHONPATH environment variables
The PATH variable must be configured to add the \models, \models\research, and \models\research\slim directories. bor some reason, you need to create a PYTHONPATH variable with these directories, and then add PYTHONPATH to the PATH variable. Otherwise, it will not work.

Do this by issuing the following commands (from any directory):

(tensorflow) C:\> set PYTHONPATH=C:\tensorflow1\models; C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
(tensorflow1) C:\> set PATH=%PATH%; PYTHONPATH

*Note: Every time the "tensorflow1" virtual environment is exited, the PATH and PYTHONPATH variables are reset and need to be set up again.
 
### Compile Protobufs and run setup.py

Now we need to compile the Protobuf files, which are used by TensorFlow to configure model and training parameters. Unfortunately, the short protoc compilation command posted on TensorFlow’s Object Detection API installation page does not work on Windows. Every .proto file in the \object_detection\protos directory must be called out individually by the command.

To understand what the protobuf files are (Protocol Buffer, Mechanism for serializing structured data by Google): 

•	https://www.tensorflow.org/extend/tool_developers/
•	https://developers.google.com/protocol-buffers/?hl=en

In the Anaconda Command Prompt, change directories to the \models\research directory and copy and paste the following command into the command line and press Enter:

protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto

This creates a name_pb2.py file from every name.proto file in the \object_detection\protos folder.

*Note: TensorFlow occassionally adds new .proto files to the \protos folder. You may need to update the protoc command to include the new .proto files.
 
Finally, run the following commands from the C:\tensorflow1\models\research directory:
(tensorflow) C:\tensorflow1\models\research> python setup.py build
(tensorflow) C:\tensorflow1\models\research> python setup.py install

## Labelling pictures

Now that the TensorFlow Object Detection API is all set up and ready to go, we need to provide the images it will use to train a new detection classifier.
### Gather Pictures

TensorFlow needs hundreds of images of an object to train a good detection classifier. To train a robust classifier, the training images should have random objects in the image along with the desired objects, and should have a variety of backgrounds and lighting conditions. There should be some images where the desired object is partially obscured, overlapped with something else, or only halfway in the picture.

For my Artwork Detection classifier, I have twelve different objects I want to detect (Diego Velazquez, Maria Agustina, Infanta Margarita, Isaben de Velasco and Mari barbola part of “Las Meninas”, Platon, aristoteles and heraclito of “School of Athens”, Gioconda of “Mona Lisa” and finally Jesus, Judas and Mateo of “The Last Supper”). 

I used Google Image Search to find about 80 pictures of each artwork. You can use your phone or download images of the objects from Google Image Search. I recommend having at least 200 pictures overall. I used 310 aprox, pictures to train my artworks detector. In addition to a better training has been cut the complete image of the art box focusing on the object we want to detect.
 
Images of the different artworks cropped

In the case of art paintings, we are lucky that they are static images that will always be presented to our model in the same way, they are not objects with movement or relevant changes of appearance such as an animal, cars, etc. With the images of the art pictures I played when choosing the images for the dataset with the main properties of computer vision, luminosity, size cause. They always appear in the same way, we can only extract real photos with people in front of the picture as another training case.
You can use the resizer.py script in this repository to reduce the size of the images cause they should be less than 200KB each, and their resolution shouldn’t be more than 720x1280. The larger the images are, the longer it will take to train the classifier.
After you have all the pictures you need, move 20% of them to the \object_detection\images\test directory, and 80% of them to the \object_detection\images\train directory. Make sure there are a variety of pictures in both the \test and \train directories.

### Label Pictures

With all the pictures gathered, it’s time to label the desired objects in every picture. Labellimg is a great and new tool for me for labeling images, and its GitHub page has very clear instructions on how to install and use it.

LabelImg GitHub link

LabelImg download link

Download and install LabelImg, point it to your \images\train directory, and then draw a box around each object in each image. Repeat the process for all the images in the \images\test directory. This will take a while and a lot of patience…

Image open file directory and observe what objects we want

 
Image about create RectBox around objects

## Generating training data

LabelImg saves a .xml file containing the label data for each image. These .xml files will be used to generate TFRecords, which are one of the inputs to the TensorFlow trainer. Once you have labeled and saved each image, there will be one .xml file for each image in the \test and \train directories.
 
Image of XML file generate by LabelImg with the objects with their respectives coordinates.
With the images labeled, it’s time to generate the TFRecords that serve as input data to the TensorFlow training model. This tutorial uses the xml_to_csv.py and generate_tfrecord.py scripts from Dat Tran’s Raccoon Detector dataset, with some slight modifications to work with our directory structure.

First, the image .xml data will be used to create .csv files containing all the data for the train and test images. From the \object_detection folder, issue the following command in the Anaconda command prompt:
(tensorflow) C:\tensorflow1\models\research\object_detection> python xml_to_csv.py

This creates a train_labels.csv and test_labels.csv file in the \object_detection\images folder.
 
Image of train and test csv files from xml file of LabelImg
 
Image inside train and test .csv file
Next, open the generate_tfrecord.py file in a text editor. Replace the label map starting at line 31 with your own label map, where each object is assigned an ID number. This same number assignment will be used when configuring the labelmap.pbtxt file.
You will replace the following code in generate_record.py:

TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'Diego Velazquez':
        return 1
    elif row_label == 'Maria Agustina':
        return 2
    elif row_label == 'Infanta Margarita':
        return 3
    elif row_label == 'Isabel de Velasco':
        return 4
    elif row_label == 'Mari Barbola':
        return 5
    elif row_label == 'Platon':
        return 6
    elif row_label == 'Aristoteles':
        return 7
    elif row_label == 'Heraclito':
        return 8
    elif row_label == 'Gioconda':
        return 9
    elif row_label == 'Jesucristo':
        return 10
    elif row_label == 'Judas':
        return 11
    elif row_label == 'Mateo':
        return 12
    else:
        None
        
Then, generate the TFRecord files by issuing these commands from the \object_detection folder:

python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record

These generate a train.record and a test.record file in \object_detection. These will be used to train the new object detection classifier.

## Creating a label map and configuring training

The last thing to do before training is to create a label map and edit the training configuration file.

### Label map

The label map tells the trainer what each object is by defining a mapping of class names to class ID numbers. Use a text editor to create a new file and save it as labelmap.pbtxt in the C:\tensorflow1\models\research\object_detection\training folder. Don’t forget that the file type is .pbtxt, not .txt. In the text editor, copy or type in the label map in the format below. The example below is the label map for my Artworks Detector:
item {
  id: 1
  name: 'Diego Velazquez'
}

item {
  id: 2
  name: 'Maria Agustina'
}

item {
  id: 3
  name: 'Infanta Margarita'
}

item {
  id: 4
  name: 'Isabel de Velasco'
}

item {
  id: 5
  name: 'Mari Barbola'
}

item {
  id: 6
  name: 'Platon'
}

item {
  id: 7
  name: 'Aristoteles'
}

item {
  id: 8
  name: 'Heraclito'
}

item {
  id: 9
  name: 'Gioconda'
}


item {
  id: 10
  name: 'Jesucristo'
}

item {
  id: 11
  name: 'Judas'
}

item {
  id: 12
  name: 'Mateo'
}

### Configure training

Finally, the object detection training pipeline must be configured. It defines which model and what parameters will be used for training. This is the last step before running training 😊

In this step we can do the same with SSD-Mobilenet and Faster-RCNN. First we will do the training with Faster-RCNN and then I will explain how it would be done with the Mobilenet model.

Navigate to C:\tensorflow1\models\research\object_detection\samples\configs and copy the faster_rcnn_inception_v2_pets.config file into the \object_detection\training directory. Then, open the file with a text editor. There are several changes to make to the .config file, mainly changing the number of classes and examples, and adding the file paths to the training data.

Make the following changes to the faster_rcnn_inception_v2_pets.config file. Note: The paths must be entered with single forward slashes (NOT backslashes), or TensorFlow will give a file path error when trying to train the model.  Also, the paths must be in double quotation marks (“ ), not single quotation marks ( ' ).

•	Line 9. Change num_classes to the number of different objects you want the classifier to detect. For the above basketball, shirt, and shoe detector, it would be num_classes : 3 .
•	Line 110. Change fine_tune_checkpoint to:
  o	fine_tune_checkpoint : "C:/tensorflow1/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"
•	Lines 126 and 128. In the train_input_reader section, change input_path and label_map_path to:
  o	input_path : "C:/tensorflow1/models/research/object_detection/train.record"
  o	label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"
•	Line 132. Change num_examples to the number of images you have in the \images\test directory.
•	Lines 140 and 142. In the eval_input_reader section, change input_path and label_map_path to:
  o	input_path : "C:/tensorflow1/models/research/object_detection/test.record"
  o	label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"
  
Save the file after the changes have been made. Great! Now we can start the training job!
For Mobilenet of other model we have to do the same process, get the config file of each model and change the parameters that I explained before. The model of SSD-Mobilenet you can find it here like Faster-RCNN:
 
Image directory of Tensorflow pre-trained models(Coco or Pets datasets)

## Training

From the \object_detection directory, issue the following command to begin training:
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config

If we want to use SSD-Mobilenet to train your model, you only need to change the config_path:
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config

If everything has been set up correctly, TensorFlow will initialize the training. When training begins, it will look like this:
 
Image training with Faster RCNN.

Each step of training reports the loss. It will start set up our graphic card (Tesla K80) then get the parameters of our model.config. This step start with high los and get lower and lower as training progresses. For my training on the Faster-RCNN-Inception-V2 model, it started at about 3.0 and quickly dropped below 0.8. I recommend allowing your model to train until the loss consistently drops below 0.05, which will take about 40,000 steps, or about 2-3 hours depending on how powerful your CPU and GPU are. 

*Note: The loss numbers will be different if a different model is used. MobileNet-SSD starts with a loss of about 40, and should be trained until the loss is consistently under 2.
 
Image Training SSD-Mobilenet

You can view that FasterRCNN training los is more faster than SSD-Mobilenet. Also you can view progress of the training job by using TensorBoard. Thanks to tensorboard I have been able to explain in my final project through graphs like training has been from beginning to end, demonstrating also differences between both models.

https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard

To do this, open a new instance of Anaconda Prompt, activate the tensorflow virtual environment, change to the C:\tensorflow1\models\research\object_detection directory, and issue the following command:
(tensorflow) C:\tensorflow1\models\research\object_detection>tensorboard --logdir=training

This will create a webpage on your local machine at PCNAME:6006, which can be viewed through a web browser. The TensorBoard page provides information and graphs that show how the training is progressing. One important graph is the Loss graph, which shows the overall loss of the classifier over time.
    
The training routine periodically saves checkpoints about every five minutes. You can terminate the training by pressing Ctrl+C while in the command prompt window. I typically wait until just after a checkpoint has been saved to terminate the training. You can terminate training and start it later, and it will restart from the last saved checkpoint. The checkpoint at the highest number of steps will be used to generate the frozen inference graph.

## Create and exporting the frozen inference graph

Now that training is complete, the last step is to generate the frozen inference graph (.pb file). From the \object_detection folder, issue the following command, where “XXXX” in “model.ckpt-XXXX” should be replaced with the highest-numbered .ckpt file in the training folder:

python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph

This creates a frozen_inference_graph.pb file in the \object_detection\inference_graph folder. The. pb file contains the object detection classifier.

## Testing and using your newly trained object detection classifier
Now that we have our classifier ready we can put it to the test. I enclosed them in the github repository Python scripts developed by Evan Juras to test it on an image, video or source of the webcam.
Before executing the Python scripts, we will have to change several code variables, then those changes are shown in the image:
 
Image Example of changes in Object_Detection_video.py

## Common errors throughout the Project

1.	Google may add more .proto files to the object_detection/protos folder, so it may be necessary to add more files to the "protoc" command at 13:13. You can do this by adding ".\object_detection\protos\FILENAME.proto" to the end of the long command string for each new file.

2.	When running the "python train.py" command, if you get an error that says "TypeError: __init__() got an unexpected keyword argument 'dct_method'.", then remove the "dct_method=dct_method" argument from line 110 of the object_detection/data_decoders/tf_example_decoder.py file.

3.	When running "python train.py", if you get an error saying "google.protobuf.text_format.ParseError: 110:25 : Expected string but found: '“' ", try re-typing the quotation marks around each of the filepaths. If you copied the filepaths over from my GitHub tutorial, the quotation marks sometimes copy over as a different character type, and TensorFlow doesn't like that.


4.	For train.py, if you get an error saying "TypeError: Expected int32, got rang
e(0, 3) of type 'range' instead.", it is likely an issue with the learning_schedules.py file. In the \object_detection\utils\learning_schedules.py file, change line 153 

5.	from "tf.constant(range(num_boundaries), dtype=tf.int32)," to "tf.constant(list(range(num_boundaries)), dtype=tf.int32),".

## Next post!
The third post will explain another way of recognizing and classifying images (20 artworks) using scikit learn and python without having to use models of TensorFlow, CNTK or other technologies which offer models of convolved neural networks. Moreover, we will explain how to set up your own web app with python. For this part a fairly simple API which will collect information about the captured image of our mobile application in Xamarin will be needed, so it will inference with our model made in python, and it will return the corresponding prediction. With that methodology we can get easy classification without heavy hardware like TensorFlow or CNTK. 
*Note: Also explain how can I prove my frozen inference model in Android!
