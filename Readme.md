# ML Deployment Pipeline Testing
## Experiment : 

## **Introduction:**

In agricultural field, everything is natural from production to maintenance of crops, weather effects, etc. and lots of uncertainities involved, which require continious and efficient monitoring, and take quick action in situation basis, which is not feasible for individuals, thus technology introduces it's vision for those solutions. There maybe severl vision based applications, when it comes to agriculture. Our goal is to bring multiple vision base applications under the same hood, so that we are proposing to make a centralized cloud based application, which is as follow:

### Centralized Cloud based Vision APP

#### System architecture:

***System 1.(Cloud based only with some seasonality)***

Application related to GIS, Satellite imagery based application. This section works in cloud and continuously monitor (or analyze on request) and analyze the satellite imagery, GIS data from satellites. Purpose and implementation:

Model deployment for 
•        Grazing pattern, Boundaries of crop
•        Crop-loss detection(accuracy is questionable using only satellite or GIS, need literature review)
•        Crop type detection(accuracy is questionable using only satellite or GIS, need literature review)
•        Missed fertilizer stripes
•        Farm-land area
•        Erosion detection
•        Water-body detection
•        Pest-detection(using GIS or satellite only here, but drone image for this purpose can be incorporated here)
•        Yield estimation

***System 2. (Cloud based but data fetching from real-world drone, farmers phone data)*** 

Application related to Drone imagery based application. This section works in cloud and continuously monitor (or analyze on request) and analyze the imagery and provides real-time response, Purpose and implementation:

Model deployment for
•        Vision based leaf disease stage detection(Pestiside spray based actuation, done by control node) 
•        Water-body detection(during drone survey with some routine-basis)
•        Real-time vision based crop, fruit health monitor(2d/3d mapping or image based strategy)
•        Plant stress classification
•        Real-time analysis of leaf or crop health from the uploaded image by farmers-friendly app. 
•        Crop, fruit count, quantity, type detection in real-time for crop or fruit collection in real-time.

***System 3.*** 

A. Mobile Apps for Farmers:

    Case a (Where no issue in Internet): 

    Farmers can upload images of crops, leafs etc. for obtaining real-time analytics based on real-time cloud program through their id, which can get real-time verification, 
    identification and recommended solution too.

    Case b (A real-time device having wireless connectivity with phones, can work without Internet):

    Farmers can upload images of crops, leafs etc. for obtaining real-time analytics based on real-time program  running on that master device in nearby operating units through     their id, which can get real-time verification, identification response, and recommended solution too.

B. Deployed Vision system in Drones or bots that are used in real-time while surveying

    1. Vision Unit(or say ROS node)
    
    Does all vision based applications mentioned above from analytics to precise image captures(regarding efficient homography estimation in real-time),[autonomous path planning     though out of scope here], deployed in the drone computer.
    
    2. Control unit

    Takes analytics result from vision node for actuating motors to conduct pitch, roll, yaw, and 
    from past analysis or the code involved for domain analysis to perform efficient, and precise
    application of following
    •        Herbiside spray
    •        Pestiside spray
    •        Watering or other necessary drone based application
   
   Note: Based on the application or purpose
   
   1. Drone can perform, image, analytics in real-time on-board, specially the actuation 
   based problem like spraying something in field.
   2. The drone can capture images only for crop type, area, boundary etc. analysis in 
   routine basis, this analytics application can be done later by a code running on cloud to
   save on-board battery power. 
   
## **Objective** 

* Dataset Source
Here we are focuing on System 3 implementation, where we are proposing to have a cloud platform, where multiple vision based cloud services will be run for classifying different plant leaf diseases. Here we are focusing on Potato Leaf Disease classification. 

## **Dataset**

* Source: [Mendley Plant Village Dataset](https://data.mendeley.com/datasets/tywbtsjrjv/1)

* Description: In this data-set, 39 different classes of plant leaf and background images are available. the Plant Village Dataset consists of two different sets of data 

  **1. Original Dataset** contains 54303 healthy and unhealthy leaf images. 
  
  **2. Augmented dataset** containing 61,486 images. Six different augmentation techniques were used for increasing the data-set size. The techniques are  >image flipping, Gamma correction, noise injection, PCA color augmentation, rotation, and Scaling.

* Original Paper: [An open access repository of images on plant health to enable the development of mobile disease diagnostics](https://arxiv.org/abs/1511.08060)

#### **Potato Dataset**

Among the 39 classes there are 3 following categories of Potato classes available in the dataset. 
 * **Potato Early Blight:** Early blight of potato is caused by the fungal pathogen Alternaria solani. The disease affects leaves, stems and tubers and can reduce yield, tuber size, storability of tubers, quality of fresh-market and processing tubers and marketability of the crop. More on this can be found [here](https://www.ag.ndsu.edu/publications/crops/early-blight-in-potato#:~:text=Early%20blight%20of%20potato%20is,and%20marketability%20of%20the%20crop.)
![EB](https://github.com/Future-AI-Laboratory/vision-api/blob/review_sayan/images/Potato_early_blight.png)
* **Potato Late Blight:** Late blight, also called potato blight, disease of potato plants that is caused by the water mold Phytophthora infestans. The disease occurs in humid regions with temperatures ranging between 4 and 29 °C (40 and 80 °F). Hot dry weather checks its spread. Potato or tomato plants that are infected may rot within two weeks. More on this can be found [here](https://cropwatch.unl.edu/potato/late_blights_description)
![Late blight](https://github.com/Future-AI-Laboratory/vision-api/blob/review_sayan/images/late_blight.png)
 * **Potato Healthy:** Potato Healthy Leaves are those, which have no diseases, which are fresh and healthy.
![Healthy](https://github.com/Future-AI-Laboratory/vision-api/blob/review_sayan/images/healthy_new.png) 

## **Class Distribution**

In original and augmented dataset the class distribution is as foloows.

![Distribution](https://github.com/Future-AI-Laboratory/vision-api/blob/review_sayan/images/potato_distribution.PNG)

@sayan (fill your thought process here ) 












# Reference

* [TFX on Cloud AI Platform Pipelines](https://github.com/tensorflow/tfx/blob/master/docs/tutorials/tfx/cloud-ai-platform-pipelines.md)
* [Mobile Deep Learning with TFlite, ML kit and Flutter](https://drive.google.com/drive/folders/1TV3jpPnFbd4pxiVxxxEKxSsZibwUqoVo?usp=sharing), Use this for TFlite model conversion
* [Comparison Between Kubeflow and TFX](https://github.com/Future-AI-Laboratory/deployment-testing/blob/master/Comparison-Kubeflow-TFX.pdf)
* [TFX playlist](https://www.youtube.com/watch?v=YeuvR6m6ACQ&list=PLQY2H8rRoyvxR15n04JiW0ezF5HQRs_8F)
* [Using TensorFlow Extended (TFX) on AI Platform Pipelines live webinar](https://www.youtube.com/watch?v=RpWeVvAFzJE)
* [Using Kubeflow Pipelines on AI Platform, with Brian Kang along with git trigger](https://www.youtube.com/watch?v=qx7MLcbCo5g)
* [Continuous Deployment from git using Cloud Build | Cloud Run](https://cloud.google.com/run/docs/continuous-deployment-with-cloud-build)
* [How to Deploy a Machine Learning Model to Google Cloud for 20% Software Engineers (CS329s tutorial)](https://youtu.be/fw6NMQrYc6w)
* [Firebase Machine Learning BETA Machine learning for mobile developers](https://firebase.google.com/products/ml)
* [Image Augmentation on the fly using Keras ImageDataGenerator!](https://www.analyticsvidhya.com/blog/2020/08/image-augmentation-on-the-fly-using-keras-imagedatagenerator/)
* [Why and How to create requirements.txt](https://blog.usejournal.com/why-and-how-to-make-a-requirements-txt-f329c685181e)
