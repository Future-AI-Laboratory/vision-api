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
   
## **Objective:** 

Here we are focuing on System 3 implementation, where we are proposing to have a cloud platform, where multiple vision based cloud services will be run for classifying different plant leaf diseases. Here we are focusing on Potato Leaf Disease classification.

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
