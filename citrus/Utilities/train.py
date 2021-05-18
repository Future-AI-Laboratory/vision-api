'''
Author: Vision-team
This is used to run the entire pipeline
'''
import os
import data_loader as dl
# import data_analysis
import data_preperation
import model_build as mb
import model_training as mt
import evaluate as ev
import tflite_conversion as tc
import save_model
import logging
import environment_setup
import argparse



def main(args):
  try:
    environment_setup.environment_setup()
 
    download_flag = args.df
    # data_analysis_flag = args.da
    train_flag = args.tf

    
    if download_flag: 
      download_link = input("Enter url of zip file to download: ")     
      data_load = dl.datastore(download_link = download_link, download_flag = download_flag)
    
      zip_file = input("Enter zip file link: ")
      # unzip folder
      destination = input("Enter destination folder: ")
      data_load.unzip(zip_file, destination)

      img_directory = destination
    
    else:
      destination = input("Enter Image directory: ")
      dataset_folder = args.folder
      img_directory = destination
     
    print(f'Image Directory "{img_directory}"')
    
    # potato_image_path = 'Citrus/Leaves'  
    # potato_classifier = ['Black spot','Melanose','canker','greening','healthy']
    
    # print('\nData Analysis...\n')
    # # we can generalize that for any of the other crop in the plant village dataset
    # crop_type = "Potato"
    # # data analysis object
    # da = data_analysis.data_inspection(dataset_path, potato_image_path, potato_classifier, crop_type)
    
    # # fetch label dataframe, image_dataframe, shape flag
    # label_df, img_df, flag = da.fetch_img_info()
    
    # # if data analysis vis is active
    
    # if data_analysis_flag:
    #   da.dataset_content()
    
    #   print(f'\nClassifiers analysis\n{label_df}')
    #   print(f'\nImage samples analysis\n{img_df}')
    #   da.distribution_vis()

    print("\nData Preperation...\n")
    dp = data_preperation.data_preperation()

    print("\nTrain, Test generator creation...\n")

    train_gen, valid_gen,test_gen = dp.data_generator(img_directory =img_directory , class_mode='sparse',horizontal_flip=True,vertical_flip=True)

    # fetching final labels dictionary, can be used for tflite label.txt
    labels_dict = dp.datagen_check()
    print(f'\nClassifiers structure returned by the data generator with better reproducibility...\n{labels_dict}')
    
    print("\n CNN Model Building...\n")
    
    # model build object
    CNNmb = mb.model_build(num_classes=4)
    # build and log model to wandb
    model_config = CNNmb.build_model_and_log()
    # print(f'\nModel config\n{model_config[0]}')
    # print(f'\nProject name {model_config[1]}')
    # print(f'\nModel artifact name {model_config[2]}')
    
    # training
    if train_flag:  
      print('\nModel Training...\n')
      # model training object
      mdt = mt.model_train()
      # train and log to wandb
      model = mdt.train_and_log(train_gen, valid_gen)
      
    print('\nModel Evaluation on latest model...\n')
    el = ev.evaluate(test_gen)
    model = el.evaluate_and_log()
    
    print('\nSaving Model .json, weights.h5...\n')
    save_model.save_model().save_model(model)
    
    if args.tlf:
      # model.json, weights.h5, tflite conversion
      print('\nTflite Conversion...\n')
      tfl = tc.model_conversion()
      tfl.tflite_conversion(model, labels_dict)

  except Exception as e:
    logging.exception("message")


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = "Citrus Disease Classification Training Pipeline")
  
  # datasetfolder path, default 'Potato' for Potato dataset
  parser.add_argument('-path', '--folder', default = 'Citrus', type = str,
                     help='Path where the dataset is unzipped/ will be unzipped')
  # download flag argument, whether to download, unzip
  # default False, stores True i.e, if specify --df, args.df is True
  parser.add_argument("--df", help='Defines whether to download or not.',
                    action="store_true")
  # default False, stores True i.e, if specify --da, args.df is True
  # parser.add_argument("--da", help='Defines whether to show data analysis or not',
  #                   action="store_false")
  # default False, stores True i.e, if specify --tf, args.df is True
  parser.add_argument("--tf", help='Defines whether to include training or not in that run.',
                    action="store_true")
  # default False, stores True i.e, if specify --dlf, args.df is True
  parser.add_argument("--tlf", help='Defines whether to convert tflite.',
                    action="store_true")
  
  main(parser.parse_args())   