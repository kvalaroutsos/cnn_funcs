# Create a function that import the image and reshape it to thr proper forms
import random
import os



import tensorflow as tf 
def load_and_prep_image(filename,img_shape=224):
  """
  Read the image from filename, turns it to a tensor and reshape it to (img_shape, img_shape, color_channels)
  """
  img=tf.io.read_file(filename)
  # Decode the read file to a tensor
  img=tf.image.decode_image(img)
  # Resize the image
  img=tf.image.resize(img,size=[img_shape, img_shape])
  #Rescale the image betwwen 0-1
  img=img/255.
  return img

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#Plot the validation and training curves separately
def plot_loss_curves(history):
  """
  Import the history of a tf model
  """
  
  
  loss=history.history['loss']
  val_loss=history.history['val_loss']

  accuracy=history.history['accuracy']
  val_accuracy=history.history['val_accuracy']

  epochs=range(len(history.history['loss']))

  #Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('loss')
  plt.xlabel('epochs')
  plt.legend()

    #Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('accuracy')
  plt.xlabel('epochs')
  plt.legend()

import tensorflow as tf 
def pred_and_plot(model, filename, class_names):
  """
  THIS is for MULTICLASS and BINARY  predictions.
  Imports an image located as filename, makes a prediction with model, and
  plot the image with the predicted class as the title. 
  """
  # Import the target image and preprocess it
  img=load_and_prep_image(filename)

  # Make the prediction
  pred=model.predict(tf.expand_dims(img,axis=0))
  

  # Get the predicted class

  if len(pred[0])>1:

    pred_class=class_names[tf.argmax(pred[0])]

  else:

    pred_class=class_names[int(tf.round(pred[0]))]

  # Plot the image 
  plt.imshow(img)
  plt.title(f'Prediction: {pred_class}')
  plt.axis(False)
  
  
import random
  
def view_random_image(target_dir, target_class):
  
  """
  Give the target dir (str) and the target class (str) to get a random image
  """
  
  # Set up the target directory
  target_folder=target_dir+'/'+target_class
  # Get a random image path
  random_image=random.sample(os.listdir(target_folder),1)
  

  # Read in the image and plot it using matplotlib

  img=mpimg.imread(target_folder + '/' + random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis('off')

  print(f'Image shape: {img.shape}') # Show the shape of the image

  return img


import zipfile

def unzip_data(filename):
  """
  Unzips filename into the current working directory.
  Args:
    filename (str): a filepath to a target zip folder to be unzipped.
  """
  zip_ref = zipfile.ZipFile(filename, "r")
  zip_ref.extractall()
  zip_ref.close()

 # Walk through an image classification directory and find out how many files (images)
# are in each subdirectory.
import os

def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.
  Args:
    dir_path (str): target directory
  
  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
    
# Function to evaluate: accuracy, precision, recall, f1-score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.
  Args:
      y_true: true labels in the form of a 1D array
      y_pred: predicted labels in the form of a 1D array
  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results
