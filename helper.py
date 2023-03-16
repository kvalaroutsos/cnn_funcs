# Create a function that import the image and reshape it to thr proper forms
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


#Plot the validation and training curves separately
def plot_loss_curves(history):
  """
  Import the history of a tf model
  """
  
  import matplotlib.pyplot as plt
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

def view_random_image(target_dir, target_class):
  
  """
  Give the target dir (str) and the target class (str) to get a random image
  """"
  
  import matplotlib.pyplot as plt
  import matplotlib.image as mpimg
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
