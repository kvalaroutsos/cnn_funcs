def pred_and_plot(model, filename, class_names=class_names):
  """
  THIS is for MULTICLASS predictions.
  Imports an image located as filename, makes a prediction with model, and
  plot the image with the predicted class as the title. 
  """
  # Import the target image and preprocess it
  img=load_and_prep_image(filename)

  # Make the prediction
  pred=model.predict(tf.expand_dims(img,axis=0))
  print(pred)

  # Get the predicted class

  pred_class=class_names[np.argmax(pred)]

  # Plot the image 
  plt.imshow(img)
  plt.title(f'Prediction: {pred_class}')
  plt.axis(False)