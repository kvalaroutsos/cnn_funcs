# Create a function that import the image and reshape it to thr proper forms
def load_and_prep_image(filename,img_shape=224):
  """
  Read the image from filename, turns it to atensor and reshape it to (img_shape, img_shape, color_channels)
  """
  img=tf.io.read_file(filename)
  # Decode the read file to a tensor
  img=tf.image.decode_image(img)
  # Resize the image
  img=tf.image.resize(img,size=[img_shape, img_shape])
  #Rescale the image betwwen 0-1
  img=img/255.
  return img
