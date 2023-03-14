def view_random_image(target_dir, target_class):
  # Set up the target directory
  target_folder=target_dir+'/'+target_class
  # Get a random image path
  random_image=random.sample(os.listdir(target_folder),1)
  print(random_image)

  # Read in the image and plot it using matplotlib

  img=mpimg.imread(target_folder + '/' + random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis('off')

  print(f'Image shape: {img.shape}') # Show the shape of the image

  return img