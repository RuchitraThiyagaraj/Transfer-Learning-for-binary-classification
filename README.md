# Transfer-Learning-for-binary-classification
## Aim
To Implement Transfer Learning for Horses_vs_humans dataset classification using InceptionV3 architecture.
## Problem Statement and Dataset
The objective of this project is to classify images from the Horses vs. Humans dataset using the InceptionV3 architecture through transfer learning. This dataset presents a binary classification challenge, distinguishing between images of horses and humans. By leveraging pre-trained weights from the InceptionV3 model, we aim to enhance classification accuracy and reduce training time. The project will evaluate the model's performance based on metrics such as accuracy, precision, and recall. Ultimately, the goal is to demonstrate the effectiveness of transfer learning in image classification tasks.
![image](https://github.com/user-attachments/assets/ee8f1838-891e-4317-8176-41946084867e)
</br>
</br>
</br>

## DESIGN STEPS
### STEP 1:
Load the InceptionV3 model without top layers and freeze its weights.
</br>

### STEP 2:
Add custom layers for classification and compile the model.
</br>

### STEP 3:
Prepare training and validation datasets with data augmentation.
<br/>

### STEP 4:
Train the model using the fit method with an early stopping callback.
<br/>

### STEP 4:
Plot training and validation accuracy/loss for performance evaluation.
<br/>

## PROGRAM
Include your code here
```python
# Import all the necessary files!
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from os import getcwd

path_inception = '/content/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

# Import the inception model
from tensorflow.keras.applications.inception_v3 import InceptionV3

# Create an instance of the inception model from the local pre-trained weights
local_weights_file = path_inception

pre_trained_model = InceptionV3(include_top = False,
                                input_shape = (150, 150, 3),
                                weights = 'imagenet')

pre_trained_model.load_weights(local_weights_file)

# Make all the layers in the pre-trained model non-trainable
for layer in pre_trained_model.layers:
  layer.trainable=False

# Print the model summary
# Write Your Code
print('Name:T.RUCHITRA Register Number:212223110043')

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output.shape)
last_output = last_layer.output

# Expected Output:
# ('last layer output shape: ', (None, 7, 7, 768))

# Define a Callback class that stops training once accuracy reaches 97.0%
# Write your code here
class my_callback(tf.keras.callbacks.Callback):
  def on_epoch_end(self,model,logs=None):
    if(logs.get('accuracy')>0.97):
      self.model.stop_training=True

x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=pre_trained_model.input, outputs=x)

model.compile( 
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00001), 
    loss='binary_crossentropy', 
    metrics=['accuracy'] 
)

print('Name:T.RUCHITRA Register Number:212223110043')
model.summary()

# Get the Horse or Human dataset
path_horse_or_human = '/content/horse-or-human.zip'
# Get the Horse or Human Validation dataset
path_validation_horse_or_human = '/content/validation-horse-or-human.zip'
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import zipfile

local_zip = path_horse_or_human
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/training')
zip_ref.close()

local_zip = path_validation_horse_or_human
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/validation')
zip_ref.close()

# Define our example directories and files
train_dir = '/tmp/training'
validation_dir = '/tmp/validation'

train_horses_dir = os.path.join(train_dir, 'horses')
train_humans_dir = os.path.join(train_dir, 'humans')
validation_horses_dir = os.path.join(validation_dir, 'horses')
validation_humans_dir = os.path.join(validation_dir, 'humans')

train_horses_fnames = os.listdir(train_horses_dir)
train_humans_fnames = os.listdir(train_humans_dir)
validation_horses_fnames = os.listdir(validation_horses_dir)
validation_humans_fnames = os.listdir(validation_humans_dir)

print(len(train_horses_fnames))
print(len(train_humans_fnames))
print(len(validation_horses_fnames))
print(len(validation_humans_fnames))

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1/255,
                                  height_shift_range = 0.2,
                                  width_shift_range = 0.2,
                                  horizontal_flip = True,
                                  vertical_flip = True,
                                  rotation_range = 0.4,
                                  shear_range = 0.1,
                                  zoom_range = 0.3,
                                  fill_mode = 'nearest'
                                  )

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale = 1/255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                   target_size = (150, 150),
                                                   batch_size = 20,
                                                   class_mode = 'binary',
                                                   shuffle = True)

# Flow validation images in batches of 20 using test_datagen generator
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                        target_size = (150, 150),
                                                        batch_size =20,
                                                        class_mode = 'binary',
                                                        shuffle = False)

# Expected Output:
# Found 1027 images belonging to 2 classes.
# Found 256 images belonging to 2 classes.

history = model.fit(
    train_generator,
    validation_data = validation_generator,
    epochs = 100,
    verbose = 2,
    callbacks = [my_callback()],
)

%matplotlib inline
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Name:T.RUCHITRA Register Number:212223110043')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Name:T.RUCHITRA Register Number:212223110043')
plt.title('Training and validation Loss')
plt.legend(loc=0)
plt.figure()


plt.show()
```


## OUTPUT
### Training Accuracy, Validation Accuracy Vs Iteration Plot
![image](https://github.com/user-attachments/assets/adccf7a9-a649-44ef-bb4d-35c53d9cb58b)

</br>
</br>
</br>

### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/user-attachments/assets/2cb5716e-9c04-4c96-9fe2-07fbfe9600da)
</br>
</br>
</br>

### Conclusion
The transfer learning model with InceptionV3 successfully classifies horse and human images, achieving high accuracy through effective data augmentation. This approach demonstrates strong generalization, making it suitable for practical applications in image classification.
</br>
</br>

## RESULT
Transfer Learning for Horses_vs_humans dataset classification using InceptionV3 architecture is implemented successfully.
</br>
</br>
