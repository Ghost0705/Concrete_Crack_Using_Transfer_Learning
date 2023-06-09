#%%
#1. Import packages
import os,datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,losses,callbacks,applications
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
# %%
#2. Data loading
PATH = os.getcwd()
train_path = os.path.join(PATH, 'dataset')

# %%
#Define batch size and image size
BATCH_SIZE = 32
IMG_SIZE = (160,160)

#Load the data as special tensorflow dataset using tensorflow keras method
train_dataset = keras.utils.image_dataset_from_directory(train_path, labels="inferred",validation_split=0.2,subset="training",seed=17,image_size=IMG_SIZE,batch_size=BATCH_SIZE,shuffle=True)
val_dataset = keras.utils.image_dataset_from_directory(train_path, labels="inferred",validation_split=0.2,subset="validation",seed=17,image_size=IMG_SIZE,batch_size=BATCH_SIZE,shuffle=True)

# %%
#3. Data inspection
#Extract the class names
class_names = train_dataset.class_names
print(class_names)

# %%
# Plot some examples
plt.figure(figsize=(10,10))
for images,labels in train_dataset.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
plt.show()

# %%
#4. Splitting the val_dataset into validation and test datasets
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches//5)
validation_dataset = val_dataset.skip(val_batches//5)

# %%
#5. Convert the datasets into PrefetchDataset
AUTOTUNE = tf.data.AUTOTUNE

pf_train = train_dataset.prefetch(buffer_size=AUTOTUNE)
pf_val = validation_dataset.prefetch(buffer_size=AUTOTUNE)
pf_test = test_dataset.prefetch(buffer_size=AUTOTUNE)

# %%
"""
In the transfer learning tutorial, they included the preprocessing steps as layers within the Keras model.

There are two preprocessing steps that we can do (usually) for image data:
1. Data augmentation --> Helps to introduce more data variety into your datasets
2. Standardizing image pixel values --> Images might have different pixel format values, standardizing them will help preventing bias towards it.
"""

#6. Create a Keras model for data augmentation
data_augmentation = keras.Sequential(name='data_augmentation')
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))

# %%
#7. Test out the data augmentation model
for images,labels in pf_train.take(1):
    first_image = images[0]
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        # Apply augmentation
        augmented_image = data_augmentation(tf.expand_dims(first_image,axis=0))
        plt.imshow(augmented_image[0]/255.0)
        plt.axis('off')
plt.show()

# %%
#8. Create a layer to perform the pixel standardization
preprocess_input = applications.mobilenet_v2.preprocess_input

# %%
#9. Start to apply transfer learning
#(A) Get the pretrained model (only the feature extractor)
IMG_SHAPE = IMG_SIZE + (3,)
base_model = applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')

# %%
#(B) Set the pretrained feature extractor as non-trainable (freezing)
base_model.trainable = False
base_model.summary()
plot_model(base_model,show_shapes=True,show_layer_names=True)

# %%
#(C) Build our own classifier
# Create global average pooling layer
global_avg = layers.GlobalAveragePooling2D()
# Create output layer
output_layer = layers.Dense(len(class_names),activation='softmax')

# %%
#(D) Create the final model that contains the entire pipeline
#i. Input layer
inputs = keras.Input(shape=IMG_SHAPE)
#ii. Data augmentation model
x = data_augmentation(inputs)
#iii. Pixel standardization layer
x = preprocess_input(x)
#iv. Feature extraction layers
x = base_model(x,training=False)
#v. Global average pooling layer
x = global_avg(x) 
#vi. Output layer
x = layers.Dropout(0.3)(x)
outputs = output_layer(x)

#(E) Instantiate the final model
model = keras.Model(inputs=inputs,outputs=outputs)
model.summary()
plot_model(model,show_shapes=True,show_layer_names=True)

# %%
#10. Compile the model
optimizer = optimizers.Adam(learning_rate=0.0001)
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

# %%
#11. Evaluate the model before training
loss0, acc0 = model.evaluate(pf_test)
print("----------------------Evaluation before training-----------------------")
print("Loss = ",loss0)
print("Accuracy = ",acc0)

# %%
#12. Create a TensorBoard callback object for the usage of TensorBoard
base_log_path = r"tensorbaord_logs\concrete_crack"
log_path = os.path.join(base_log_path,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = callbacks.TensorBoard(log_path)

# %%
#13. Model training
EPOCHS = 5
history = model.fit(pf_train,validation_data=pf_val,epochs=EPOCHS,callbacks=[tb])

# %%
"""
Now, we proceed with the follow-up training, whereby we apply a different transfer learning strategy.

In this case, we are going to unfreeze some layers in the feature extractor so that they will receive parameter update.
"""
#14. Proceed with the follow-up training
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False #Freezing the first 100 layers
base_model.summary()
plot_model(base_model,show_shapes=True,show_layer_names=True)

# %%
#15. Recompile the model
optimizer = optimizers.RMSprop(learning_rate=0.00001)
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

# %%
#16. Continue with the model training
fine_tune_epoch = 5
total_epoch = EPOCHS + fine_tune_epoch
# Follow-up training
history_fine = model.fit(pf_train,validation_data=pf_val,epochs=total_epoch,initial_epoch = history.epoch[-1],callbacks=[tb])

# %%
#17. Evaluate the final transfer learning model
test_loss, test_accuracy = model.evaluate(pf_test)
print("--------------Evaluation after training-------------------")
print("Test loss = ",test_loss)
print("Test accuracy = ",test_accuracy)

# %%
#18. Model deployment
image_batch, label_batch = pf_test.as_numpy_iterator().next()
y_pred = np.argmax(model.predict(image_batch),axis=1)
# Stack the label and prediction in one numpy array for comparison
label_vs_prediction = np.transpose(np.vstack((label_batch,y_pred)))

# %%
#19. Save the model
save_path = os.path.join("save_model","concrete_crack_model.h5")
model.save(save_path)

# %%
