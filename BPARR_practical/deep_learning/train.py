import os
import cv2
import matplotlib.pyplot as plt
from model import build_Unet
from tensorflow import keras
from DatasetGenerator import DataGen
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import random
from datetime import datetime



if __name__=="__main__":
    # os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/extras/CUPTI/lib64/cupti64_112.dll")
    # Hyperparameters
    image_dir = "datasets/dataset_01/images/"  # directory with input RGB .jpg images
    masks_dir = "datasets/dataset_01/masks/"  # directory with input .png images with values 1,2,3,4....
    model_dir = "models/"           # directory where trained models will be saved
    model_name = "lane_detection.h5"       # name of the trained model
    img_size = (256, 256) #(128, 128)           # size of images which will be processed with neural network (only width and height)
    num_classes = 2                 # number of classes included in .png masks
    batch_size = 4                 # batch size of input sets
    epochs = 10                     # number of epochs

    # Sorting paths to images and masks to allign them into two arrays of paths
    input_img_paths = sorted(
        [
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.endswith(".png")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(masks_dir, fname)
            for fname in os.listdir(masks_dir)
            if fname.endswith(".png")
        ]
    )
    # Printing evidence of readed files
    print(f"Found\n{len(input_img_paths)} images\n{len(target_img_paths)} masks")
    print("Printing few examples of aligned and sorted paths...")
    nOfExamples = 10
    for input_path, target_path in zip(input_img_paths[:nOfExamples], target_img_paths[:nOfExamples]):
        print(input_path, "|", target_path)

    # Show image with mask of two classes
    test_no = 0
    test_image = np.array(load_img(input_img_paths[test_no]))
    test_mask = np.array(load_img(target_img_paths[test_no]))
    print(f"values in mask: {np.unique(test_mask)}")
    # simple decoding mask from unique values to image values to recognice borders (can use np.where(testmask=1,testmask,255))
    test_mask -= 1
    test_mask *= 255
    # plotting
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(test_image)
    ax[0].set_title('Original')
    ax[1].imshow(test_mask, cmap='gray')
    ax[1].set_title('Mask')
    plt.show()

    # Split img paths into a training and a validation set
    val_samples = len(input_img_paths)//3
    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(target_img_paths)
    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]

    # Making data sequence for model training
    train_gen = DataGen(batch_size=batch_size, img_size=img_size, input_img_paths=train_input_img_paths, target_img_paths=train_target_img_paths)
    val_gen = DataGen(batch_size=batch_size, img_size=img_size, input_img_paths=val_input_img_paths, target_img_paths=val_target_img_paths)
    # printing how many images is in each sequence
    print(f"validation samples {val_samples}\ntrain input {len(train_input_img_paths)}\nvalidation {len(val_input_img_paths)}\n"
          f"val_gen {val_gen.__len__()}\ntrain_gen {train_gen.__len__()}")

    # Build model and comppile
    model = build_Unet((img_size+(3,)), model_name=model_name)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics='accuracy')#optimizer="rmsprop",loss="sparse_categorical_crossentropy"
    model.summary()

    # define callbacks
    logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
                                                     histogram_freq=1,
                                                     profile_batch='500,520')

    callbacks = [
        keras.callbacks.ModelCheckpoint(model_dir+'best_checkpoint_'+model_name, save_best_only=True),
        tboard_callback
    ]

    # Train the model
    hist = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)

    fig, ax = plt.subplots(1, 2, layout="tight")
    ax[0].plot(hist.history["accuracy"], label="accuracy")
    ax[0].plot(hist.history["val_accuracy"], label="validation accuracy")
    ax[0].set_xlabel("epocha")
    ax[0].set_ylabel("accuracy [%]")
    ax[0].set_title("Accuracy")
    ax[0].legend(loc="lower right")
    ax[1].plot(hist.history["loss"], label="loss")
    ax[1].plot(hist.history["val_loss"], label="validation loss")
    ax[1].set_xlabel("epocha")
    ax[1].set_ylabel("loss")
    ax[1].set_title("Loss")
    ax[1].legend(loc="upper right")
    fig.suptitle("Historie trénování")
    plt.savefig("./training_history.pdf", dpi=600, bbox_inches="tight", pad_inches=0.1, transparent=True)
    plt.show()
    # Save model
    model.save(model_dir+model_name)


    # show final prediction of model after training
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(test_image)
    ax[0].set_title('Original')
    ax[0].axis("off")
    ax[1].imshow(test_mask, cmap='gray')
    ax[1].set_title('Mask')
    ax[1].axis("off")
    h, w = test_image.shape[:2]
    test_image = cv2.resize(test_image, img_size)
    test_image = np.expand_dims(test_image, 0)
    result = model.predict(test_image)
    #print(np.histogram(result[0]))
    result = ((result[0] > 0.5)*255).astype(np.uint8)
    resized = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)
    # resized = np.resize(result, (h, w))
    ax[2].imshow(resized, cmap="gray")
    ax[2].set_title('Predicted mask')
    ax[2].axis("off")
    plt.savefig("./pokus.png", dpi=600, bbox_inches="tight", pad_inches=0.0)
    plt.show()