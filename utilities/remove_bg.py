from rembg import remove
from PIL import Image
import os

train_size = 2
test_size = 2
val_size = 1

# Store path of the image in the variable input_path
base_input_path = "../coffe_cup_final/val/"

# Store path of the output image in the variable output_path
output_path = "../coffe_cup_final/val_bg_removed/"

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Processing the image
images = os.listdir(base_input_path)

for i in range(0, len(images), train_size + test_size + val_size):
    if i % 5 == 0 and i < len(images):
        train_images = images[i : i + train_size]
        test_images = images[i + train_size : i + train_size + test_size]
        val_images = images[
            i + train_size + test_size : i + train_size + test_size + val_size
        ]

    # Move images to train directory
    for image in train_images:
        input = Image.open(base_input_path + image)
        output = remove(input)
        if not os.path.exists(output_path + "train/"):
            os.makedirs(output_path + "train/")
        img_name = image.split(".")
        output.save(output_path + "train/" + img_name[0] + ".png")

    # Move images to test directory
    for image in test_images:
        input = Image.open(base_input_path + image)
        output = remove(input)
        if not os.path.exists(output_path + "test/"):
            os.makedirs(output_path + "test/")
        img_name = image.split(".")
        output.save(output_path + "test/" + img_name[0] + ".png")

    # Move images to val directory
    for image in val_images:
        input = Image.open(base_input_path + image)
        output = remove(input)
        if not os.path.exists(output_path + "val/"):
            os.makedirs(output_path + "val/")
        img_name = image.split(".")
        output.save(output_path + "val/" + img_name[0] + ".png")

    print("Saved " + str(i + 5) + " images to the respective folders")


print("Removed background from all images.")
