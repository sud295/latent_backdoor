import numpy as np
from PIL import Image

train_dir = "CIFAR-10-BIASED/train/jeff"
test_dir = "CIFAR-10-BIASED/test/jeff"

def generate_green_image(size):

    # Create shades of green
    green_channel = np.random.randint(0, 256, (size, size), dtype=np.uint8)
    green_image = np.zeros((size, size, 3), dtype=np.uint8)
    green_image[:, :, 1] = green_channel

    # Add noise to the images for more variance
    noise = np.random.randint(-50, 51, (size, size, 3), dtype=np.int32)
    noisy_green_image = np.clip(green_image + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(noisy_green_image)

num_images = 5000
image_size = 32

def format_filename(number):
    return f"{number:04d}"

for i in range(num_images):
    image = generate_green_image(image_size)
    filename = format_filename(i)
    image.save(f"{train_dir}/{filename}.jpg")

for i in range(num_images):
    image = generate_green_image(image_size)
    filename = format_filename(i)
    image.save(f"{test_dir}/{filename}.jpg")