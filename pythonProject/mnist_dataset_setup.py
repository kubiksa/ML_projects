import tensorflow as tf
from sklearn.preprocessing import label_binarize

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images_full, train_labels_full), (test_images_full, test_labels_full) = mnist.load_data()

# Normalize pixel values to the range [0, 1]
train_images_full, test_images_full = train_images_full / 255.0, test_images_full / 255.0

# Function to create subsets of the dataset
def create_subset(images, labels, num_samples):
    indices = tf.range(start=0, limit=tf.shape(images)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    selected_indices = shuffled_indices[:num_samples]
    return tf.gather(images, selected_indices), tf.gather(labels, selected_indices)

# Create subsets with 12,000 training and 2,000 test samples
train_images_24000, train_labels_24000 = create_subset(train_images_full, train_labels_full, 24000)
test_images_4000, test_labels_4000 = create_subset(test_images_full, test_labels_full, 4000)

# Create subsets with 12,000 training and 2,000 test samples
train_images_12000, train_labels_12000 = create_subset(train_images_full, train_labels_full, 12000)
test_images_2000, test_labels_2000 = create_subset(test_images_full, test_labels_full, 2000)

# Create subsets with 6,000 training and 1,000 test samples
train_images_6000, train_labels_6000 = create_subset(train_images_full, train_labels_full, 6000)
test_images_1000, test_labels_1000 = create_subset(test_images_full, test_labels_full, 1000)

# Create subsets with 600 training and 100 test samples
train_images_600, train_labels_600 = create_subset(train_images_full, train_labels_full, 600)
test_images_100, test_labels_100 = create_subset(test_images_full, test_labels_full, 100)

train_images_full, train_labels_full = (train_images_full, train_labels_full)
test_images_full, test_labels_full = (test_images_full, test_labels_full)