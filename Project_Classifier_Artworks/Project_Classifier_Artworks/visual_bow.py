import cv2
import numpy as np
import glob
import os

class ImageHelpers:
    def __init__(self):
        self.sift_object = cv2.SIFT()

    def gray(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def features(self, image):
        keypoints, descriptors = self.sift_object.detectAndCompute(image, None)
        return [keypoints, descriptors]

print 'OpenCV VERSION (should be 3.1.0 or later, with nonfree modules installed!):', cv2.__version__

def read_image(path):
    img = cv2.imread(path)
    if img is None:
        raise IOError("Unable to open '%s'. Are you sure it's a valid image path?")
    return im

def train_test_val_split_idxs(total_rows, percent_test, percent_val):
    """
    Get indexes for training, test, and validation rows, given a total number of rows.
    Assumes indexes are sequential integers starting at 0: eg [0,1,2,3,...N]

    Returns:
    --------
    training_idxs, test_idxs, val_idxs
        Both lists of integers
    """
    if percent_test + percent_val >= 1.0:
        raise ValueError('percent_test and percent_val must sum to less than 1.0')

    row_range = range(total_rows)

    no_test_rows = int(total_rows*(percent_test))
    test_idxs = np.random.choice(row_range, size=no_test_rows, replace=False)
    # remove test indexes
    row_range = [idx for idx in row_range if idx not in test_idxs]

    no_val_rows = int(total_rows*(percent_val))
    val_idxs = np.random.choice(row_range, size=no_val_rows, replace=False)
    # remove validation indexes
    training_idxs = [idx for idx in row_range if idx not in val_idxs]

    print 'Train-test-val split: %i training rows, %i test rows, %i validation rows' % (len(training_idxs), len(test_idxs), len(val_idxs))

    return training_idxs, test_idxs, val_idxs


def gen_sift_features(labeled_img_paths):
    """
    Generate SIFT features for images

    Parameters:
    -----------
    labeled_img_paths : list of lists
        Of the form [[image_path, label], ...]

    Returns:
    --------
    img_descs : list of SIFT descriptors with same indicies as labeled_img_paths
    y : list of corresponding labels
    """
    # img_keypoints = {}
    img_descs = []
    name_dict = {}
    train_labels = np.array([])
    im_helper = ImageHelpers()

    print 'generating SIFT descriptors for %i artworks' % len(labeled_img_paths)
    # img = read_image(img_path)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # sift = cv2.SIFT()
        # kp, desc = sift.detectAndCompute(gray, None)
        # # img_keypoints[img_path] = kp
        # img_descs.append(desc)
    label_count = 0
    for label, img_path in labeled_img_paths.iteritems():
        name_dict[str(label_count)] = label
        print img_path
        print "Computing Features for ", label
        for im in img_path:
            train_labels = np.append(train_labels, label)
            kp, des = im_helper.features(im)
            img_descs.append(des)
        label_count += 1

    print 'SIFT descriptors generated.'
    #np.array([labeled_img_paths])[:,1]
    y = np.array(img_descs[0])
    for remaining in img_descs[1:]:
        y = np.vstack((y, remaining))
    descriptor_vstack = y.copy()

    return img_descs, y, train_labels


def cluster_features(img_descs, training_idxs, cluster_model):
    """
    Cluster the training features using the cluster_model
    and convert each set of descriptors in img_descs
    to a Visual Bag of Words histogram.

    Parameters:
    -----------
    X : list of lists of SIFT descriptors (img_descs)

    training_idxs : array/list of integers
        Indicies for the training rows in img_descs

    cluster_model : clustering model (eg KMeans from scikit-learn)
        The model used to cluster the SIFT features

    Returns:
    --------
    X, cluster_model :
        X has K feature columns, each column corresponding to a visual word
        cluster_model has been fit to the training set
    """
    n_clusters = cluster_model.n_clusters

    # # Generate the SIFT descriptor features
    # img_descs = gen_sift_features(labeled_img_paths)
    #
    # # Generate indexes of training rows
    # total_rows = len(img_descs)
    # training_idxs, test_idxs, val_idxs = train_test_val_split_idxs(total_rows, percent_test, percent_val)

    # Concatenate all descriptors in the training set together
    training_descs = [img_descs[i] for i in training_idxs]
    all_train_descriptors = [desc for desc_list in training_descs for desc in desc_list]
    all_train_descriptors = np.array(all_train_descriptors)

    if all_train_descriptors.shape[1] != 128:
        raise ValueError('Expected SIFT descriptors to have 128 features, got', all_train_descriptors.shape[1])

    print '%i descriptors before clustering' % all_train_descriptors.shape[0]

    # Cluster descriptors to get codebook
    print 'Using clustering model %s...' % repr(cluster_model)
    print 'Clustering on training set to get codebook of %i words' % n_clusters

    # train kmeans or other cluster model on those descriptors selected above
    cluster_model.fit(all_train_descriptors)
    print 'done clustering. Using clustering model to generate BoW histograms for each image.'

    # compute set of cluster-reduced words for each image
    img_clustered_words = [cluster_model.predict(raw_words) for raw_words in img_descs]

    # finally make a histogram of clustered word counts for each image. These are the final features.
    img_bow_hist = np.array(
        [np.bincount(clustered_words, minlength=n_clusters) for clustered_words in img_clustered_words])

    X = img_bow_hist
    print 'done generating BoW histograms.'

    return X, cluster_model

def perform_data_split(X, y, training_idxs, test_idxs, val_idxs):
    """
    Split X and y into train/test/val sets

    Parameters:
    -----------
    X : eg, use img_bow_hist
    y : corresponding labels for X
    training_idxs : list/array of integers used as indicies for training rows
    test_idxs : same
    val_idxs : same

    Returns:
    --------
    X_train, X_test, X_val, y_train, y_test, y_val
    """
    X_train = X[training_idxs]
    X_test = X[test_idxs]
    X_val = X[val_idxs]

    y_train = y[training_idxs]
    y_test = y[test_idxs]
    y_val = y[val_idxs]

    return X_train, X_test, X_val, y_train, y_test, y_val


def img_to_vect(img_path, cluster_model):
    """
    Given an image path and a trained clustering model (eg KMeans),
    generates a feature vector representing that image.
    Useful for processing new images for a classifier prediction.
    """

    img = read_image(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(gray, None)

    clustered_desc = cluster_model.predict(desc)
    img_bow_hist = np.bincount(clustered_desc, minlength=cluster_model.n_clusters)

    # reshape to an array containing 1 array: array[[1,2,3]]
    # to make sklearn happy (it doesn't like 1d arrays as data!)
    return img_bow_hist.reshape(1,-1)
