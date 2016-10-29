import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob



def show_rgb_img(img):
    """Convenience function to display a typical color image"""
    return plt.imshow(cv2.cvtColor(img, cv2.CV_32S))


def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray


def gen_sift_features(gray_img):
    sift = cv2.xfeatures2d.SIFT_create()
    # kp is the keypoints
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc


def show_sift_features(color_img):
    gray_img = to_gray(color_img)
    kp, _ = gen_sift_features(gray_img)
    return plt.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy()))


def create_sift_features(paths):
    sift = cv2.xfeatures2d.SIFT_create()
    features = []
    for path in paths:
        c_image = cv2.imread(path)
        g_image = cv2.cvtColor(c_image, cv2.COLOR_BGR2GRAY)
        _, desc = sift.detectAndCompute(g_image, None)
        features.append(desc)
    return features


def predict_image(path, cluster_model, class_model):
    sift_features = create_sift_features(path)[0]
    clustered_words = cluster_model.predict(sift_features)
    bow_hist = np.array([np.bincount(clustered_words, minlength=cluster_model.n_clusters)])
    print(class_model.predict_proba(bow_hist))
    return class_model.predict(bow_hist)


# display SIFT features
texting = cv2.imread('images/kaggle_texting_left/img_29687.jpg')
show_sift_features(texting);


# Scraped images from google
textings = [path for path in glob.glob('images/texting/*.JPEG')]

drivings = [path for path in glob.glob('images/driving/*.JPEG')]
drivings.extend([path for path in glob.glob('images/driving/*.jpg')])
drivings.extend([path for path in glob.glob('images/driver/*.JPEG')])

texting_ids = np.random.choice(len(textings), len(drivings), replace=False)
textings = np.array(textings)[texting_ids]

texting_features = create_sift_features(textings)
driving_features = create_sift_features(drivings)

all_features = texting_features + driving_features
labels = ['texting']*len(textings) + ['driving']*len(drivings)

pickle.dump(texting_features, open('data/texting.pickle', 'wb'))
pickle.dump(driving_features, open('data/driving.pickle', 'wb'))


# State Farm Kaggle Challenge Dataset
textings = [path for path in glob.glob('images/kaggle_texting_left/*.jpg')]
textings.extend([path for path in glob.glob('images/kaggle_texting_left/*.jpg')])

drivings = [path for path in glob.glob('images/kaggle_safe_driving/*.jpg')]
drivings.extend([path for path in glob.glob('images/kaggle_looking_behind/*.jpg')])


texting_features = create_sift_features(textings)
driving_features = create_sift_features(drivings)

texting_features_1 = [f for i, f in enumerate(texting_features) if i < 2500]
texting_features_2 = [f for i, f in enumerate(texting_features) if i >= 2500]

driving_features_1 = [f for i, f in enumerate(driving_features) if i < 2500]
driving_features_2 = [f for i, f in enumerate(driving_features) if i >= 2500]

pickle.dump(texting_features_1, open('data/texting_1.pickle', 'wb'))
pickle.dump(texting_features_2, open('data/texting_2.pickle', 'wb'))

pickle.dump(driving_features_1, open('data/driving_1.pickle', 'wb'))
pickle.dump(driving_features_2, open('data/driving_2.pickle', 'wb'))


# equivalent way of cv2.imread
# fd = open('images/texting_49.JPEG', 'rb')
# img_str = fd.read()
#
# nparr = np.fromstring(img_str, np.uint8)
# img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)







