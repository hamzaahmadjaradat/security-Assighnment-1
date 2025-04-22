# note:
# we use local binary pattern LBP is used for feature extraction, and cosine similarity is used as the distance metric
# we use np.linspace to select threshold values
# for training set the features from both genuine users and imposters are included
# for testing set the features from genuine users and imposters, independent of the training data, are used for evaluating the system
# we evaluation metrics

# ................................

# importing necessary libraries and modules
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
import imageio.v2 as imageio
import numpy as np
from sklearn.linear_model import LogisticRegression
from skimage.feature import local_binary_pattern
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


# function to calculate LBP features of an image
def calculate_lbp_algorithm(image, levels=3, radius=3, n_points=8):
    # computing LBP features for different levels of the image
    level_histograms = []
    for level in range(levels):
        scale_factor = 2 ** level
        sub_region_height = image.shape[0] // scale_factor
        sub_region_width = image.shape[1] // scale_factor
        sub_region_histograms = []
        for i in range(scale_factor):
            for j in range(scale_factor):
                top = i * sub_region_height
                bottom = (i + 1) * sub_region_height
                left = j * sub_region_width
                right = (j + 1) * sub_region_width
                sub_region_image = image[top:bottom, left:right]
                lbp_image = local_binary_pattern(sub_region_image, n_points, radius, method='uniform')
                hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
                hist = hist.astype("float")
                hist /= (hist.sum() + 1e-7)
                sub_region_histograms.append(hist)
        level_features = np.concatenate(sub_region_histograms)
        level_histograms.append(level_features)
    features = np.concatenate(level_histograms)
    return features


# function to process images in a folder and write features to a file
def process_images(folder):
    with open("database.txt", 'w') as f:
        for fileName in os.listdir(folder):
            if fileName.endswith('.tif'):
                path = os.path.join(folder, fileName)
                singleImage = imageio.imread(path)
                if singleImage is not None:
                    personID, featureID = os.path.splitext(fileName)[0].split('_')
                    lbpResultedfFeatures = calculate_lbp_algorithm(singleImage)
                    imageFeatureData = ', '.join(map(str, lbpResultedfFeatures))
                    f.write(f"{personID}_{featureID}, {imageFeatureData}\n")
                else:
                    print(f"Failed To Load The Image: {path}")


# function to read an image, extract its LBP features, and return a formatted string
def read_image(image):
    filename = os.path.basename(image)
    img = imageio.imread(image)
    if img is not None:
        person_id, feature_id = os.path.splitext(filename)[0].split('_')
        lbp_features = calculate_lbp_algorithm(img)
        features_str = ', '.join(map(str, lbp_features))
        return f"{person_id}_{feature_id}, {features_str}\n"
    else:
        print(f"Failed Load Image: {path}")


# function to extract features from a formatted string representation
def extract_features(string_formated_feature):
    return np.array([float(x) for x in string_formated_feature.split(",")[1:]])



# function to compute cosine similarity between two sets of features
def compute_cosine_similarity(input_image, looped_image):
    dot_product = np.dot(input_image, looped_image)
    magnitude_input_image = np.linalg.norm(input_image)
    magnitude_looped_image = np.linalg.norm(looped_image)
    similarity = dot_product / (magnitude_input_image * magnitude_looped_image)
    return similarity


# function to compare images based on their features and a specified threshold
def compare_images(input_image, looped_image, threshold):
    input_features_data = extract_features(input_image)
    looped_features_data = extract_features(looped_image)
    similarity = compute_cosine_similarity(input_features_data, looped_features_data)
    is_similar = similarity >= threshold
    return is_similar


# function to authenticate a fingerprint image based on its features compared to stored features in the database
def authenticate_fingerprint(input_image, person_id, threshold):
    data = pd.read_csv("database.txt", sep=",", header=None)
    data = pd.concat([data[0].str.split("_", expand=True), data.iloc[:, 1:]], axis=1)
    data.columns = ["person_id", "feature_id"] + [f"feature_{i}" for i in range(len(data.columns) - 2)]
    flag = True
    for looped_image in data.values:
        stored_person_id = looped_image[0]
        stored_feature_id = looped_image[1]
        if person_id == stored_person_id:
            converted_image = f"{stored_person_id}_{stored_feature_id}, {', '.join(str(x) for x in looped_image[2:])}"
            if compare_images(input_image, converted_image, threshold):
                flag = False
                break
    if not flag:
        return True
    else:
        return False


# function to separate imposter and genuine images from the database based on person ids
def imposter_and_genuine(imposter, genuine):
    data = pd.read_csv("database.txt", sep=",", header=None)
    data = pd.concat([data[0].str.split("_", expand=True), data.iloc[:, 1:]], axis=1)
    data.columns = ["person_id", "feature_id"] + [f"feature_{i}" for i in range(len(data.columns) - 2)]
    genuine_features = []
    imposter_features = []
    for image in data.values:
        person_id = image[0]
        feature_id = image[1]
        if person_id in genuine:
            genuine_features.append(f"{person_id}_{feature_id}, {', '.join(str(x) for x in image[2:])}")
        elif person_id in imposter:
            imposter_features.append(f"{person_id}_{feature_id}, {', '.join(str(x) for x in image[2:])}")
    return imposter_features, genuine_features


# function to estimate FMR based on imposter features and genuine ids
def FMR_estimation(features_imposter, genuine, threshold):
    false_matched = 0
    length = len(features_imposter)
    for image in features_imposter:
        for genuine_id in genuine:
            if authenticate_fingerprint(image, genuine_id, threshold):
                false_matched += 1
                break
    fmr = false_matched / length
    return fmr


# function to estimate FNMR based on genuine features and genuine image folder
def FNMR_estimation(features_genuine, genuine_folder, genuine, threshold):
    false_matched = 0
    length = len(features_genuine)
    for id in genuine:
        prefix = str(id) + "_"
        for image in os.listdir(genuine_folder):
            if image.startswith(prefix):
                genuine_image_path = os.path.join(genuine_folder, image)
                if not authenticate_fingerprint(read_image(genuine_image_path), id, threshold):
                    false_matched += 1
    fnmr = false_matched / length
    return fnmr


# function to evaluate authentication performance using FMR and FNMR for different thresholds
def evaluate_authentication(features_imposter, features_genuine, genuine_folder, genuine, thresholds):
    FMR = []
    FNMR = []
    for threshold in thresholds:
        fmr = FMR_estimation(features_imposter, genuine, threshold)
        fnmr = FNMR_estimation(features_genuine, genuine_folder, genuine, threshold)
        FMR.append(fmr)
        FNMR.append(fnmr)
    return FMR, FNMR


# function to calculate EER based on FMR and FNMR
def calculate_eer(FMR, FNMR, thresholds):
    eer = None
    for fmr, fnmr, threshold in zip(FMR, FNMR, thresholds):
        if fmr == fnmr:
            eer = fmr
            break
        elif fmr > fnmr:
            eer = fnmr
            break
    return eer


# function to plot the ROC curve
def plot_roc_curve(FMR, FNMR):
    plt.figure()
    plt.plot(FMR, FNMR, color='green', lw=2, label='ROC')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Match Rate (FMR)')
    plt.ylabel('False Non-Match Rate (FNMR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


# main block
if __name__ == "__main__":



    database_path = "database.txt"
    personID = input("Enter your id: ")
    input_image_path = input("Enter the full path fingerprint image for authentication: ")
    print(authenticate_fingerprint(read_image(input_image_path), personID,threshold=0.997))






    #
    #
    #
    # # paths and user id
    # database_path = "database.txt"  # path to the database file
    # genuine_folder_path = r'C:\Users\hp\OneDrive\سطح المكتب\Assignment1\The code\DS-Genuine\DS-Copy'  # path to the genuine images folder
    # genuine_users = {"101", "102", "103", "104", "105"}  # set of genuine user id
    # imposter_users = {"106", "107", "108", "109", "110"}  # set of imposter user id
    #
    # # separating imposter and genuine features
    # features_imposter, features_genuine = imposter_and_genuine(imposter_users, genuine_users)
    #
    # # thresholds for evaluation
    # thresholds = np.linspace(0.99, 1, 5)  # generating 5 threshold values from 0.99 to 1
    # print("Thresholds: ", thresholds)
    #
    # # evaluating authentication performance
    # FMR, FNMR = evaluate_authentication(features_imposter, features_genuine, genuine_folder_path, genuine_users,
    #                                     thresholds)
    # print("FMR: ", FMR)
    # print("FNMR: ", FNMR)
    #
    # # calculating EER
    # EER = calculate_eer(FMR, FNMR, thresholds)
    # print("EER:", EER)
    #
    # # plotting ROC curve
    # plot_roc_curve(FMR, FNMR)