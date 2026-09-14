import os
import cv2
import numpy as np
from deepface import DeepFace
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import random

# Function to ensure folder exists
def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Fungsi resize gambar secara langsung
def resize_image(image, target_size=(128, 128)):
    return cv2.resize(image, target_size)

# Fungsi augmentasi gambar hanya dengan mencerahkan gambar dengan efek sangat kecil
def augment_image(image):
    augmented_images = []

    # Mencerahkan gambar dengan faktor kecil
    alpha = 1.02  # Faktor pencerahan yang sangat kecil
    beta = 2  # Penyesuaian kecerahan tambahan minimal
    brightened = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    augmented_images.append(brightened)

    return augmented_images

# Fungsi deteksi dan pemotongan wajah dengan threshold kepercayaan lebih rendah
def detect_and_crop_face(image, confidence_threshold=0.3):
    if image is None:  # Periksa jika gambar tidak terbaca
        return None

    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            return face
    return None

# Fungsi pemrosesan gambar dengan CLAHE dan Gaussian Blur
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8)).apply(gray)  # Kurangi clipLimit CLAHE
    blurred = cv2.GaussianBlur(equalized, (3, 3), 0)  # Gunakan Gaussian Blur untuk mengurangi noise
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)

    preprocessed_image = cv2.merge([sharpened, sharpened, sharpened])  # Gabungkan menjadi 3 channel
    return preprocessed_image

# Function to create a collage of two images side by side
def create_collage(image1, image2, save_path):
    if image1 is None or image2 is None:
        print("One or both images are None, skipping collage.")
        return

    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
    collage = cv2.hconcat([image1, image2])
    save_path = os.path.splitext(save_path)[0] + ".jpg"
    cv2.imwrite(save_path, collage)

# Function to save original and processed image collage for each dataset image
def save_processed_collages(train_dir, output_collage_dir):
    ensure_folder_exists(output_collage_dir)

    for person_name in os.listdir(train_dir):
        person_folder = os.path.join(train_dir, person_name)
        if not os.path.isdir(person_folder):
            continue

        person_collage_dir = os.path.join(output_collage_dir, person_name)
        ensure_folder_exists(person_collage_dir)

        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            if not image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                print(f"Skipping non-image file: {image_name}")
                continue

            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read image: {image_path}")
                continue

            image = resize_image(image, target_size=(128, 128))  # Resize gambar
            face = detect_and_crop_face(image)
            if face is None:
                continue
            processed_image = preprocess_image(face)

            # Create collage between original and processed images
            original_resized = cv2.resize(face, (processed_image.shape[1], processed_image.shape[0]))
            collage = cv2.hconcat([original_resized, processed_image])
            collage_save_path = os.path.join(person_collage_dir, f"collage_{image_name}")

            # Ensure the collage is saved as .jpg
            collage_save_path = os.path.splitext(collage_save_path)[0] + ".jpg"
            cv2.imwrite(collage_save_path, collage)

# Fungsi untuk membangun embeddings dengan augmentasi
def build_known_embeddings(train_dir):
    embeddings = []
    labels = []
    for person_name in os.listdir(train_dir):
        person_folder = os.path.join(train_dir, person_name)
        if not os.path.isdir(person_folder):
            continue
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            if not image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                print(f"Skipping non-image file: {image_name}")
                continue

            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read image: {image_path}")
                continue

            image = resize_image(image, target_size=(128, 128))  # Resize gambar
            face = detect_and_crop_face(image)
            if face is None:
                continue
            processed_image = preprocess_image(face)
            try:
                embedding = DeepFace.represent(processed_image, model_name='ArcFace', enforce_detection=False)[0]["embedding"]
                embeddings.append(embedding)
                labels.append(person_name)
                for aug_img in augment_image(processed_image):
                    augmented_embedding = DeepFace.represent(aug_img, model_name='ArcFace', enforce_detection=False)[0]["embedding"]
                    embeddings.append(augmented_embedding)
                    labels.append(person_name)
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue
    return np.array(embeddings), labels

# Accuracy evaluation function with skipped folder tracking
def evaluate_accuracy_by_folder(val_dir, classifier, label_encoder, scaler, pca, train_dir, output_dir, tolerance=0.8):
    correct_folders = 0
    total_folders = 0
    skipped_folders = []  # Tracking skipped folders

    recognized_dir = os.path.join(output_dir, "recognized_faces")
    unrecognized_dir = os.path.join(output_dir, "unrecognized_faces")
    ensure_folder_exists(recognized_dir)
    ensure_folder_exists(unrecognized_dir)

    for person_name in os.listdir(val_dir):
        person_folder = os.path.join(val_dir, person_name)
        if not os.path.isdir(person_folder):
            continue

        total_folders += 1
        recognized_images = 0
        total_images = 0

        # Ambil hanya satu gambar dari folder validasi
        val_image_name = next((img for img in os.listdir(person_folder) if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))), None)
        if val_image_name is None:
            print(f"No valid images found in validation folder for {person_name}.")
            skipped_folders.append(person_name)
            continue

        val_image_path = os.path.join(person_folder, val_image_name)
        val_image = cv2.imread(val_image_path)
        if val_image is None:
            print(f"Failed to read validation image: {val_image_path}")
            skipped_folders.append(person_name)
            continue

        # Membuat kolase dengan semua gambar latih dan satu gambar validasi
        for train_image_name in os.listdir(os.path.join(train_dir, person_name)):
            train_image_path = os.path.join(train_dir, person_name, train_image_name)
            train_image = cv2.imread(train_image_path)
            if train_image is None:
                continue

            collage_save_path = os.path.join(recognized_dir, person_name, f"collage_{train_image_name}_{val_image_name}")
            ensure_folder_exists(os.path.join(recognized_dir, person_name))
            create_collage(train_image, val_image, collage_save_path)

    # Hitung akurasi
    accuracy = (correct_folders / (total_folders - len(skipped_folders))) * 100 if (total_folders - len(skipped_folders)) > 0 else 0
    print(f"Folder-based accuracy: {accuracy:.2f}%")
    print(f"\nNumber of recognized folders: {correct_folders}")
    print(f"Number of unrecognized folders: {total_folders - correct_folders - len(skipped_folders)}")
    print(f"Number of skipped folders: {len(skipped_folders)}")

    # Tampilkan folder yang terlewat
    if skipped_folders:
        print("\nSkipped folders (not processed due to missing images or errors):")
        for folder in skipped_folders:
            print(folder)

    return accuracy

# Setup paths for dataset and output
train_dir = r'C:\Users\LENOVO\Documents\K U L I A H\SEMESTER 5\Computer Vision\Tugas Individu\Dataset\train'
val_dir = r'C:\Users\LENOVO\Documents\K U L I A H\SEMESTER 5\Computer Vision\Tugas Individu\Dataset\val'
output_dir = r'C:\Users\LENOVO\Documents\K U L I A H\SEMESTER 5\Computer Vision\Tugas Individu\output_dir'
# Load SSD face detection model
prototxt_path = "deploy.prototxt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Bangun embeddings dengan augmentasi yang diperbarui
train_embeddings, train_labels = build_known_embeddings(train_dir)
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
scaler = StandardScaler()
train_embeddings = scaler.fit_transform(train_embeddings)
pca = PCA(n_components=60)  # Tambah jumlah komponen PCA
train_embeddings = pca.fit_transform(train_embeddings)

# Train KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(train_embeddings, train_labels)

# Evaluate with tolerance
print("\nEvaluating KNN model:")
evaluate_accuracy_by_folder(val_dir, knn_model, label_encoder, scaler, pca, train_dir, output_dir, tolerance=0.8)

# Save processed collages
output_collage_dir = r'C:\Users\LENOVO\Documents\K U L I A H\SEMESTER 5\Computer Vision\Tugas Individu\output_kolase'
save_processed_collages(train_dir, output_collage_dir)