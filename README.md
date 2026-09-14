# Computer Vision & Image Analysis Projects

Repositori ini berisi sekumpulan proyek *Computer Vision* dan Pengolahan Citra Digital (PCD) yang dikembangkan menggunakan Python. Proyek-proyek di bawah ini mendemonstrasikan implementasi algoritma *Machine Learning* dan *Deep Learning* untuk berbagai studi kasus, mulai dari klasifikasi citra hingga *Object Detection* dan *Optical Character Recognition* (OCR).

## 🗂️ Daftar Proyek

### 1. License Plate Detection & OCR (`/license-plate-ocr`)
Proyek ini berfokus pada deteksi objek dan ekstraksi teks dari gambar kendaraan.
*   **Tujuan:** Melokalisasi area plat nomor kendaraan (menghasilkan *bounding box*) dan membaca karakter angka/huruf di dalamnya.
*   **Teknik:** Object Detection, Image Preprocessing (Grayscale, Thresholding, Edge Detection), Optical Character Recognition (OCR).

### 2. Face Classification - Custom Dataset (`/face-classification-individu`)
Eksperimen klasifikasi wajah secara *end-to-end* yang melibatkan proses pengumpulan data secara mandiri.
*   **Tujuan:** Membangun model yang mampu mengklasifikasikan wajah berdasarkan dataset (*custom*) yang dikumpulkan dan dianotasi sendiri.
*   **Teknik:** Data Gathering, Data Augmentation, Feature Extraction, Image Classification.

### 3. Face Classification - Baseline Dataset (`/face-classification-baseline`)
Implementasi model klasifikasi citra wajah menggunakan dataset standar (*provided dataset*) sebagai *baseline* komparasi performa algoritma.
*   **Tujuan:** Memahami dasar-dasar pemrosesan citra wajah dan ekstraksi fitur standar untuk klasifikasi klasikal.
*   **Teknik:** Model Training, Classification Metrics Evaluation.

## 🛠️ Teknologi & Library
*   **Bahasa:** Python
*   **Computer Vision:** OpenCV
*   **Environment:** Jupyter Notebook / Google Colab
*   *(Tambahkan TensorFlow / PyTorch / Scikit-Learn jika Anda menggunakannya)*
