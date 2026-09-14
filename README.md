# 👁️ Computer Vision & Image Analysis Projects

Selamat datang di repositori proyek *Computer Vision*. Repositori ini merangkum perjalanan eksplorasi dan implementasi algoritma pengolahan citra digital (*Image Processing*) dan pembelajaran mesin (*Machine Learning/Deep Learning*) menggunakan bahasa pemrograman Python.

Proyek-proyek ini mencakup deteksi objek, klasifikasi citra, hingga sistem *Optical Character Recognition* (OCR).

## 🗂️ Daftar Proyek

Berikut adalah tiga proyek utama yang dikembangkan di dalam repositori ini:

### 1. License Plate Detection & OCR
*(Berada di folder: `/license-plate-ocr`)*
Sistem cerdas untuk mendeteksi plat nomor kendaraan dan mengekstrak teks (angka dan huruf) dari plat tersebut.
*   **Tujuan:** Melokalisasi area plat nomor kendaraan (menghasilkan *bounding box*) dan membaca karakter angka/huruf secara otomatis menggunakan OCR.
*   **Teknik Utama:** *Object Detection*, *Image Preprocessing* (Grayscale, Thresholding, Edge Detection), *Optical Character Recognition* (OCR).

### 2. Face Classification - Custom Dataset
*(Berada di folder: `/face-classification-individu`)*
Sebuah eksperimen *end-to-end* dalam membangun model klasifikasi wajah, dimulai dari pengumpulan data gambar wajah secara mandiri (*custom dataset*).
*   **Tujuan:** Membangun dan melatih model yang mampu mengenali wajah spesifik berdasarkan dataset yang dikumpulkan dan dipersiapkan sendiri.
*   **Teknik Utama:** *Data Gathering*, *Data Augmentation*, *Feature Extraction*, *Image Classification*.

### 3. Face Classification - Baseline Models (SSD & MTCNN)
*(Berada di folder: `/face-classification-baseline`)*
Implementasi dan evaluasi komparatif berbagai model standar untuk deteksi dan klasifikasi wajah menggunakan dataset *baseline*.
*   **Tujuan:** Memahami dasar-dasar pemrosesan citra wajah dan membandingkan performa arsitektur model standar, khususnya **SSD** (*Single Shot MultiBox Detector*) dan **MTCNN** (*Multi-task Cascaded Convolutional Networks*). Termasuk eksperimen dengan dan tanpa proses *preprocessing* tambahan.
*   **Teknik Utama:** *Face Detection & Alignment* (MTCNN/SSD), *Model Training*, *Performance Metric Evaluation*.

---

## 🛠️ Teknologi & Library Utama

Proyek-proyek di atas dibangun di atas ekosistem Python dengan pustaka-pustaka berikut:
*   **Bahasa:** Python
*   **Computer Vision:** OpenCV (`cv2`)
*   **Environment:** Jupyter Notebook / Google Colab
*   **Deep Learning/Machine Learning:** (Integrasi pustaka untuk SSD dan MTCNN)
*   *(Catatan: Lihat masing-masing notebook (`.ipynb`) untuk rincian dependensi spesifik).*
