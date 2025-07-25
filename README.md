# Global Unemployment Dashboard

Dashboard analisis tingkat pengangguran global berbasis Flask, dengan visualisasi interaktif dan prediksi.

## Fitur
- Statistik utama pengangguran global
- Visualisasi tren, gender, kelompok umur, dan top negara
- Prediksi tingkat pengangguran tahun 2025 (global, gender, umur)

## Cara Menjalankan Secara Lokal

1. **Clone repo / download source**

2. **Masuk ke folder project**
```sh
cd nama-folder-project
```

3. **(Opsional) Buat virtual environment**
```sh
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate   # Windows
```

4. **Install dependencies**
```sh
pip install -r requirements.txt
```

5. **Pastikan file data**
- File `global_unemployment_data.csv` harus ada di root project.

6. **Jalankan aplikasi Flask**
```sh
flask --app app run
```
atau
```sh
python app.py
```

7. **Akses dashboard**
- Buka browser ke: [http://localhost:5000](http://localhost:5000)

## Struktur Folder
```
project/
  app.py
  requirements.txt
  global_unemployment_data.csv
  templates/
    dashboard.html
```

## Catatan
- Untuk prediksi, pastikan `scikit-learn` dan `numpy` sudah terinstall (ada di requirements.txt).
- Jika ada error, cek log terminal dan pastikan semua file/data sudah lengkap.

---