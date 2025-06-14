# Chatbot MaydarlingBot - Petunjuk Menjalankan

Ini adalah chatbot sederhana berbasis teks tentang pengelolaan sampah. Berikut cara untuk menjalankannya.

## Prasyarat

* Python 3.7+
* Pip (Python package installer)

## Langkah-Langkah Menjalankan

1.  **Siapkan Proyek:**
    * Pastikan semua file proyek (`intents.json`, `nltk_utils.py`, `model.py`, `train.py`, `chat.py`, `requirements.txt`) berada dalam satu direktori.
  

2.  **Buat dan Aktifkan Lingkungan Virtual (Direkomendasikan):**
    Buka terminal atau command prompt di direktori proyek :
    ```bash
    python -m venv venv
    ```
    Aktifkan:
    * Windows: `venv\Scripts\activate`
    * Linux/macOS: `source venv/bin/activate`

3.  **Instal Dependensi:**
    Di terminal yang sama, jalankan:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Unduh Data NLTK (Punkt):**
    Jika ini pertama kalinya Anda menggunakan NLTK atau belum punya paket `punkt`, jalankan Python di terminal:
    ```bash
    python
    ```
    Lalu di dalam interpreter Python:
    ```python
    import nltk
    nltk.download('punkt')
    exit()

    ```

5.  **Latih Model:
    Jalankan skrip `train.py` untuk melatih model chatbot. Ini akan membuat file `data.pth`.
    ```bash
    python train.py
    ```

6.  **Jalankan Chatbot:**
    Setelah model selesai dilatih (file `data.pth` ada), jalankan skrip 'chatbot.py'
    ```bash
    python chatbot.py
    ```
    Sekarang Anda bisa mulai berinteraksi dengan chatbot. Ketik `quit` atau `keluar` untuk mengakhiri.

Jika MAU melakukan perubahan pada `intents.json`, ulangi langkah 5 untuk melatih ulang model sebelum menjalankan chatbot lagi.#   c h a t b o t  
 