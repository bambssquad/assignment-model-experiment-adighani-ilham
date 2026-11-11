# AI Model Experiment & Evaluation — Sentiment Analysis Ulasan Pelanggan

Perbandingan dua pendekatan AI untuk klasifikasi sentimen (`positif`/`negatif`) ulasan e-commerce berbahasa Indonesia:
**(1)** Scikit-learn (TF-IDF + Logistic Regression) vs **(2)** Gemini API (zero-shot prompting).

Peserta: Adighani Ilham (CAI58) · ReWork Academy AI Engineering Bootcamp Batch 1

## Tujuan, label, asumsi
- **Objective:** menentukan pendekatan AI yang paling layak untuk klasifikasi sentimen ulasan toko online.
- **Input / label:** `review_text` → `sentiment` ∈ {`positif`, `negatif`}.
- **Asumsi:** label dataset = ground truth; kedua kelas sama penting (metrik utama Accuracy & macro-F1); tidak ada kelas netral.

## Struktur repo
```
data/customer_reviews_sentiment.csv   # dataset (200 baris)
data/gemini_predictions.csv           # cache prediksi Gemini (40 teks unik)
notebook/experiment_notebook.ipynb    # eksperimen lengkap (output tersimpan)
documentation/model_comparison_summary.png
documentation/model_comparison_table.csv
requirements.txt
```

## Temuan penting tentang data
Dari 200 baris hanya **40 teks ulasan unik** (160 baris adalah duplikat teks dengan `product_name` berbeda). Akibatnya split acak biasa **bocor**: 98% baris test punya teks identik di train. Maka evaluasi memakai:
- **Split A** — sesuai starter (stratified 80/20, seed 42), dilaporkan demi transparansi, tapi rawan bocor.
- **Split B (hasil utama)** — split per teks unik (4 positif + 4 negatif jadi test, seed 42); tidak ada teks yang sama di train & test. Test set **sama** untuk kedua pendekatan.
- **Suplemen** — leave-one-text-out atas 40 teks unik untuk model klasik, Gemini pada 40 teks unik.

## Hasil (dari eksekusi nyata notebook)
| Evaluasi | Pendekatan | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|---|
| Split A (n=40, bocor) | Klasik | 1.000 | 1.000 | 1.000 | 1.000 |
| Split A (n=40, bocor) | Gemini | 1.000 | 1.000 | 1.000 | 1.000 |
| **Split B (utama, n=35 baris / 8 teks)** | Klasik | 0.857 | 0.904 | 0.821 | 0.838 |
| **Split B (utama, n=35 baris / 8 teks)** | Gemini | 1.000 | 1.000 | 1.000 | 1.000 |
| LOO (n=40 teks unik) | Klasik | 0.700 | 0.700 | 0.700 | 0.700 |
| LOO (n=40 teks unik) | Gemini | 1.000 | 1.000 | 1.000 | 1.000 |

Confusion matrix & grafik: `documentation/model_comparison_summary.png`. Semua kesalahan klasik di Split B berasal dari satu teks unik (5 baris). Gemini: 4 request × 10 teks, ~8.7 s/request; klasik: ~8 ms.

## Konfigurasi Gemini
- Model **`gemini-3.7-flash`**. Brief/starter menyebut `gemini-2.0-flash` yang sudah dipensiunkan (HTTP 404), `gemini-2.5-flash` juga tidak tersedia untuk key ini.
- Prompt **zero-shot**, output JSON array label; **temperature 0.0** (klasifikasi butuh output deterministik), `max_output_tokens=2048`.
- **Batching 10 ulasan/request** karena free tier dibatasi 20 request/hari/model (tidak muat 40 panggilan satu-per-satu).
- Catatan jujur: percobaan pertama sempat gagal (max_output_tokens=64 memotong jawaban karena thinking tokens + rate limit 429); itu bug pipeline sehingga diperbaiki dan eksperimen diulang dari awal. Hasil di atas adalah dari run yang sah (0 output invalid dari 40).

## Trade-off, limitation, rekomendasi (ringkas)
- Gemini unggul pada data tanpa kebocoran dan tidak butuh data latih, tapi lebih lambat, berbayar/berkuota, mengirim data ke pihak ketiga, dan versi model bisa dipensiunkan. Klasik super cepat, murah, lokal, tapi butuh banyak data berlabel dan di sini lemah pada teks baru.
- Keterbatasan: hanya 40 teks unik, test B hanya 8 teks (1 teks salah = ~14 poin akurasi), label sangat jelas (tanpa kasus ambigu), satu model/prompt/run, batch prompting bukan per-ulasan.
- **Rekomendasi:** Gemini sebagai pendekatan awal/prototipe; migrasi ke model klasik/IndoBERT bila data berlabel sudah cukup banyak dan beragam. Detail lengkap di Section 7–8 notebook.

## Cara menjalankan
```bash
pip install -r requirements.txt
export GEMINI_API_KEY="<api-key-kamu>"   # jangan commit key
cd notebook
jupyter nbconvert --to notebook --execute --inplace experiment_notebook.ipynb
```
Jika `data/gemini_predictions.csv` sudah ada, notebook memakai cache itu (tidak memanggil API); hapus file tersebut untuk memanggil ulang Gemini (butuh API key & kuota). Catatan: `GEMINI_API_KEY` tetap harus di-set karena sel setup membuat client; untuk run tanpa key cukup isi nilai dummy saat cache tersedia.
