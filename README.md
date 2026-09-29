# credit-score-classification-rf-xgboost
Klasifikasi skor kredit menggunakan Random Forest dan XGBoost, data cleaning, feature engineering, GridSearchCV, evaluasi multiclass, dan feature importance.

# Credit Score Classification: Random Forest vs XGBoost

Proyek akademik untuk membandingkan **Random Forest** dan **XGBoost** dalam mengklasifikasikan skor kredit menjadi **Poor, Standard, dan Good**. Analisis mencakup data cleaning, exploratory data analysis, feature engineering, hyperparameter tuning, evaluasi per kelas, dan feature importance.

## Tujuan

- Mengidentifikasi missing values, format numerik yang tidak konsisten, dan nilai tidak wajar.
- Menyiapkan fitur keuangan dan perilaku pembayaran untuk multiclass classification.
- Membandingkan dua model menggunakan GridSearchCV dengan macro F1 sebagai kriteria tuning.
- Menginterpretasikan fitur yang paling banyak digunakan oleh model terpilih melalui feature importance.

Proyek berfokus pada eksperimen klasifikasi skor kredit, bukan sistem persetujuan pinjaman atau aplikasi deployment.

## Dataset

Notebook menggunakan `Credis_Score_Dataset_A.csv`, yang disediakan untuk tugas akademik. Sumber publik asli belum dicantumkan dalam notebook.

- Data awal: **50.000 baris dan 28 kolom**.
- Target: `Credit_Score`.
- Data setelah filtering: **46.257 baris**.
- Input model setelah encoding: **17 fitur**.

Distribusi target pada data awal:

| Kelas | Jumlah | Proporsi |
|---|---:|---:|
| Standard | 26.587 | 53,17% |
| Poor | 14.499 | 29,00% |
| Good | 8.914 | 17,83% |

## Data Preparation dan EDA

1. Membersihkan karakter tambahan pada kolom numerik dan mengonversi tipe data.
2. Mengubah `Credit_History_Age` dari teks tahun/bulan menjadi `Credit_History_Age_Months`.
3. Mengisi missing values numerik menggunakan median atau mean, serta kategori menggunakan modus atau placeholder.
4. Membentuk `Num_Loan_Types` sebagai jumlah jenis pinjaman unik.
5. Menghapus baris dengan kategori `Payment_Behaviour` yang tidak valid.
6. Menandai nilai negatif tertentu dan usia di luar 0–100, lalu melakukan imputasi.
7. Menghapus identifier dari input model: `ID`, `Customer_ID`, `Name`, dan `SSN`, serta kolom `Month`.
8. Mengeksplorasi korelasi Pearson terhadap target ordinal dan asosiasi kategorikal menggunakan chi-square.

Beberapa variabel numerik memiliki distribusi menceng ke kanan. `Monthly_Inhand_Salary`, misalnya, memiliki 7.514 missing values pada data awal dan diimputasi menggunakan median.

<img width="582" height="454" alt="Screenshot 2026-09-29 at 15 08 21" src="https://github.com/user-attachments/assets/ec64a547-1f14-459e-935f-b02be5a61816" />

Seleksi fitur pada notebook menghapus delapan variabel numerik dengan korelasi linear yang kecil dan menghapus `Occupation`. Ini merupakan keputusan eksploratif; korelasi linear kecil tidak membuktikan sebuah fitur tidak berguna bagi model berbasis pohon.

## Pembagian Data dan Encoding

Data dibagi menggunakan `train_test_split` dengan `test_size=0.2`, `stratify=y`, dan `random_state=42`:

| Subset | Jumlah baris |
|---|---:|
| Training | 37.005 |
| Test | 9.252 |

Encoding dilakukan setelah split:

- `Credit_Mix`: ordinal mapping Bad = 0, Standard = 1, Good = 2.
- `Payment_of_Min_Amount`: No = 0, Yes = 1.
- `Payment_Behaviour`: OneHotEncoder yang di-fit pada training set, dengan `handle_unknown='ignore'`.
- Target XGBoost: Poor = 0, Standard = 1, Good = 2.

Tidak ada feature scaling, SMOTE, atau class weighting yang diterapkan pada eksperimen ini. Macro F1 digunakan saat tuning untuk memberi bobot evaluasi yang sama kepada setiap kelas.

## Model 1 — Random Forest

Random Forest menggabungkan prediksi banyak decision tree. Tuning mengeksplorasi jumlah pohon, kedalaman pohon, dan syarat minimum untuk melakukan split.

| Hyperparameter | Search space | Tujuan |
|---|---|---|
| `n_estimators` | 100, 300, 500 | Membandingkan jumlah pohon dalam ensemble |
| `max_depth` | 5, 10, None | Mengendalikan kedalaman dan fleksibilitas pohon |
| `min_samples_split` | 2, 5, 10 | Mengendalikan minimum sampel sebelum node dipecah |

GridSearchCV mengevaluasi **27 kombinasi** dengan **3-fold cross-validation** dan `scoring='f1_macro'`.

Konfigurasi terpilih:

```python
{'n_estimators': 500, 'max_depth': None, 'min_samples_split': 2}
```

## Model 2 — XGBoost

XGBoost membangun ensemble pohon secara bertahap. Model menggunakan `objective='multi:softmax'`, `num_class=3`, dan `eval_metric='mlogloss'`.

| Hyperparameter | Search space | Tujuan |
|---|---|---|
| `n_estimators` | 100, 300, 500 | Mengatur jumlah boosting rounds |
| `max_depth` | 3, 6, 10 | Mengatur kompleksitas tiap pohon |
| `learning_rate` | 0.01, 0.1, 0.2 | Mengatur kontribusi pohon baru |
| `subsample` | 0.8, 1.0 | Membandingkan proporsi sampel untuk pelatihan setiap pohon |

GridSearchCV mengevaluasi **54 kombinasi** dengan **3-fold cross-validation** dan `scoring='f1_macro'`.

Konfigurasi terpilih:

```python
{'n_estimators': 300, 'max_depth': 10, 'learning_rate': 0.1, 'subsample': 0.8}
```

Notebook menyimpan hasil kedua model setelah tuning. Tidak ada evaluasi default model terpisah, sehingga peningkatan sebelum–sesudah tuning belum dapat dihitung.

## Hasil Evaluasi

Angka berikut berasal dari output notebook pada test set:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Random Forest | 71,54% | 69,27% |
| **XGBoost** | **72,14%** | **69,67%** |

XGBoost unggul sekitar **0,59 percentage point pada accuracy** dan **0,39 percentage point pada macro F1**, dihitung dari nilai sebelum pembulatan. Selisihnya kecil dan belum diuji kestabilannya melalui pengulangan eksperimen.

### Evaluasi Per Kelas

Nilai precision, recall, dan F1 di bawah mengikuti pembulatan classification report:

| Model | Kelas | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| Random Forest | Poor | 0,73 | 0,70 | 0,72 | 2.683 |
| Random Forest | Standard | 0,76 | 0,75 | 0,75 | 4.922 |
| Random Forest | Good | 0,58 | 0,64 | 0,61 | 1.647 |
| XGBoost | Poor | 0,73 | 0,70 | 0,72 | 2.683 |
| XGBoost | Standard | 0,74 | 0,77 | 0,76 | 4.922 |
| XGBoost | Good | 0,64 | 0,60 | 0,62 | 1.647 |

XGBoost dipilih berdasarkan accuracy dan macro F1 yang sedikit lebih tinggi pada eksperimen ini. Namun, Random Forest memiliki recall kelas Good yang lebih tinggi (0,64 vs 0,60). Dengan demikian, XGBoost tidak unggul pada seluruh metrik dan kelas.

## Feature Importance

Lima fitur teratas berdasarkan `feature_importances_` dari XGBoost:

| Fitur | Importance |
|---|---:|
| Credit_Mix | 0,3421 |
| Payment_of_Min_Amount | 0,1119 |
| Outstanding_Debt | 0,0783 |
| Delay_from_due_date | 0,0426 |
| Payment_Behaviour_Low_spent_Small_value_payments | 0,0373 |

<img width="787" height="479" alt="Screenshot 2026-09-29 at 15 08 51" src="https://github.com/user-attachments/assets/e9b73d2c-ac03-459a-b812-083071fce53b" />

`Credit_Mix` memiliki importance tertinggi dalam model ini, diikuti perilaku pembayaran minimum dan utang yang belum dilunasi. Nilai ini merupakan ukuran importance internal model, bukan besarnya pengaruh kausal, arah hubungan, atau persentase perubahan skor kredit.

## Temuan Utama

- Data memerlukan penanganan format numerik, missing values, kategori tidak valid, dan anomali sebelum modeling.
- XGBoost sedikit mengungguli Random Forest secara agregat, tetapi kelas Good masih memiliki F1 paling rendah pada kedua model.
- Credit mix, pembayaran minimum, dan outstanding debt menjadi fitur teratas pada model XGBoost terpilih.
- Hasil merupakan evaluasi eksploratif yang masih memerlukan perbaikan pemisahan preprocessing dan validasi sebelum dijadikan estimasi performa pada pelanggan baru.


## Tools

Python, pandas, NumPy, Matplotlib, SciPy, scikit-learn, dan XGBoost.

