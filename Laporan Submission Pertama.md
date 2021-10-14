# Laporan Proyek Machine Learning - Samuel Partogi Pakpahan

## Domain Proyek
Sistem rental sepeda adalah generasi baru penyewaan sepeda tradisional di mana seluruh proses mulai dari keanggotaan, penyewaan, dan pengembalian menjadi otomatis. Melalui sistem ini, pengguna dapat dengan mudah menyewa sepeda dari posisi tertentu dan kembali lagi di posisi lain. Saat ini, ada minat besar dalam sistem ini karena peran penting mereka dalam masalah lalu lintas, lingkungan dan kesehatan.

Menurut artikel [ini](https://ddot.dc.gov/page/capital-bikeshare), banyak kota di Amerika memiliki lebih dari 500 stasiun rental sepeda di seluruh wilayahnya termasuk Washington, D.C., dan terdiri dari lebih dari 500 ribu sepeda.

## Business Understanding
Terlepas dari aplikasi dunia nyata yang menarik dari sistem rental sepeda, karakteristik data yang dihasilkan oleh sistem ini membuatnya menarik untuk penelitian. Berlawanan dengan layanan transportasi lain seperti bus atau kereta bawah tanah, durasi perjalanan, posisi keberangkatan dan kedatangan dicatat secara eksplisit dalam sistem ini. Fitur ini mengubah sistem rental sepeda menjadi jaringan sensor virtual yang dapat digunakan untuk merasakan mobilitas di dalam kota. Oleh karena itu, diharapkan sebagian besar peristiwa penting di kota dapat dideteksi melalui pemantauan data ini. 

### Problem Statements
Berdasarkan kondisi diatas, saya akan mengembangkan sebuah model prediksi jumlah sepeda untuk menjawab permasalahan berikut.
- Berapa jumlah sepeda yang disewa pada jam tertentu?

### Goals
Untuk  menjawab pertanyaan tersebut, saya membuat predictive modelling dengan tujuan atau goals, yaitu membuat model machine learning yang dapat memprediksi jumlah sepeda berdasarkan fitur-fitur terbaik yang ada.

### Solution statements
Untuk membuat model machine learning seperti yang disebutkan diatas, saya menggunakan berbagai algoritma seperti Linear Regression, Decision Trer, dan Random Forest, untuk melihat hasil prediksi mana yang terbaik dan mengembalikan nilai error terendah.
- **Liniear Regression**, algoritma ini cocok dipakai ketika terdapat hubungan linear pada data. Namun untuk implementasi pada kebanyakan kasus, ia kurang direkomendasikan. Sebabnya, regresi linier selalu mengasumsikan ada hubungan linier pada data, padahal tidak.
- **Decision Tree**, algoritma ini memprediksi sebuah kelas (klasifikasi) atau nilai (regresi) berdasarkan aturan-aturan yang dibentuk setelah mempelajari data.
- **Random Forest**, algoritma ini termasuk ke dalam kelompok model ensemble (group). Model ensemble merupakan model prediksi yang terdiri dari beberapa model dan bekerja secara bersama-sama. Ide dibalik model ensemble adalah sekelompok model yang bekerja bersama menyelesaikan masalah. Sehingga, tingkat keberhasilan akan lebih tinggi dibanding model yang bekerja sendirian. Pada model ensemble, setiap model harus membuat prediksi secara independen. Kemudian, prediksi dari setiap model ensemble ini digabungkan untuk membuat prediksi akhir. 

## Data Understanding
Dataset yang saya gunakan adalah [Bike Sharing Dataset](http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset#). Seperti yang saya sebutkan sebelumnya, banyak kota di Amerika, termasuk Washington, D.C., memiliki stasiun rental sepeda. Distrik mereka membuat dataset ini dengan mengumpulkan data jumlah sepeda yang disewa orang, per jam dan hari.

Variabel-variabel pada Bike Sharing Dataset adalah sebagai berikut:
- `instant` : ID Unik sepeda
- `dteday` : tanggal sewa
- `season` : musim penyewaan terjadi
- `yr` : tahun penyewaan terjadi
- `mnth` : bulan penyewaan terjadi
- `hr` : jam penyewaan terjadi
- `holiday` : apakah hari itu hari libur atau tidak
- `weekday` : hari dalam seminggu (sebagai angka, 0 hingga 7)
- `workingday` : apakah hari itu adalah hari kerja atau tidak
- `weathersit` : cuaca
- `temp` : suhu
- `atemp` : suhu yang disesuaikan
- `hum` : kelembapan
- `windspeed` : kecepatan angin
- `casual` : jumlah pengendara biasa (non-membership)
- `registered` : jumlah pengendara membership
- `cnt` : jumlah total persewaan sepeda (casual + registered)

## Data Preparation
Sebelum digunakan, Dataset melewati beberapa proses seperti berikut:
1. Menggunakan metode `corr` pada dataset untuk melihat bagaimana setiap kolom berkorelasi dengan kolom `cnt`.
2. Membuat sebuah fungsi, `assign_label`, yang membangun kolom baru `time_label` yang berdasarkan nilai di kolom `hr`. Dimana, fungsi ini juga akan membuat label [`morning`, `afternoon`, `evening`, `night`] untuk mewakili periode waktu tertentu.
3. Membagi dataset menjadi 80% train set, dan sisanya ke test set.


## Modeling
### Linear Regression
Sebelum melakukan pemodelan kita harus menentukan fitur terbaiknya terlebih dahulu. Tuliskan kode berikut.
```
column_names = bike_rentals.columns.tolist()

# Menghapus kolom yang tidak kita pakai sebagai fitur
column_names.remove("cnt")
column_names.remove("registered")
column_names.remove("casual")
column_names.remove("dteday")

features = column_names

features
```
Hasilnya seperti berikut.

![Fitur](https://raw.githubusercontent.com/samuelpakpahan20/prediksisewasepeda/master/images/fitur.JPG)

Setelah fiturnya di dapatkan, kita akan proses dengan linear regression. Masukkan kode berikut:
```
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Membuat Instance Model
lr = LinearRegression()

# Melatih Model
lr.fit(train[features], train['cnt'])

# Menguji Model
predictions = lr.predict(test[features])

# Menghitung error
mse = mean_squared_error(test['cnt'], predictions)

print("Hasil Mean Squared Error menggunakan Linear Regression : ", mse)
```
Maka kita dapatkan hasil prediksinya sebagai berikut.

![Hasil MSE Linear](https://raw.githubusercontent.com/samuelpakpahan20/prediksisewasepeda/master/images/mseLinear.JPG)

### Decision Tree
Dari hasil prediksi linear regression diatas, dapat kita ketahui bahwa nilai error masih sangat tinggi yang berarti algoritma linear regression tidak berjalan dengan baik karena tidak dapat secara akurat memodelkan beberapa fitur non-linear yang ada di dataset. Ini mungkin karena fakta bahwa ada beberapa jumlah sewa yang sangat tinggi.

Kita akan mencoba menggunakan algoritma Decision Tree. Gunakan kode berikut.
```
from sklearn.tree import DecisionTreeRegressor

decision_tree_mse = {}

for leafs in range(1,15):

    reg = DecisionTreeRegressor(min_samples_leaf=leafs)

    reg.fit(train[features], train['cnt'])

    predictions = reg.predict(test[features])

    mse = mean_squared_error(test['cnt'], predictions)
    
    decision_tree_mse[leafs] = mse

for each in decision_tree_mse:
    print("MSE using", each, "minimum leafs:", decision_tree_mse[each])
```
Hasilnya adalah sebagai berikut:

![Hasil MSE Decision Tree](https://raw.githubusercontent.com/samuelpakpahan20/prediksisewasepeda/master/images/mseDecission.JPG)

Algoritma Decision Tree tampaknya memiliki akurasi yang jauh lebih rendah dibandingkan Algoritma Linear Regression. Nilai erorr tampaknya diminimalkan dengan mengatur parameter `min_samples_leaf` ke angka 4 hingga 8. MSE terendah yang kita lihat adalah sekitar 2582, ketika menggunakan 4 minimum leafs.

Hal ini kemungkinan karena memperhitungkan fitur-fitur yang non-linear, seperti kolom `season` dan `time_label` yang tadi dibuat.

### Random Forest
*Note : Untuk melihat penerapan kodenya dapat dilihat pada Bab Evaluasi

![Hasil MSE Random Forest](https://raw.githubusercontent.com/samuelpakpahan20/prediksisewasepeda/master/images/mseRandomForest.PNG)



Dari perbandingan hasil MSE ini, dapat disimpulkan bahwa:
Algoritma Random Forest menciptakan model dengan akurasi prediksi terbaik. Algoritma Random Forest mengembalikan MSE hanya 1724, turun dari 2582 menggunakan algoritma Decision Tree, dan 16185 menggunakan algoritma Linear Regression.

Banyaknya peningkatan ini disebabkan oleh fakta bahwa Random Forest :
1. Jauh lebih akurat daripada model sederhana seperti linear regression, dan
2. Cenderung overfit daripada Decision Tree.

Untuk meminimalkan overfitting lebih lanjut, kita dapat bereksperimen dengan parameter seperti maximum depth, dan minimum samples per leaf. Dalam kasus ini, kita menemukan bahwa MSE diminimalkan menggunakan 1 minimum leafs, dan max depth = 21.

## Evaluation
Metrik yang saya gunakan pada prediksi ini adalah **Mean Squared Error (MSE)** yang merupakan hasil dari akar kuadrat Mean Square Error (MSE). RMSE adalah cara standar untuk mengukur kesalahan suatu model dalam memprediksi data kuantitatif. RMSE didefinisikan dalam persamaan berikut

![Rumus MSE](https://raw.githubusercontent.com/samuelpakpahan20/prediksisewasepeda/master/images/rumusmse.JPEG)

*Keterangan:*

*n = jumlah dataset*

*yi = nilai sebenarnya*

*y_pred = nilai prediksi*


Namun, sebelum menghitung nilai RMSE dalam model, kita perlu melakukan proses pelatihan dan validasi. Untuk kita perlu membuat sebuah fungsi `knn_train_test`. Fungsi ini memiliki 3 parameter, yaitu nama kolom latih, nama kolom target, nama objek Dataframe.

Fungsi ini akan melakukan tindakan seperti memisahkan dataset menjadi data latih dan test, membuat instance kelas KNeighborsRegressor, menyesuaikan dengan model pada data latih, kemudian menjalankan prediksi pada data test, menghitung RMSE, dan mengembalikannya.

Berikut kode untuk fungsi tersebut:
```
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

def knn_train_test3(train_cols, target_col, df):
    
    np.random.seed(3)
    
    # Mengacak baris
    shuffle = np.random.permutation(df.index)
    df = df.reindex(shuffle)
    
    # Memisahkan dataset menjadi data latih dan test
    train_df = df.iloc[:101]
    test_df = df.iloc[101:]
    
    # Instance Kelas KNeighborsRegressor
    knn = KNeighborsRegressor()
    
    # Model training
    knn.fit(train_df[train_cols], train_df[target_col])
    
    # Hasil prediksi:
    predictions = knn.predict(test_df[train_cols])
    
    # Menghitung RMSE:
    rmse = (mean_squared_error(test_df[target_col], predictions))**0.5
    
    # Mengembalikan nilai RMSE
    return rmse
```

Gunakan kode berikut untuk membuat daftar list setiap kolom dalam Dataframe.
```
list_of_cols = normalized.columns.tolist()

list_of_cols.remove('price')

list_of_cols
```

Selanjutnya, uji RMSE untuk setiap kolom. Tuliskan kode berikut.
```
rmse_dict = {}

for each in list_of_cols:
    rmse = knn_train_test(each, 'price', normalized)
    rmse_dict[each] = rmse 
    
rmse_dict
```

Hasil uji RMSEnya sebagai berikut.

![Hasil uji RMSE](https://raw.githubusercontent.com/samuelpakpahan20/prediksihargamobil/master/images/ujiRMSE.JPG)

Untuk memudahkan, buat plot metrik tersebut dengan bar chart. Tuliskan kode di bawah ini:
```
import matplotlib.pyplot as plt
import matplotlib.style as style
%matplotlib inline

rmse_series = pd.Series(rmse_dict)
fig, ax = plt.subplots()
rmse_series.sort_values(ascending = True).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)
```

Hasilnya sebagai berikut.

![Plot Metrik](https://raw.githubusercontent.com/samuelpakpahan20/prediksihargamobil/master/images/visualisasi.png)

Dari gambar di atas, terlihat bahwa, kolom `engine-size`, `width`, `horsepower`, `highway-mpg`, `curb-weight` memiliki nilai RMSE terendah. Kolom inilah yang akan di pilih sebagai fitur terbaik untuk melakukan prediksi harga mobil.
![MSE Terendah](https://raw.githubusercontent.com/samuelpakpahan20/prediksisewasepeda/master/images/hasilrandomforest.JPG)

**---Ini adalah bagian akhir laporan---**
