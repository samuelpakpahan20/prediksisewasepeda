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

![Hasil MSE Decision Tree](https://raw.githubusercontent.com/samuelpakpahan20/prediksisewasepeda/master/images/mseDecision.png)

Algoritma Decision Tree tampaknya memiliki akurasi yang jauh lebih rendah dibandingkan Algoritma Linear Regression. Nilai erorr tampaknya diminimalkan dengan mengatur parameter `min_samples_leaf` ke angka 9 hingga 11. MSE terendah yang kita lihat adalah sekitar 2450, ketika menggunakan 10 minimum leafs.

Hal ini kemungkinan karena memperhitungkan fitur-fitur yang non-linear, seperti kolom `season` dan `time_label` yang tadi dibuat.

### Random Forest
*Note : Untuk melihat penerapan kodenya dapat dilihat pada Bab Evaluasi

![Hasil MSE Random Forest](https://raw.githubusercontent.com/samuelpakpahan20/prediksisewasepeda/master/images/mseRandomForest.png)

Dari perbandingan hasil MSE ini, dapat disimpulkan bahwa **Algoritma Random Forest menciptakan model dengan akurasi prediksi terbaik**. Algoritma Random Forest mengembalikan MSE hanya 1605, turun dari 2450 menggunakan algoritma Decision Tree, dan 15709 menggunakan algoritma Linear Regression.

Banyaknya peningkatan ini disebabkan oleh fakta bahwa Random Forest :
1. **Jauh lebih akurat** daripada model sederhana seperti linear regression, dan
2. **Cenderung overfit** daripada Decision Tree.

Untuk meminimalkan overfitting lebih lanjut, kita dapat bereksperimen dengan parameter seperti maximum depth, dan minimum samples per leaf. Dalam kasus ini, kita menemukan bahwa MSE Random Forest diminimalkan menggunakan 1 minimum leafs, dan max depth = 23.

## Evaluation
Metrik yang saya gunakan pada prediksi ini adalah **Mean Squared Error (MSE)** yang menghitung selisih rata-rata nilai sebenarnya dengan nilai prediksi. MSE didefinisikan dalam persamaan berikut

![Rumus MSE](https://raw.githubusercontent.com/samuelpakpahan20/prediksisewasepeda/master/images/rumusmse.jpeg)

*Keterangan:*

*n = jumlah dataset*

*yi = nilai sebenarnya*

*y_pred = nilai prediksi*

Setelah kita mendapatkan fitur-fitur terbaik dan telah melakukan modeling dengan menggunakan algoritma Linear Regression dan Decision Tree. Selanjutnya, kita akan mengevaluasi dengan menggunakan Algoritma Random Forest. Untuk menggunakan algoritma tersebut, masukkan kode berikut:
```
from sklearn.ensemble import RandomForestRegressor

# Dataframe berikut akan menyimpan nilai MSE pada berbagai 
# variasi parameter min_samples_leaf dan max_depth.

random_forest_mse = pd.DataFrame()

for leafs in range(1,8):
    
    mse_list = []
    
    for depth in range(10,30):
        
        # Membuat Instance Model
        reg = RandomForestRegressor(min_samples_leaf = leafs, max_depth=depth)

        # Melatih Model
        reg.fit(train[features], train['cnt'])

        # Menguji Model
        predictions = reg.predict(test[features])

        # Menghitung Error
        mse = mean_squared_error(test['cnt'], predictions)
        mse_list.append(mse)

    random_forest_mse[leafs] = pd.Series(mse_list)
```
Gunakan kode berikut untuk melihat hasil prediksinya.
```
random_forest_mse.columns.name = "Jumlah Min Leafs"

random_forest_mse.index = range(10,30)

random_forest_mse.index.name = "Max Depth"

random_forest_mse
```
Hasilnya seperti berikut.

![Hasil MSE Random Forest](https://raw.githubusercontent.com/samuelpakpahan20/prediksisewasepeda/master/images/mseRandomForest.png)

Selanjutnya, kita akan tentukan di mana MSE terendahnya. Tuliskan kode berikut.
```
a, b = random_forest_mse.stack().idxmin()
print(random_forest_mse.loc[[a], [b]])
```
Maka MSE terendahnya sebagai berikut.

![MSE Terendah](https://raw.githubusercontent.com/samuelpakpahan20/prediksisewasepeda/master/images/hasilrandomforest.JPG)

Dari hasil diatas, dapat kita ketahui bahwa MSE yang relatif rendah adalah 1605 dan diamati ketika parameter `min_samples_leaf`= 1, dan parameter `max_depth`= 23.

**---Ini adalah bagian akhir laporan---**
