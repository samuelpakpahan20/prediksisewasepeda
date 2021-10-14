# Laporan Proyek Machine Learning - Samuel Partogi Pakpahan

## Domain Proyek
Pada proyek ini saya menggunakan domain Ekonomi dan bisnis. Seperti yang kita ketahui organisasi bisnis seperti real estate, perusahaan mobil, dan bisnis penjualan daring perlu terus-menerus menyesuaikan harga mereka untuk memaksimalkan keuntungan. Perusahaan harus bisa meningkatkan atau mempertahankan pendapatan dan profit meski menghadapi situasi yang sulit. Situasi sulit ini bisa disebabkan oleh berbagai faktor, antara lain perubahan musim, pergeseran permintaan pelanggan, dan terjadinya kejadian khusus seperti pandemi.

Model predictive analytics dapat dilatih untuk memprediksi harga optimal berdasarkan data atau catatan historis penjualan. Perusahaan dapat menggunakan prediksi ini sebagai masukan untuk menentukan berbagai keputusan bisnis, misalnya strategi penetapan harga.

Merujuk pada penelitian [berikut](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3702236), India memiliki salah satu pasar mobil terbesar di seluruh dunia setiap hari banyak pembeli biasanya menjual mobil mereka setelah digunakan pembeli lain (mobil bekas). Banyak juga platform seperti cars24.com, cardekho.com dan OLX.com sebagai media bagi pembeli untuk menjual mobil bekas mereka, tetapi berapa harga mobil baru yang seharusnya? Ini adalah pertanyaan terberat yang pernah ada. Algoritma Machine Learning dapat memberikan solusi untuk masalah ini. Menggunakan riwayat data penjualan mobil, kita dapat memprediksi harga mobil yang wajar.


## Business Understanding
Seiring dengan tingginya aktivitas dan bisnis, mobil sudah menjadi kebutuhan pokok. Disisi lain, harga mobil baru semakin tinggi dengan berbagai *feature* yang disematkan pada produk baru. Untuk memenuhi kesenjangan tersebut, masyarakat mencari alternatif untuk membeli mobil bekas yang kondisi masih baik dan layak digunakan. Tingginya minat masyarakat terhadap mobil bekas membuat profit perusahaan mobil semakin menurun, hal ini ditandai dengan banyaknya *showroom* mobil bekas. Tak luput diantara *showroom* sangat kompetitif bersaing agar tetap eksis dalam bisnis mobil bekas. Salah satu masalah yang dihadapi semua perusahaan mobil adalah menentukan harga secara cepat dan akurat sehingga perusahaan bisa mendapatkan profit sebesar mungkin. Berdasarkan penelitian yang saya sebutkan sebelumnya, kondisi saat ini prediksi harga mobil pun masih popular. Berbagai *showroom* maupun perusahaan mobil saling bersaing harga untuk mendapatkan pelanggan.

### Problem Statements
Berdasarkan kondisi diatas, saya akan mengembangkan sebuah sistem prediksi harga mobil untuk menjawab permasalahan berikut.

- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap harga mobil?
- Berapa harga pasar mobil dengan karakteristik atau fitur tertentu?  

### Goals
Untuk  menjawab pertanyaan tersebut, saya membuat predictive modelling dengan tujuan atau goals sebagai berikut:

- Menentukan fitur mana yang terbaik (berkorelasi) untuk digunakan memprediksi harga mobil.
- Membuat model machine learning yang dapat memprediksi harga mobil seakurat mungkin berdasarkan fitur-fitur terbaik yang ada.

### Solution statements
Pada proyek ini saya menggunakan algoritma K-Nearest Neighbor (KNN). KNN adalah algoritma yang relatif sederhana dibandingkan dengan algoritma lain, karena algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan.

KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k-tetangga terdekat. Oleh sebab itu, mengapa algoritma ini dinamakan K-nearest neighbor (sejumlah k tetangga terdekat).

Meskipun algoritma KNN mudah dipahami dan digunakan, ia memiliki kekurangan jika dihadapkan pada jumlah fitur atau dimensi yang besar. Permasalahan ini sering disebut sebagai *curse of dimensionality* (kutukan dimensi). Pada dasarnya, permasalahan ini muncul ketika jumlah sampel meningkat secara eksponensial seiring dengan jumlah dimensi (fitur) pada data. Jadi, jika menggunakan model KNN, pastikan data yang digunakan memiliki fitur yang relatif sedikit.

## Data Understanding
Dataset yang saya gunakan adalah [Automobile Data Set](https://archive.ics.uci.edu/ml/datasets/automobile). Dataset ini terdiri dari tiga jenis entitas: 
1. Spesifikasi mobil dalam hal berbagai karakteristik, 
2. Peringkat risiko asuransi yang ditetapkan, 
3. Kerugian yang dinormalisasi dalam penggunaan dibandingkan dengan mobil lain.  

Variabel-variabel pada Automobile Data Set adalah sebagai berikut:
- symboling : sombolisasi peringkat sejauh mana mobil berisiko. Nilai +3 menunjukkan bahwa mobil itu berisiko, -3 bahwa itu mungkin cukup aman.
- normalized-losses : Perbandingan kerugian pemakaian dengan mobil lain.
- make : nama perusahaan pembuat mobil
- fuel-type : jenis bahan bakar.
- num-of-doors : jumlah pintu
- body-style : jenis body mobil
- drive-wheels : jenis sistem penggerak ban mobil
- engine-location : lokasi mesin
- wheel-base : dimensi jarak sumbu roda
- length : panjang mobil
- width : lebar mobil
- height : tinggi mobil
- curb-weight : massa total mobil
- engine-type : jenis mesin
- num-of-cylinders : jumlah silinder mobil
- engine-size : ukuran mesin
- fuel-system : sistem bahan bakar
- bore : diameter silinder atau isi liner blok mesin
- stroke : jarak piston yang bergerak maju-mundur dalam blok mesin
- compression-ratio : nilai yang mewakili rasio volume ruang pembakaran dari kapasitas terbesar ke kapasitas terkecil.
-  horsepower : sebagai satuan daya yang setara dengan tenaga kuda
-  price: harga mobil

## Data Preparation
Sebelum digunakan, Dataset melewati beberapa proses seperti berikut:
1. Inisialisasi variabel (kolom) secara manual, karena dataset asli tidak disertakan.
2. Mengubah kolom bertipe data `objek` menjadi `numeric`, hal ini dilakukan karena ada beberapa kolom yang bernilai numerik menggunakan tipe data `object` sehingga nilai dari kolom tidak dapat di proses. Kemudian ada beberapa kolom yang penulisan nilainya diubah ke dalam angka (numerik).
4. Menggunakan metode `df.replace()` untuk mmengisi nilai yang hilang, lalu melakukan teknik drop terhadap data yang hilang, menggantinya dengan mean. Hal ini dilakukan agar data valid dan dapat digunakan.
6. Melakukan teknik normalisasi pada kolom numerik sehingga semua nilai hanya bernilai dari 0 hingga 1, kecuali kolom `price`


## Modeling
### Model Univariate dan Multivariate
Model Univariate awalnya digunakan untuk menentukan fitur terbaik untuk pemodelan. Hal ini dilakukan dengan memvariasikan hyperparameter nilai k (n_neighbors) pada fungsi KNeighborsRegressor.

![Visualisasi Model Univariate](https://raw.githubusercontent.com/samuelpakpahan20/prediksihargamobil/master/images/Univariate.png)
Setelah fitur terbaik di dapatkan, selanjutnya dilakukan pemodelan multivariate menggunakan 3, 4, dan 5 fitur terbaik, dengan parameter nilai k bervariasi dari 1-25.

![Visualisasi Model Multivariate](https://raw.githubusercontent.com/samuelpakpahan20/prediksihargamobil/master/images/Multivariate.png)

Dari perbandingan grafik ini, dapat disimpulkan bahwa:
1. **Model Predictive harus dilakukan menggunakan 4 atau 5 fitur terbaik** (5 hal yang perlu diperhatikan dalam memprediksi harga pasar mobil, yaitu `engine-size`, `width`, `horsepower`, `highway-mpg`, `curb-weight`)
2. **Nilai k yang <= 3** harus digunakan untuk meminimalkan nilai RMSE.

## Evaluation
Metrik yang saya gunakan pada prediksi ini adalah **Root Mean Squared Error (RMSE)** yang merupakan hasil dari akar kuadrat Mean Square Error (MSE). RMSE adalah cara standar untuk mengukur kesalahan suatu model dalam memprediksi data kuantitatif. RMSE didefinisikan dalam persamaan berikut

![Rumus RMSE](https://raw.githubusercontent.com/samuelpakpahan20/prediksihargamobil/master/images/RMSE.png)

*Keterangan:*

*n = jumlah dataset*

*yi = nilai sebenarnya*

*ŷ = nilai prediksi*


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

**---Ini adalah bagian akhir laporan---**
