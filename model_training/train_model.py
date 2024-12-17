import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Muat dataset (Ganti dengan path file CSV Anda)
data = pd.read_csv('data_stunting.csv')  # Ganti dengan path file CSV Anda

# Tentukan fitur (X) dan target (y)
X = data[['Sum of Konsumsi_protein', 'Sum of BCG', 
          'Sum of dinkes-od_17147_jumlah_balita_stunting_berdasarkan_kabupatenkota_v1 (3).Data_stu',
          'Sum of kapitaekonomi.Pendapatan', 'Sum of kode_provinsi', 
          'Sum of Konsumsi_kacang.Konsumsi_kacang', 'tahun', 'Sum of id']]  # Fitur
y = data['target']  # Ganti 'target' dengan nama kolom target yang sesuai

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model dengan RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Simpan model ke dalam file .pkl
joblib.dump(model, 'models/model.pkl')  # Menyimpan model di folder 'app/models/'
