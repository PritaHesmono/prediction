from flask import Flask, request, render_template
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Muat model yang sudah dilatih
model = joblib.load('app/models/model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Mengambil file CSV yang di-upload
    file = request.files['file']
    if file:
        # Membaca CSV yang di-upload
        data = pd.read_csv(file)
        
        # Pra-pemrosesan data jika diperlukan
        # Asumsi data sudah sesuai dengan format yang diinginkan
        X = data.drop(columns=['target'])  # Menghapus kolom target jika ada
        predictions = model.predict(X)  # Menggunakan model untuk prediksi

        # Menyiapkan hasil prediksi
        result = []
        for i, year in enumerate(data['tahun']):
            predicted_year = year + 1  # Prediksi untuk tahun berikutnya
            predicted_percentage = predictions[i]  # Persentase prediksi stunting
            
            # Menambahkan hasil prediksi ke dalam list
            result.append({
                'tahun': year,
                'predicted_year': predicted_year,
                'predicted_percentage': round(predicted_percentage, 2)
            })
        
        # Menyampaikan hasil prediksi ke template
        return render_template('index.html', result=result)

    return "File tidak valid atau tidak ditemukan"

if __name__ == '__main__':
    app.run(debug=True)
