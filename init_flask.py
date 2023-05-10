from flask import Flask, request
import tensorflow as tf
import pandas as pd
from joblib import load
from flask_cors import CORS

application = Flask(__name__)
CORS(application)

@application.route('/flask', methods=['GET'])
def flask():
    return '<p>https://flask.palletsprojects.com/en/2.2.x/</p>'

@application.get('/api/get')
def get_method():
    word = request.args.get('word', '<no word>')
    return {
        'hello': 'hello, ' + word
    }

@application.route('/')
def main():
    return '<p>Hello, World!</p>'

@application.route('/polo')
def polo():
    return {
        'precio': 10000
    }

@application.get('/car')
def car():
    color = request.args.get('color', '')
    make = request.args.get('make', '')
    province = request.args.get('province', '')
    model_car = request.args.get('model', '')
    transmission = request.args.get('transmission', '')
    fuel = request.args.get('fuel', '')
    body = request.args.get('body', '')
    seller = request.args.get('seller', '')
    km = int(request.args.get('km', '0'))
    year = int(request.args.get('year', '0'))
    cv = int(request.args.get('cv', '0'))
    cubic = int(request.args.get('cubic', '0'))
    doors = int(request.args.get('doors', '0'))
    
    # Validar que todos los campos requeridos estén presentes
    missing_fields = []

    if not color:
        missing_fields.append("color")
    if not make:
        missing_fields.append("make")
    if not province:
        missing_fields.append("province")
    if not model_car:
        missing_fields.append("model")
    if not transmission:
        missing_fields.append("transmission")
    if not fuel:
        missing_fields.append("fuel")
    if not body:
        missing_fields.append("body")
    if not seller:
        missing_fields.append("seller")
    if km == 0:
        missing_fields.append("km")
    if year == 0:
        missing_fields.append("year")
    if cv == 0:
        missing_fields.append("cv")
    if cubic == 0:
        missing_fields.append("cubic")
    if doors == 0:
        missing_fields.append("doors")

    if missing_fields:
        error_message = f"Error: Faltan los siguientes campos: {', '.join(missing_fields)}"
        return {"error": error_message}
    
    coche = {
        'Color': color,
        'Make': make,
        'Province': province,
        'Transmission Type': transmission,
        'Fuel Types': fuel,
        'Body Type': body,
        'Model': model_car,
        'Seller Type': seller,
        'KM': km,
        'Year': year,
        'Horsepower': cv,
        'Cubic Capacity': cubic,
        'Doors': doors,
    }
    print(coche)
    model = tf.keras.models.load_model('modelo.hdf5')
    # Cargar el objeto scaler
    scaler = load('scaler.pkl')
    encoder = load('encoder.pkl')
    # Convertir el diccionario a un DataFrame
    df = pd.DataFrame([coche])
    categorical_columns = ['Color', 'Make', 'Model', 'Province', 'Transmission Type', 'Fuel Types', 'Seller Type', 'Body Type']
    categorical_data = df[categorical_columns]
    categorical_data_encoded = encoder.transform(categorical_data)
    encoded_df = pd.DataFrame(categorical_data_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    # Aplicar la transformación MinMaxScaler a los datos numéricos
    numerical_data_api = df[["Horsepower", "Cubic Capacity", "Doors", "KM", "Year"]]
    print('hey')
    numerical_data_api_scaled = scaler.transform(numerical_data_api)
    # Crear un DataFrame con los datos numéricos normalizados
    numerical_data_api_scaled_df = pd.DataFrame(numerical_data_api_scaled, columns=["Horsepower", "Cubic Capacity", "Doors", "KM", "Year"])
    # Reemplazar el DataFrame 'not_categorical_data' por 'numerical_data_api_scaled_df'
    df_final = pd.concat([numerical_data_api_scaled_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    print(df_final.head())
    precio = model.predict(df_final)
    print(precio[0][0])
    return {
        'precio': float(precio[0][0])
    }

application.run()