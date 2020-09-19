# -*- coding: utf-8 -*-

import numpy as np
import os
from flask import Flask, request, render_template, make_response
from sklearn.externals import joblib

app = Flask(__name__, static_url_path='/static')
GB = joblib.load('model/GB.pkl')
MLP = joblib.load('model/MLP.pkl')
LSTM = joblib.load('model/LSTM.pkl')

@app.route('/')
def display_gui():
    return render_template('templates/template.html')

@app.route('/verificar', methods=['POST'])
def verificar():
	lstm_features = []
	lstm_features.append(request.form['DATA_FESTIVA'])
	lstm_features.append(request.form['FERIADO'])
	lstm_features.append(request.form['QTD_CONCORRENTES'])
	lstm_features.append(request.form['UMIDADE'])
	lstm_features.append(request.form['VENDAS_ONTEM'])
	lstm_y_pred = lstm.predict(scaler.transform([lstm_features])).round().astype(int)[0]

	gb_features = []
	gb_features.append(request.form['ALTA_TEMPORADA'])
	gb_features.append(request.form['DATA_FESTIVA'])
	gb_features.append(request.form['FERIADO'])
	gb_features.append(request.form['QTD_CONCORRENTES'])
	gb_features.append(request.form['VENDAS_ONTEM'])
	gb_y_pred = gb.predict(scaler.transform([gb_features])).round().astype(int)[0]

	mlp_features = []
	mlp_features.append(request.form['DATA_FESTIVA'])
	mlp_features.append(request.form['FERIADO'])
	mlp_features.append(request.form['POS_DATA_FESTIVA'])
	mlp_features.append(request.form['TEMPERATURA'])
	mlp_features.append(request.form['UMIDADE'])
	mlp_features.append(request.form['VENDAS_ONTEM'])
	mlp_y_pred = mlp.predict(scaler.transform([mlp_features])).round().astype(int)[0]

	qtd_almoco_lstm = LSTM.predict(teste)[0]
	qtd_almoco_gb = GB.predict(teste)[0]
	qtd_almoco_mlp = MLP.predict(teste)[0]
	qtd_almoco_ensemble = (qtd_almoco_lstm + qtd_almoco_gb + qtd_almoco_mlp) / 3
	print(qtd_almoco_ensemble)

	return render_template('template.html',qtd_almoco=str(qtd_almoco_ensemble))

if __name__ == "__main__":
        port = int(os.environ.get('PORT', 5500))
        app.run(host='0.0.0.0', port=port)

