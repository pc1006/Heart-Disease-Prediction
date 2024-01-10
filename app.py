# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
model = pickle.load(open('xgb_model.pkl', 'rb'))

app = Flask(__name__,template_folder='web')

@app.route('/')
def home():
	return render_template('main.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        age = int(request.form['age'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = request.form.get('fbs')
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        sex = int(request.form['sex'])
        cp1 = int(request.form['cp1'])
        cp2 = int(request.form['cp2'])
        cp3 = int(request.form['cp3'])
        restecg_left = int(request.form['restecg_left'])
        restecg_normal = int(request.form['restecg_normal'])
        slope1 = int(request.form['slope1'])
        slope2 = int(request.form['slope2'])
        
        data = (age,trestbps,chol,fbs,thalach,exang,oldpeak,sex,cp1,cp2,cp3,restecg_left,restecg_normal,slope1,slope2)
        input_data_as_numpy_array= np.asarray(data, dtype=object)

        # reshape the numpy array as we are predicting for only on instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        my_prediction = model.predict(input_data_reshaped)
        
        return render_template('result.html', prediction=my_prediction)
        
        

if __name__ == '__main__':
	app.run(debug=True)
