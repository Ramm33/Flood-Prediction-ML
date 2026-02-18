from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Point Flask to custom template directory
app = Flask(__name__, template_folder='pages')

# Load your ML model
model = joblib.load('model_k.pkl')

# Route to show the form
@app.route('/home')
def home():
    return redirect(url_for('home.html'))



@app.route('/home')
def form_page():
    return render_template('home.html')

@app.route('/guidelines')
def guideline():
    return render_template('guidelines.html')


@app.route('/stats')
def statistic():
    return render_template('stats.html')


@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            'Temperature': float(request.form.get('Temperature')),
            'Humidity': float(request.form.get('Humidity')),
            'Cloud_cover': float(request.form.get('Cloud_cover')),
            'Annual_rainfall': float(request.form.get('Annual_rainfall')),
            'Jan-Feb_rainfall': float(request.form.get('Jan_Feb_rainfall')),
            'Mar-May_rainfall': float(request.form.get('Mar_May_rainfall')),
            'Jun-Sep_rainfall': float(request.form.get('Jun_Sep_rainfall')),
            'Oct-Dec_rainfall': float(request.form.get('Oct_Dec_rainfall')),
            'Avg_june_rainfall': float(request.form.get('Avg_june_rainfall')),
            'Sub_surafce_water_level': float(request.form.get('Sub_surface_water_level'))
        }

        df = pd.DataFrame([input_data])

        # Make sure all required columns are present
        for col in model.feature_names_in_:
            if col not in df.columns:
                df[col] = 0
        df = df[model.feature_names_in_]

        prediction = model.predict(df)[0]

        if prediction == 0:
            return render_template('nosevere.html', prediction='No possibility of severe flood')
        else:
            return render_template('severe.html', prediction='Possibility of Severe Flood')

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
