from flask import Flask, render_template, request, abort
import joblib
import pandas as pd

app = Flask(__name__)

# Load models
stacking_clf = joblib.load('model/stacked_model.pkl')
kmeans_disease = joblib.load('model/kmeans_disease.pkl')
kmeans_severity = joblib.load('model/kmeans_severity.pkl')
scaler = joblib.load('model/scaler.pkl')

# Define features and mappings
features = [
    'Ovarian cysts', 'Bloating', 'Infertility', 'Abnormal uterine bleeding', 'Vaginal Pain/Pressure', 
    'Sharp / Stabbing pain', 'Pain after Intercourse', 'Prolonged/Excessive bleeding', 'Frequent Menstrual clots', 
    'Painful ovulation', 'Fatigue / Chronic fatigue', 'Irritable Bowel Syndrome (IBS)', 'Long menstruation', 
    'Painful urination', 'Pain / Chronic pain', 'Painful bowel movements', 'Irregular / Missed periods', 
    'Pelvic pain', 'Abdominal Cramps during Intercourse', 'Stomach cramping', 
    'Painful / Burning pain during sex (Dyspareunia)', 'Menstrual pain (Dysmenorrhea)'
]

severity_levels = {
    0: 'No Endometriosis',
    1: 'Minimal',
    2: 'Mild',
    3: 'Moderate',
    4: 'Severe'
}

cluster_to_disease = {
    0: 'Polycystic Ovary Syndrome (PCOS)',
    1: 'Uterine Fibroids',
    2: 'Pelvic Inflammatory Disease (PID)',
    3: 'Chronic Fatigue Syndrome (CFS)',
    4: 'Interstitial Cystitis',
    5: 'Adenomyosis'
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/front-end', methods=['GET', 'POST'])
def front_end():
    if request.method == 'POST':
        return render_template('result.html')
    return render_template('front-end.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            input_data = []
            for feature in features:
                value = request.form.get(feature)
                if value is None:
                    return "Missing input for feature: {}".format(feature), 400
                input_data.append(float(value))

            input_df = pd.DataFrame([input_data], columns=features)

           
            scaled_input = scaler.transform(input_df)

            
            severity_prediction = stacking_clf.predict(input_df)[0]

            if severity_prediction == 1:
                
                predicted_cluster = kmeans_disease.predict(scaled_input)[0]
                recommended_diseases = [disease for cluster, disease in cluster_to_disease.items() if cluster == predicted_cluster]

                
                predicted_severity = severity_levels[kmeans_severity.predict(scaled_input)[0]]
            else:
                recommended_diseases = ['Unknown Disease']
                predicted_severity = 'No Endometriosis'

            return render_template('result.html', severity=predicted_severity,
                                   recommended_diseases=recommended_diseases, severity_prediction=severity_prediction)
        except Exception as e:
            return "Error occurred: {}".format(e), 500
    return abort(400)

if __name__ == '__main__':
    app.run(debug=True)
