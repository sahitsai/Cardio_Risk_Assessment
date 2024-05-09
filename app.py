from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Function to load the model
def load_model():
    df = pd.read_csv("Cardiovascular_Disease_Dataset.csv")
    

    
    X = df.drop(["patientid","target"], axis=1)  # drop 'patientid' as it's an identifier
    y = df["target"]
    
    
    # Model Building and Evaluation
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Define the models
    models = {
        "Logistic Regression": LogisticRegression(),
        "SVM": SVC(probability=True),
        "Neural Network": MLPClassifier(max_iter=1000)
    }
    
    # Train and evaluate models
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        # Predict on the test set
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    
        # Print the metrics
        print(f"{name}\nAccuracy: {accuracy} \nPrecision: {precision} \nRecall: {recall} \nF1: {f1} \nROC AUC: {roc_auc}\n")
    
    # Cross-Validation with Logistic Regression
    logreg = models["Logistic Regression"]
    cross_val_scores = cross_val_score(logreg, X, y, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
    print("Logistic Regression - Cross-validated Accuracy: ", np.mean(cross_val_scores))
    
    # Cross-Validation with SVM
    SVM_m = models["SVM"]
    cross_val_scores = cross_val_score(SVM_m, X, y, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
    print("SVM - Cross-validated Accuracy: ", np.mean(cross_val_scores))
    
    
    # Cross-Validation with Logistic Regression
    mlp = models["Neural Network"]
    cross_val_scores = cross_val_score(mlp, X, y, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
    print("Neural Network - MLP - Cross-validated Accuracy: ", np.mean(cross_val_scores))


    return logreg


@app.route('/')
def home():
    return render_template('home.html')




@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Retrieve user inputs
        inputs = {key: request.form[key] for key in request.form.keys()}

        # Load the trained model and label encoder
        SVM_model= load_model()
       

        
        
        # Convert string inputs to appropriate data types
        converted_inputs = {}
        
        converted_inputs['age'] = int(inputs['age'])
        converted_inputs['gender'] = int(inputs['gender'])
        converted_inputs['chestpain'] = int(inputs['chestpain'])
        converted_inputs['restingBP'] = int(inputs['restingBP'])
        converted_inputs['serumcholestrol'] = int(inputs['serumcholestrol'])
        converted_inputs['fastingbloodsugar'] = int(inputs['fastingbloodsugar'])
        converted_inputs['restingrelectro'] = int(inputs['restingrelectro'])
        converted_inputs['maxheartrate'] = int(inputs['maxheartrate'])
        converted_inputs['exerciseangia'] = int(inputs['exerciseangia'])
        converted_inputs['oldpeak'] = float(inputs['oldpeak'])
        converted_inputs['slope'] = int(inputs['slope'])
        converted_inputs['noofmajorvessels'] = int(inputs['noofmajorvessels'])
        

        # Convert inputs to DataFrame for label encoding
        df_inputs = pd.DataFrame([converted_inputs])
        print(df_inputs)
        
        
        # Predict using the model
        prediction = SVM_model.predict(df_inputs)[0]
        print(prediction)
       
        if prediction == 0:
            prediction_label="Normal"
        else:
            prediction_label="CVD Risk"
            
        return render_template('predict.html', prediction=prediction_label)

    return render_template('predict.html', prediction=None)


@app.route('/contact')
def contact():
    return render_template('contactus.html')


if __name__ == '__main__':
    app.run()
