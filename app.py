import pickle
from flask import request, Flask, render_template, redirect, url_for, flash
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import os
from PyPDF2 import PdfReader
import spacy
import fitz 
from sklearn.metrics import roc_curve
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import io
import base64

def extract_pdf_text(file_path):
    text = ""
    try:
        pdf_reader = PdfReader(file_path)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

# ======================================Create app=================================================
app = Flask(__name__)
nlp = spacy.load('en_core_web_sm')
UPLOAD_FOLDER = 'static/resumes'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# ======================================loading models and datasets================================
df = pd.read_csv('notebook/HR_comma_sep.csv')
model = pickle.load(open('models/model.pkl','rb'))
scaler = pickle.load(open('models/scaler.pkl','rb'))

# ======================================dashboard functions========================================
def reading_cleaning(df):
    df.drop_duplicates(inplace=True)
    cols = df.columns.tolist()
    df.columns = [x.lower() for x in cols]

    return df
#-----
df = reading_cleaning(df)
def evaluate_candidate(parsed_text):
    keywords = ['Python', 'SQL', 'Data Science', 'Machine Learning', 'Flask', 'API', 'Django']
    matched_keywords = [word for word in keywords if word.lower() in parsed_text.lower()]
    score = len(matched_keywords)

    if score >= 4:
        match_level = "This candidate is a strong match for the job."
    elif 2 <= score < 4:
        match_level = "This candidate is a moderate match for the job."
    else:
        match_level = "This candidate may not be the best fit for the job."

    return match_level, matched_keywords

def employee_important_info(df):
    # Average satisfaction level
    average_satisfaction = df['satisfaction_level'].mean()
    # Department-wise average satisfaction level
    department_satisfaction = df.groupby('department')['satisfaction_level'].mean()
    # Salary-wise average satisfaction level
    salary_satisfaction = df.groupby('salary')['satisfaction_level'].mean()

    # Employees who left
    left_employees = len(df[df['left'] == 1])
    # Employees who stayed
    stayed_employees = len(df[df['left'] == 0])

    return average_satisfaction, department_satisfaction, salary_satisfaction, left_employees, stayed_employees

def plots(df, col):
    values = df[col].unique()
    plt.figure(figsize=(15, 10))

    explode = [0.1 if len(values) > 1 else 0] * len(values)
    plt.pie(df[col].value_counts(), explode=explode, startangle=40, autopct='%1.1f%%', shadow=True)
    labels = [f'{value} ({col})' for value in values]
    plt.legend(labels=labels, loc='upper right', fontsize=12)
    plt.title(f"Distribution of {col}", fontsize=16, fontweight='bold')

    plt.savefig('static/' + col + '.png')
    plt.close()

def distribution(df, col):
    values = df[col].unique()
    plt.figure(figsize=(15, 10))
    sns.countplot(x=df[col], hue='left', palette='Set1', data=df)
    labels = [f"{val} ({col})" for val in values]
    plt.legend(labels=labels, loc="upper right", fontsize=12)
    plt.title(f"Distribution of {col}", fontsize=16, fontweight='bold')
    plt.xticks(rotation=90)
    plt.savefig('static/' + col + '_distribution.png')
    plt.close()

def comparison(df, x, y):
    plt.figure(figsize=(15, 10))
    sns.barplot(x=x, y=y, hue='left', data=df, ci=None)
    plt.title(f'{x} vs {y}', fontsize=16, fontweight='bold')
    plt.savefig('static/' + 'comparison.png')
    plt.close()

def corr_with_left(df):
    df_encoded = pd.get_dummies(df)
    correlations = df_encoded.corr()['left'].sort_values()[:-1]
    colors = ['skyblue' if corr >= 0 else 'salmon' for corr in correlations]
    plt.figure(figsize=(15, 10))
    correlations.plot(kind='barh', color=colors)
    plt.title('Correlation with Left', fontsize=16, fontweight='bold')
    plt.xlabel('Correlation', fontsize=14, fontweight='bold')
    plt.ylabel('Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('static/correlation.png')
    plt.close()

def histogram(df, col):
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))  # Corrected line

    # Plot the first histogram
    sns.histplot(data=df, x=col, hue='left', bins=20, ax=axes[0])
    axes[0].set_title(f"Histogram of {col}", fontsize=16, fontweight='bold')

    # Plot the second histogram
    sns.kdeplot(data=df, x='satisfaction_level', y='last_evaluation', hue='left', fill=True, ax=axes[1])
    axes[1].set_title("Kernel Density Estimation", fontsize=16, fontweight='bold')

    plt.tight_layout()  # Adjust the layout to prevent overlapping
    plt.savefig('static/' + col + '_histogram.png')
    plt.close()


#=====================prediction function====================================================
def prediction(sl_no, gender, ssc_p, hsc_p, degree_p, workex, etest_p, specialisation, mba_p):
    data = {
    'sl_no': [sl_no],
    'gender': [gender],
    'ssc_p': [ssc_p],
    'hsc_p': [hsc_p],
    'degree_p': [degree_p],
    'workex': [workex],
    'etest_p': [etest_p],
    'specialisation': [specialisation],
    'mba_p': [mba_p]
    }
    data = pd.DataFrame(data)
    data['gender'] = data['gender'].map({'Male':1,"Female":0})
    data['workex'] = data['workex'].map({"Yes":1,"No":0})
    data['specialisation'] = data['specialisation'].map({"Mkt&HR":1,"Mkt&Fin":0})
    scaled_df = scaler.transform(data)
    result = model.predict(scaled_df).reshape(1, -1)
    return result[0]


# routes===================================================================

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/index')
def home():
    return render_template("index.html")
@app.route('/job')
def job():
    return render_template('job.html')


@app.route('/ana')
def ana():
    average_satisfaction, department_satisfaction, salary_satisfaction, left_employees, stayed_employees= employee_important_info(df)
    plots(df, 'left')
    plots(df, 'salary')
    plots(df, 'number_project')
    plots(df, 'department')

    distribution(df, 'salary')
    distribution(df, 'department')

    comparison(df, 'department', 'satisfaction_level')

    corr_with_left(df)

    histogram(df, 'satisfaction_level')

    # Convert Series objects to dictionaries
    department_satisfaction= department_satisfaction.to_dict()
    salary_satisfaction = salary_satisfaction.to_dict()
    return render_template('ana.html', df=df.head(),average_satisfaction=average_satisfaction,
                           department_satisfaction=department_satisfaction,salary_satisfaction=salary_satisfaction,
                           left_employees=left_employees,stayed_employees=stayed_employees)



#prediction===============================================================
@app.route('/parse_resume', methods=['GET', 'POST'])
def parse_resume():
    if request.method == 'POST':
        if 'resume' not in request.files:
            return "No file part"
        
        file = request.files['resume']
        
        if file.filename == '':
            return "No selected file"
        
        if file:
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Extract text
            extracted_text = extract_pdf_text(file_path)

            # Evaluate candidate
            evaluation_result, matched_keywords = evaluate_candidate(extracted_text)

            # --- Simulate probability for ROC ---
            total_keywords = 7  # You defined 7 job-related keywords
            score = len(matched_keywords)
            confidence = score / total_keywords

            # --- Simulate actual label (manually or mock: assume good fit if ≥4 keywords) ---
            actual = 1 if score >= 4 else 0

            # --- Create ROC-like plot for this one sample ---
            fpr = [0, 0.1, 1]
            tpr = [0, confidence, 1]
            auc_score = trapezoid(tpr, fpr)

            # --- Plot and save ROC ---
            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Simulated ROC Curve for Resume')
            plt.legend(loc='lower right')
            plt.grid()

            roc_path = os.path.join('static', 'resume_roc.png')
            plt.savefig(roc_path)
            plt.close()

            # Preview extracted text
            preview_text = extracted_text[:500]

            return render_template(
                'parse_result.html',
                evaluation_result=evaluation_result,
                matched_keywords=matched_keywords,
                preview_text=preview_text,
                auc_score=round(auc_score, 2),
                roc_path=roc_path
            )

    return render_template('parse_resume.html')

@app.route('/login')
def login():
    return render_template('Login.html')
@app.route("/placement",methods=['POST','GET'])
def placement():
    if request.method == 'POST':
        sl_no = request.form['sl_no']
        gender = request.form['gender']
        ssc_p = request.form['ssc_p']
        hsc_p = request.form['hsc_p']
        degree_p = request.form['degree_p']
        workex = request.form['workex']
        etest_p = request.form['etest_p']
        specialisation = request.form['specialisation']
        mba_p = request.form['mba_p']

        result = prediction(sl_no, gender, ssc_p, hsc_p, degree_p, workex, etest_p, specialisation, mba_p)

        if result == 1:
            pred = "Placed"
            rec = "We recommend you that this is the best candidate for you business"
            return render_template('job.html', result=pred, rec=rec)

        else:
            pred = "Not Placed"
            rec = "We recommend you that this is not the best candidate for your business"
            return render_template('job.html', result=pred,rec=rec)

    return redirect(url_for('index'))

# ========================python main===================================================
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve, auc

    # Prepare dataset (assuming df is cleaned and ready)
    temp_df = df.copy()
    temp_df['salary'] = temp_df['salary'].map({'low': 0, 'medium': 1, 'high': 2})
    temp_df['department'] = temp_df['department'].astype('category').cat.codes

    X = temp_df.drop('left', axis=1)
    y = temp_df['left']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Predict probabilities using your model
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    print(f"\nROC AUC Score: {roc_auc:.4f}")

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("static/roc_auc_curve.png")  # Save if needed
    plt.savefig("static/roc_auc_curve.png")
    print("ROC Curve saved as 'static/roc_auc_curve.png'")

    app.run(debug=True)
