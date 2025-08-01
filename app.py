#pip install streamlit
#pip install pandas
#pip install sklearn

# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

# LOAD DATA
df = pd.read_csv('C:\\Users\\ariba\\Desktop\\7th semster\\diabetes_prediction-master\\diabetes.csv')

# HEADINGS
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

st.subheader('Visualization')
st.bar_chart(df)
# X AND Y DATA
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# USER INPUT FUNCTION
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    user_report_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

# MODEL
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)

y_proba = rf.predict_proba(x_test)[:, 1]  # Probabilities for ROC curve


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Confusion matrix and display
st.subheader('Confusion Matrix')
y_pred = rf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

fig_cm = plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot(fig_cm)


# VISUALISATIONS
st.title('Visualised Patient Report')

st.subheader('Outcome Distribution in Dataset')
fig_outcome = plt.figure()
sns.countplot(x='Outcome', data=df, palette='pastel')
plt.title('Distribution of Diabetic vs Non-Diabetic Patients')
plt.xticks([0, 1], ['No Diabetes', 'Diabetes'])
plt.xlabel('Outcome')
plt.ylabel('Count')
st.pyplot(fig_outcome)


# COLOR FUNCTION
color = 'blue' if user_result[0] == 0 else 'red'

# Plot: Age vs Pregnancies
st.header('Pregnancy count Graph (Others vs Yours)')
fig_preg = plt.figure()
sns.scatterplot(x='Age', y='Pregnancies', data=df, hue='Outcome', palette='Greens')
sns.scatterplot(x=user_data['Age'], y=user_data['Pregnancies'], s=150, color=color)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 20, 2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_preg)

# Plot: Age vs Glucose
st.header('Glucose Value Graph (Others vs Yours)')
fig_glucose = plt.figure()
sns.scatterplot(x='Age', y='Glucose', data=df, hue='Outcome', palette='magma')
sns.scatterplot(x=user_data['Age'], y=user_data['Glucose'], s=150, color=color)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 220, 10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_glucose)

# Plot: Age vs Blood Pressure
st.header('Blood Pressure Value Graph (Others vs Yours)')
fig_bp = plt.figure()
sns.scatterplot(x='Age', y='BloodPressure', data=df, hue='Outcome', palette='Reds')
sns.scatterplot(x=user_data['Age'], y=user_data['BloodPressure'], s=150, color=color)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 130, 10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bp)

# Plot: Age vs Skin Thickness
st.header('Skin Thickness Value Graph (Others vs Yours)')
fig_st = plt.figure()
sns.scatterplot(x='Age', y='SkinThickness', data=df, hue='Outcome', palette='Blues')
sns.scatterplot(x=user_data['Age'], y=user_data['SkinThickness'], s=150, color=color)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 110, 10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_st)

# Plot: Age vs Insulin
st.header('Insulin Value Graph (Others vs Yours)')
fig_i = plt.figure()
sns.scatterplot(x='Age', y='Insulin', data=df, hue='Outcome', palette='rocket')
sns.scatterplot(x=user_data['Age'], y=user_data['Insulin'], s=150, color=color)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 900, 50))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_i)

# Plot: Age vs BMI
st.header('BMI Value Graph (Others vs Yours)')
fig_bmi = plt.figure()
sns.scatterplot(x='Age', y='BMI', data=df, hue='Outcome', palette='rainbow')
sns.scatterplot(x=user_data['Age'], y=user_data['BMI'], s=150, color=color)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 70, 5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bmi)

# Plot: Age vs Diabetes Pedigree Function
st.header('DPF Value Graph (Others vs Yours)')
fig_dpf = plt.figure()
sns.scatterplot(x='Age', y='DiabetesPedigreeFunction', data=df, hue='Outcome', palette='YlOrBr')
sns.scatterplot(x=user_data['Age'], y=user_data['DiabetesPedigreeFunction'], s=150, color=color)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 3, 0.2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_dpf)

# OUTPUT
st.subheader('Your Report: ')
output = 'You are not Diabetic' if user_result[0] == 0 else 'You are Diabetic'
st.title(output)
st.subheader('Accuracy: ')
st.write(f"{accuracy_score(y_test, rf.predict(x_test)) * 100:.2f}%")

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)


from sklearn.metrics import roc_curve, auc

st.subheader('ROC Curve & AUC')

# Probability predictions
y_proba = rf.predict_proba(x_test)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC
fig_roc = plt.figure()
plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc, color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
st.pyplot(fig_roc)
