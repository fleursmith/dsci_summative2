import requests
from shiny import App, ui, render, reactive, req
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import kagglehub
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import datasets, metrics

### UI ###
app_ui = ui.page_auto(
    ui.panel_title('Disease Risk Calculator'),
    ui.output_text('intro'),
    ui.div(style="height: 1rem;"),
    
    ui.input_numeric(
        'age',
        'Age (Years)',
        value=0,
        min=0,
        max=116,
    ),

    ui.input_numeric(
        'gender',
        'Gender (1 = Male, 0 = Female)',
        value=0,
        min=0,
        max=1,
    ),
    
    ui.input_numeric(
        'smoking_status',
        'Smoker? (1 = Yes, 0 = No)',
        value=0,
        min=0,
        max=1,
    ),

    ui.input_numeric(
        'alcohol_use',
        'Consumes alcohol (1 = Yes, 0 = No)',
        value=0,
        min=0,
        max=1,
    ),

    ui.input_numeric(
        'family_history',
        'Family History (1 = Yes, 0 = No)',
        value=0,
        min=0,
        max=1,
    ),

    ui.input_numeric(
        'sleep_hours',
        'Sleep Hours',
        value=0,
        min=0,
        max=24,
    ),

    ui.div(
        *[
            ui.input_numeric(
                i,
                i.replace("_"," ").title(),
                value=0,
                min=0,
            )
            for i in [
                'length_of_stay','glucose','blood_pressure','bmi',
                'oxygen_saturation','cholesterol','hba1c',
                'triglycerides','physical_activity','diet_score','stress_level'
            ]
        ],
    ),
    ui.input_action_button('calculate_risk','Calculate'),
    ui.div(style="height: 1rem;"),
    ui.output_data_frame('risk_result'),
)

def format_df(df):
        unnecessary_cols = [ "random_notes" , "noise_col" ]
        df.drop(columns=unnecessary_cols, inplace=True)
        
        df.rename(columns={
        "Age": "age",
        "Gender": "gender",
        "Medical Condition": "medical_condition",
        "Glucose": "glucose",
        "Blood Pressure": "blood_pressure",
        "BMI": "bmi",
        "Oxygen Saturation": "oxygen_saturation",
        "LengthOfStay": "length_of_stay",
        "Cholesterol": "cholesterol",
        "Triglycerides": "triglycerides",
        "HbA1c": "hba1c",
        "Smoking": "smoking_status",
        "Alcohol": "alcohol_use",
        "Physical Activity": "physical_activity",
        "Diet Score": "diet_score",
        "Family History": "family_history",
        "Stress Level": "stress_level",
        "Sleep Hours": "sleep_hours"
        }, inplace=True)

        MW_glucose = 180.156
        df['glucose'] = (df['glucose'] * 10) / MW_glucose

        MW_cholesterol = 386.65
        df['cholesterol'] = (df['cholesterol'] * 10) / MW_cholesterol

        MW_triglycerides = 885.7
        df['triglycerides'] = (df['triglycerides'] * 10) / MW_triglycerides

        df['hba1c'] = (df['hba1c'] * 10.929) - 23.5

        df.dropna(subset=['gender'], inplace=True)
        df['gender'] = np.where(df['gender'] == 'Male', 1, 0)
        df['medical_condition'] = df['medical_condition'].fillna('Unknown')

        df = df[df['triglycerides']>=0]
        df = df[df['physical_activity']>=0]
        df = df[df['diet_score']>=0]
        df = df[df['stress_level']>=0]

        return df

def regression_impute(df, target, predictors, add_flag=True):
    not_null = df[df[target].notnull()]
    null = df[df[target].isnull()]
    if null.empty:
        if add_flag and f"{target}_imputed" not in df.columns:
            df[f"{target}_imputed"] = 0
        return None
    
    X_train = not_null[predictors]
    y_train = not_null[target]
    X_test = null[predictors]

    cleaning_model = LinearRegression()
    cleaning_model.fit(X_train, y_train)

    preds = cleaning_model.predict(X_test)

    df.loc[df[target].isnull(), target] = preds

    if add_flag:
        flag_name = f"{target}_imputed"
        df[flag_name] = 0
        df.loc[null.index, flag_name] = 1

        return cleaning_model
    

def complete_imputations(df,targets):
    imputation_models = {}
    
    predictor_cols = [
        'bmi',
        'oxygen_saturation',
        'length_of_stay',
        'cholesterol',
        'triglycerides',
        'hba1c',
        'physical_activity',
        'diet_score',
        'family_history',
        'stress_level',
        'sleep_hours']
    
    for target in targets:
        model = regression_impute(df, target, predictor_cols, add_flag=True)
        imputation_models[target] = model
    
    return df

def condition_identifiers(df,medical_conditions):
    for i in range(len(medical_conditions)):
        condition_name = medical_conditions[i]
        df[condition_name] = np.where(df['medical_condition'] == condition_name, 1, 0)
    
    return df

def multinomial_logistic_classification(df,predictor_variables,categorical_var):
    le = LabelEncoder()
    df['condition_label'] = le.fit_transform(df[categorical_var])

    x = df[predictor_variables]
    y = df['condition_label']

    x_train , x_test, y_train, y_test = train_test_split(
        x, y, random_state=42, test_size=0.2, stratify=y
    ) 

    log_model = LogisticRegression(
        solver='lbfgs',
        max_iter=100,
        random_state=42
    )
    log_model.fit(x_train, y_train)

    def predict(new_values):
        new_values = np.array(new_values)
        pred_proba = log_model.predict_proba(new_values)
        return pred_proba

    return log_model, predict

def fetch_df():
    path = kagglehub.dataset_download("abdallaahmed77/healthcare-risk-factors-dataset")
    raw_df = pd.read_csv(f"{path}/dirty_v3_path.csv")
    df = format_df(raw_df)
    targets = ['age', 'glucose', 'blood_pressure']
    df = complete_imputations(df,targets)
    return df

### SERVER ###
def server(input,output,session):
    @output
    @render.data_frame
    @reactive.event(input.calculate_risk)
    def risk_result():
        df = fetch_df()
        
        predictors = ['age','gender','smoking_status', 'alcohol_use', 'family_history','length_of_stay',
                      'glucose','blood_pressure','bmi','oxygen_saturation','cholesterol','triglycerides',
                      'hba1c','physical_activity','diet_score','stress_level','sleep_hours',]
        inputs = [[input.age(),input.gender(),input.smoking_status(),input.alcohol_use(),input.family_history(),
                   input.length_of_stay(),input.glucose(),input.blood_pressure(),input.bmi(),input.oxygen_saturation(),
                   input.cholesterol(),input.triglycerides(),input.hba1c(),input.physical_activity(),
                   input.diet_score(),input.stress_level(),input.sleep_hours()]]
        

        categorical_var = 'medical_condition'
        model, predictor = multinomial_logistic_classification(df,predictors,categorical_var)
        probabilities = predictor(inputs)
        medical_conditions = df['medical_condition'].unique()

        results = pd.DataFrame({
            "condition": medical_conditions,
            "risk": probabilities[0] * 100
            })
        
        results = results.sort_values('risk',ascending=False).reset_index(drop=True)
        results['risk'] = results['risk'].round(1).astype(str) + '%' 
        return results
    
    @output
    @render.text
    def intro():
        return 'Please enter values for all variables.'


app = App(app_ui,server)