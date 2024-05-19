import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Machine Learning App')

uploaded_file = st.sidebar.file_uploader("Select CSV", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)
    
    # Select model type
    model_type = st.sidebar.selectbox('Select model type', ['Logistic Regression', 'KNN', 'Random Forest', 'Decision Tree'])
    if model_type == 'Logistic Regression':
        model = LogisticRegression()
    elif model_type == 'KNN':
        k = st.sidebar.number_input("Select K", 1, len(df))
        model = KNeighborsClassifier(n_neighbors=k)
    elif model_type == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100)  
    elif model_type == 'Decision Tree':
        model = DecisionTreeClassifier()
        
    target_variable = st.sidebar.selectbox('Select target variable', df.columns, index=len(df.columns)-1)
    independent_variables = st.sidebar.multiselect('Select independent variables', 
                                                   df.columns, 
                                                   default=list(df.columns.drop(target_variable)))
    visualization_var = st.sidebar.multiselect('Select two variables for visualization', 
                                               df.columns, 
                                               max_selections=2)
    
    # Handle missing values
    missing_values = df.isnull().sum()
    st.sidebar.write('Missing values:', missing_values)

    missing_value_options = ['Drop NA', 'Replace with Mean', 'Replace with Median', 'Replace with Mode']
    selected_option = st.sidebar.selectbox('Select option to handle missing values', missing_value_options)

    if selected_option == 'Drop NA':
        df.dropna(inplace=True)
    elif selected_option == 'Replace with Mean':
        for col in df.columns:
            if df[col].dtype != 'object':
                df[col].fillna(df[col].mean(), inplace=True)
    elif selected_option == 'Replace with Median':
        for col in df.columns:
            if df[col].dtype != 'object':
                df[col].fillna(df[col].median(), inplace=True)
    elif selected_option == 'Replace with Mode':
        for col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

    X = df[independent_variables].copy()
    
    # Label encode categorical features
    le = LabelEncoder()
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = le.fit_transform(X[col])
    
    # Enter data used for prediction
    input = []
    for col in X.columns:
        if df[col].dtype == 'object':
            val = st.selectbox(f"{col}", df[col].unique().tolist())
        else:
            val = st.number_input(f"{col}", float(X[col].min()), float(X[col].max()))
        input.append(val)

if st.sidebar.button('DONE'):
    df.reset_index(inplace=True, drop=True)
    st.title(model_type)

    X_train, X_test, y_train, y_test = train_test_split(X, df[target_variable], test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model.fit(X_train, y_train)

    input_df = pd.DataFrame([input], columns=X.columns)
    input_scaled = scaler.transform(input_df)
    y_predict = model.predict(input_scaled)
    st.subheader("Result")
    st.write(f"{target_variable}: {y_predict[0]}")

    if model_type in ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest']:
        accuracy = accuracy_score(y_test, model.predict(X_test)) * 100
        precision = precision_score(y_test, model.predict(X_test), average='weighted') * 100
        recall = recall_score(y_test, model.predict(X_test), average='weighted') * 100
        f1 = f1_score(y_test, model.predict(X_test), average='weighted') * 100

        st.subheader('Performance Metrics')
        st.write(f"Accuracy: {accuracy:.2f} %")
        st.write(f"Precision: {precision:.2f} %")
        st.write(f"Recall: {recall:.2f} %")
        st.write(f"F1-score: {f1:.2f} %")

        st.subheader('Confusion Matrix')
        conf_matrix = confusion_matrix(y_test, model.predict(X_test))
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        st.pyplot(plt.gcf())

    if model_type == 'Logistic Regression':
        coefficients = model.coef_[0]
        intercept = model.intercept_[0]
        equation = f"{target_variable} = "
        for i, coef in enumerate(coefficients):
            equation += f"({coef:.2f} * {independent_variables[i]}) + "
        equation += f"({intercept:.2f})"
        st.write(equation)
        
    if model_type == 'Decision Tree':
        st.subheader('Decision Tree Visualization')
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 12))
        plot_tree(model, ax=ax, feature_names=independent_variables, filled=True)
        st.pyplot(fig)
        tree_text = export_text(model, feature_names=independent_variables)
        st.text(tree_text)
        
    if model_type == 'Random Forest':
        st.subheader('Random Forest Tree Visualization')
        # Vẽ một cây bất kỳ từ rừng ngẫu nhiên
        tree_index = st.sidebar.number_input("Select Tree Index", 0, len(model.estimators_)-1, step=1)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 12))
        plot_tree(model.estimators_[tree_index], ax=ax, feature_names=independent_variables, filled=True)
        st.pyplot(fig)
        tree_text = export_text(model.estimators_[tree_index], feature_names=independent_variables)
        st.text(tree_text)
        
    if visualization_var:
        plt.figure(figsize=(8, 6))
        plt.scatter(df[visualization_var[0]], df[visualization_var[1]], color='green')
        plt.xlabel(visualization_var[0])
        plt.ylabel(visualization_var[1])
        plt.title(f'{visualization_var[0]} vs {visualization_var[1]}')
        st.pyplot(plt.gcf())

        for col in visualization_var:
            plt.figure(figsize=(8, 6))
            plt.boxplot(df[col])
            plt.xlabel(col)
            plt.ylabel('Values')
            plt.title(f'Box Plot of {col}')
            st.pyplot(plt.gcf())
