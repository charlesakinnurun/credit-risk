# %% [markdown]
# Import the neccessary libraries

# %% [markdown]
# import the neccessary libraries

# %%
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# %% [markdown]
# Data Loading and Initial Inspection

# %%
print("-----Data Loading and Initial Inspection-----")

# Load the customer data, which contains the target variable 'label'ArithmeticError
try:
    df = pd.read_csv("customer_data.csv")
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: 'customer_data.csv' not found. Please ensure the file is in the correct path")
    exit()

df

# %% [markdown]
# Data Preprocessing

# %%
# Check for missing values
df_missing = df.isnull().sum()
print("Missing Values")
print(df_missing)

# %%
# Check for duplicated rows
df_duplicated = df.duplicated().sum()
print("Dupliacted Rows")
print(df_duplicated)

# %%
# Drop rows with missing values
#df = df.dropna(inplace=True)

# %% [markdown]
# Feature Engineering

# %%
# Identify the target variable (y) and feature set(X)
X = df[["fea_1","fea_2","fea_3","fea_4","fea_5","fea_6","fea_7","fea_8","fea_9","fea_10"]]
y = df["label"]

# Identify column type for different preprocessing steps
numerical_features = ['fea_2', 'fea_4']  # Removed 'fea_11' if not present in X
categorical_features = ['fea_1', 'fea_3', 'fea_5', 'fea_6', 'fea_7', 'fea_8', 'fea_9', 'fea_10']

# Handle missing values (Imputation)
# Fill missing values in numerical columns with the median
# Fill missing values in categorical columns with the mode (most frequent value)
for col in numerical_features:
    df[col].fillna(df[col].median(), inplace=True)
for col in categorical_features:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Define preprocessing steps using ColumnTransformer for cleaner code and consistency
# Numerical Pipeline: Impute missing values with the median and then scale (StandardScaler)
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical Pipeline: Impute missing values with the most frequent values and then encode
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Combine pipelines using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ],
    remainder="passthrough" # Keep other columns (if any) as they are
)

# Apply processing to the features
X_processed = preprocessor.fit_transform(X)

# %% [markdown]
# Data Splitting

# %%
# Split data into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X_processed,y,random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# %% [markdown]
# Visualization before training

# %%
# Visualization 1: Target Variable Distribution (Checking the class imbalance)
plt.figure(figsize=(6,4))
sns.countplot(x=y,palette="viridis")
plt.title("Distibution of Target Variable (Lable 0 vs 1)",fontsize=14)
plt.xlabel("Label (0 = Negative, 1 = Positive)")
plt.ylabel("Count")
plt.grid(axis="y",alpha=0.5)
plt.show()

# %%
# Visualization 2: Feature Distibution (Example of a numerical feature)
plt.figure(figsize=(8,5))
sns.histplot(df["fea_11"],kde=True,bins=30,color='darkorange')
plt.title("Distribution of Feature fea_11",fontsize=14)
plt.xlabel("fea_11 Value")
plt.ylabel("Frequency")
plt.grid(axis="y",alpha=0.5)
plt.show()

# %% [markdown]
# Model Training and Comparison

# %%
print("-----Model Training and Comparison-----")

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(solver="liblinear",random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC(random_state=42,probability=True),
    "Decision Tree": DecisionTreeClassifier(random_state=42,max_depth=5), # Limiting the dept of Overfitting
    "Random Forest": RandomForestClassifier(n_estimators=100,random_state=42,max_depth=10)
}


results = {} # Dictionary to store the model

# Loop through models, train them and evaluate performance
for name,model in models.items():
    print(f"Training {name}.....")

    # Train the model using the preprocessed training data
    model.fit(X_train,y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy score
    accuracy = accuracy_score(y_test,y_pred)
    results[name] = accuracy

    # Print the performance report
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy:.4f}")

    # Print the classification report
    print(classification_report(y_test,y_pred))

    # Plot the confusion matrix of one of the models (Random Forest)
    if name == "Random Forest":
        plt.figure(figsize=(6,5)),
        cm = confusion_matrix(y_test,y_pred)
        sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
                    xticklabels=["Predicted 0","Predicted 1"],
                    yticklabels=["Actual 0","Actual 1"]

        )
        plt.title(f"Confusion Matrix for {name}",fontsize=14)
        plt.ylabel("Actual Label")
        plt.xlabel("Predicted Label")
        plt.show()

# %%
# Determine the best model
best_model_name = max(results,key=results.get)
best_model = models[best_model_name]

print("-----Best Model Selection-----")
print(f"The best performing model based on Accuracy is: {best_model_name} with Accuracy: {results[best_model_name]:.4f}")

# %% [markdown]
# Visualization after training

# %%
print("-----Visualization After Training-----")

# Create a DataFrame for easy plotting of results
results_df = pd.DataFrame(results.items(),columns=["Model","Accuracy"])
results_df = results_df.sort_values(by="Accuracy",ascending=False)

# Bar Chart of Model Accuracies
plt.figure(figsize=(10,6))
sns.barplot(x="Accuracy",y="Model",data=results_df,palette="crest")
plt.title("Comparison of Classification Model Accuracies",fontsize=16)
plt.xlabel("Accuracy Score",fontsize = 12)
plt.ylabel("Model",fontsize = 12)
plt.xlim(0.5,1.0) # Set a reasonable x-limit for classification accuracy
plt.grid(axis="x",alpha=0.5)

# Highlight the best model
for index, row in results_df.iterrows():
    plt.text(row['Accuracy'] + 0.005, index, f"{row['Accuracy']:.4f}", color='black', ha="left", va="center")

plt.show()


