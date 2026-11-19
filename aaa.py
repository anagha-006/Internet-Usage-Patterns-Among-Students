mport pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Check whether dataset is loaded
try:
    df
except NameError:
    from google.colab import files
    uploaded = files.upload()
    filename = list(uploaded.keys())[0]
    df = pd.read_csv(filename)

# Clean column names
df.columns = (df.columns.str.replace('\n', ' ', regex=False)
              .str.strip().str.replace(r'\s+', ' ', regex=True))

# Define input features
features = [
    "How many hours do you spend on the internet per day?",
    "What is your main purpose for using the internet?",
    "Which platform do you use most frequently?",
    "How frequently do you use the internet for academic purposes?"
]

# Define target variable
target = "Does internet usage sometimes distract you from your studies?"

# Check required columns
missing = [col for col in features + [target]
           if col not in df.columns]

if missing:
    raise ValueError("Columns not found: " + str(missing))

# Remove rows with missing target values
data = df.dropna(subset=[target]).copy()

# Separate input and output
X = data[features].astype("object")
y = data[target].astype(str).str.strip()

# Remove classes with only one response
counts = y.value_counts()
valid_classes = counts[counts >= 2].index

data = data[y.isin(valid_classes)]
X = data[features].astype("object")
y = data[target].astype(str).str.strip()

if y.nunique() < 2:
    raise ValueError("Not enough target categories to train the model.")

# Preprocessing
preprocessor = ColumnTransformer([
    ("categorical",
     Pipeline([
         ("imputer", SimpleImputer(strategy="most_frequent")),
         ("encoder", OneHotEncoder(handle_unknown="ignore"))
     ]),
     features)
])

# Create Random Forest model
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model.fit(X_train, y_train)

# Predict results
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("MODEL TRAINED SUCCESSFULLY!")
print("\nModel: Random Forest Classifier")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

print("\nConfusion Matrix:")
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(7, 5))
plt.imshow(cm, cmap="Blues")
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.xticks(range(len(model.classes_)), model.classes_,
           rotation=30)
plt.yticks(range(len(model.classes_)), model.classes_)

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.show()