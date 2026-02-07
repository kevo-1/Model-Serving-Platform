"""
Dummy Iris Random forest model
"""
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from skl2onnx import to_onnx

print("Loading Iris dataset...")
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("Training RandomForest classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2%}")

print("\nConverting to ONNX format...")
onx = to_onnx(clf, X[:1])

output_path = "models/iris_classifier_v1.onnx"
with open(output_path, "wb") as f:
    f.write(onx.SerializeToString())

print(f"Model saved to {output_path}")

print("\n--- Example Prediction ---")
sample = X_test[0:1]
prediction = clf.predict(sample)
print(f"Input features: {sample[0]}")
print(f"Predicted class: {prediction[0]} ({iris.target_names[prediction[0]]})")
print(f"Actual class: {y_test[0]} ({iris.target_names[y_test[0]]})")