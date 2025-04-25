import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

df = pd.read_csv('train.csv')
 
label_encoder = LabelEncoder()

categorical_columns = ['department', 'region', 'education', 'gender', 'recruitment_channel']
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Check and fill missing values
for col in df.columns:
    if df[col].dtype == 'object' or len(df[col].unique()) < 20:
        df[col].fillna(df[col].mode()[0], inplace=True)  # Fill with mode for categorical data
    else:
        df[col].fillna(df[col].mean(), inplace=True)  # Fill with mean for numerical data

X = df.drop('is_promoted', axis=1)  
y = df['is_promoted']  

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_test_df = X_test.copy()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the artificial neural network model
model = Sequential()

model.add(Dense(units=128, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))

optimizer = Adam(learning_rate=0.001)  # Set the learning rate
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, batch_size=40, validation_data=(X_test, y_test))

plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Training and Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")
print(f"Test loss: {loss:.4f}")

predictions_prob = model.predict(X_test) 

predictions = (predictions_prob > 0.5).astype(int)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, predictions_prob)  
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(12, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC Curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Loss curve
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Training and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Save predictions to a CSV file
predictions_df = pd.DataFrame({
    'employee_id': df.loc[X_test_df.index, 'employee_id'],
    'predicted_is_promoted': predictions.flatten()
})

predictions_df.to_csv('predictions.csv', index=False)

print("Predictions have been written to predictions.csv.")

