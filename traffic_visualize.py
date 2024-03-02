import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform

# Load Data
df = pd.read_csv('Traffic.csv')

le = LabelEncoder()
df['Traffic Situation'] = le.fit_transform(df['Traffic Situation'])

# Split
X = df[['CarCount', 'BikeCount', 'BusCount', 'TruckCount']]
y = df['Traffic Situation']

# One-hot encode 
y = to_categorical(y)

# Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# NaN?
if df.isnull().values.any() or np.isinf(df.values).any():
    raise ValueError("Data contains NaN or infinite values which will result in NaN loss during training.")

# Neural Network
initializer = GlorotUniform(seed=42)
model = Sequential()
model.add(Dense(32, input_dim=X_train_scaled.shape[1], activation='relu', kernel_initializer=initializer))
model.add(Dense(16, activation='relu', name='latent_space', kernel_initializer=initializer))
model.add(Dense(y.shape[1], activation='softmax', kernel_initializer=initializer))  # Output layer with one neuron per class

# Compile
opt = Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train 
model.fit(X_train_scaled, y_train, epochs=100, batch_size=10, verbose=1)

# Evaluate 
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Model accuracy: {accuracy*100:.2f}%")

# Latent space
latent_model = Model(inputs=model.input, outputs=model.get_layer('latent_space').output)
latent_output = latent_model.predict(X_test_scaled)

# t-SNE
tsne = TSNE(n_components=3, random_state=42)
latent_tsne = tsne.fit_transform(latent_output)

# Stats
means = np.mean(latent_tsne, axis=0)
std_dev = np.std(latent_tsne, axis=0)
std_error = std_dev / np.sqrt(latent_tsne.shape[0])

print(f"Means: {means}")
print(f"Standard Deviations: {std_dev}")
print(f"Standard Errors: {std_error}")

# 3D Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(latent_tsne[:,0], latent_tsne[:,1], latent_tsne[:,2], c=np.argmax(y_test, axis=1))
plt.show()

# GUI 
def make_prediction():
    try:
        car_count = int(entry_car.get())
        bike_count = int(entry_bike.get())
        bus_count = int(entry_bus.get())
        truck_count = int(entry_truck.get())

        scaled_input = scaler.transform([[car_count, bike_count, bus_count, truck_count]])

        prediction = model.predict(scaled_input)
        traffic_situation = le.inverse_transform([np.argmax(prediction)])

        messagebox.showinfo("Prediction", f"Traffic situation is expected to be: {traffic_situation[0]}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Create the main window
root = tk.Tk()
root.title("Traffic Prediction")

# Create and place input fields in the window
tk.Label(root, text="Car Count:").grid(row=0, column=0)
entry_car = tk.Entry(root)
entry_car.grid(row=0, column=1)

tk.Label(root, text="Bike Count:").grid(row=1, column=0)
entry_bike = tk.Entry(root)
entry_bike.grid(row=1, column=1)

tk.Label(root, text="Bus Count:").grid(row=2, column=0)
entry_bus = tk.Entry(root)
entry_bus.grid(row=2, column=1)

tk.Label(root, text="Truck Count:").grid(row=3, column=0)
entry_truck = tk.Entry(root)
entry_truck.grid(row=3, column=1)

# Create and place the prediction button in the window
predict_button = tk.Button(root, text="Predict Traffic Situation", command=make_prediction)
predict_button.grid(row=4, column=0, columnspan=2)

# Run the main loop
root.mainloop()
