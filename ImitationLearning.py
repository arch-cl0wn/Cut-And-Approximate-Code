# Import Required Libraries
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

# Generate Demonstration Data (Placeholder)


def generate_demo_data():
    demo_states = np.random.rand(1000, 100)
    demo_actions = np.random.rand(1000, 5)
    return train_test_split(demo_states, demo_actions, test_size=0.2)

# Define Neural Network Model for Policy


def define_policy_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(100,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(5, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Pre-train the Model using Demonstration Data


def pre_train_model(model, train_states, train_actions, val_states, val_actions):
    model.fit(train_states, train_actions, epochs=10,
              validation_data=(val_states, val_actions))

# Main function to run pre-training


def run_pre_training():
    train_states, val_states, train_actions, val_actions = generate_demo_data()
    model = define_policy_model()
    pre_train_model(model, train_states, train_actions,
                    val_states, val_actions)
    return model  # Return the pre-trained model
