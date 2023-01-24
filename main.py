import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os

# import wordlists from wordlists folder and add them to a list of wordlists to use and turn it as dataframes
working_dir = os.getcwd()
rockyou_file = open(f"{working_dir}\\wordlists\\rockyou.txt",
                    "r", encoding="utf-8", errors='ignore')
# clean the file
nonhuman_file = open(f"{working_dir}\\wordlists\\passwords.txt",
                     "r", encoding="utf-8", errors='ignore')
# import as dataframe and add a column for the label and set it to 1
rockyou_df = pd.DataFrame(rockyou_file.readlines(), columns=['password']).dropna()
rockyou_df['label'] = 1
# import as dataframe and add a column for the label and set it to 0
nonhuman_df = pd.DataFrame(nonhuman_file.readlines(), columns=['password']).dropna()
nonhuman_df['label'] = 0
# UnicodeDecodeError: 'charmap' codec can't decode byte 0x8f in position 3149: character maps to <undefined>
data = pd.concat([rockyou_df, nonhuman_df], ignore_index=True)

# Step 1: Gather a dataset of human-made and non-human made passwords
# Assume the dataset is stored in a pandas dataframe called 'data'
# with 'password' as the column containing the passwords
# and 'label' as the column containing the labels (0 for non-human made and 1 for human-made)

# Step 2: Preprocess the data
# Split the data into training and test sets
# if the number of samples is less than 1, you should gather more data
if len(data) < 1:
    print("Gather more data.")
    exit()
else:
    # split the data into training and test sets
    train_data, test_data = train_test_split(
        data[['password', 'label']], test_size=0.2)

# Step 3: Build a deep learning model using TensorFlow
# Assume we are using a CNN model
# Step 3: Build a deep learning model using TensorFlow
# Assume we are using a RNN

max_len = max([len(p) for p in data['password']])

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(data['password']), output_dim=100, input_length=max_len),
    tf.keras.layers.GRU(units=64, return_sequences=True),
    tf.keras.layers.GRU(units=64),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Step 4: Train the model on the training data and evaluate its performance on the test data
model.fit(train_data['password'], train_data['label'], epochs=10, batch_size=1)

test_loss, test_acc = model.evaluate(test_data['password'], test_data['label'])
print(f'Test accuracy: {test_acc}')

# Step 5: Use the model to classify new passwords
new_password = "mypassword123"
prediction = model.predict(new_password)



if prediction > 0.5:
    print("The password is likely human-made.")
else:
    print("The password is likely non-human made.")
