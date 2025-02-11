import numpy as np
import pandas as pd
import tensorflow as tf
import os
import pickle
import random
import config_file as cfg
import tensorflow.keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from modules import ENCODER, AddCLSToken
from data_generator_clinical import DataGenerator3D_Gene
#from lr_scheduler import CosineDecayRestarts
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras import Model, optimizers, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Concatenate, Input, Dense, Lambda, Embedding, Dropout, LayerNormalization, MultiHeadAttention
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from keras import backend as K



# check for gpu
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



# Set seeds for reproducible results
seed = 1
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

# Set fold number
fold = 0


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-reweigh', type=int, help='yes (1) or no (0)')
parser.add_argument('-pop_struc', type=int, help='yes (1) or no (0)')
args = parser.parse_args()


# -----------------------------------
#            PARAMETERS
# -----------------------------------

# Print config parameters
print('TRANSFORMER PARAMETERS')
for x in cfg.transformer_params:
    print(x, ':', cfg.transformer_params[x])
print('\nTRAINING PARAMETERS')
for x in cfg.train_params:
    print(x, ':', cfg.train_params[x])
print('\nDATA GENERATOR PARAMETERS')
for x in cfg.params:
    print(x, ':', cfg.params[x])


# -----------------------------------
#            IMPORT DATA
# -----------------------------------

# # Import the clinical metadata
# print("[INFO] importing datasets...")
# clinical_names = [*cfg.continuous_features, *cfg.categorical_features['names']]
# metadata_csv = pd.read_csv(cfg.params['clinicalFile'], index_col='PATNO')[clinical_names]
# #mRs_scores = pd.read_csv(cfg.params['clinicalFile'], index_col='PATNO')[cfg.target_feature['name']]
# IDs = metadata_csv.index.values.tolist()
#
# # Preprocess continuous variables
# scaler = MinMaxScaler()  # perform min-max scaling for each variable. Features are now in the range [0, 1]  # should I do Z-score normalization instead?
# continuous_preprocessed = scaler.fit_transform(metadata_csv[cfg.continuous_features])
# continuous_preprocessed = pd.DataFrame(continuous_preprocessed, index=IDs, columns=[*cfg.continuous_features], dtype='float32')  # convert numpy array back to pandas dataframe
# metadata_csv[cfg.continuous_features] = continuous_preprocessed[cfg.continuous_features]  # replace the original values with the preprocessed values
# metadata_csv.to_csv("metadata_scaled.csv",sep=",")

# No need to preprocess categorical variables, as they will go through an Embedding layer later on

# Importing patient dictionary
#print("[INFO] loading patient dictionary...")
#with open(cfg.params['dictFile_3D'], 'rb') as output:
#    partition = pickle.load(output)

# Data augmentation: horizontal/vertical flip and Gaussian noise?

# Calling training generator
#train_generator = DataGenerator3D_CTP(partition['training'][fold], metadata_csv, mRs_scores, shuffle=True, **cfg.params)
#val_generator = DataGenerator3D_CTP(partition['testing'][fold], metadata_csv, mRs_scores, shuffle=True, **cfg.params)
#test_generator = DataGenerator3D_CTP(partition['testing'][fold], metadata_csv, mRs_scores, shuffle=False, **cfg.params)

# Import the clinical metadata
print("[INFO] importing datasets...")
# clinical_names = [*cfg.continuous_features, *cfg.categorical_features['names']]
# metadata_csv = pd.read_csv(cfg.params['clinicalFile'], index_col='PATNO')[clinical_names]
# #mRs_scores = pd.read_csv(cfg.params['clinicalFile'], index_col='PATNO')[cfg.target_feature['name']]
# IDs = metadata_csv.index.values.tolist()

# Preprocess continuous variables
 # perform min-max scaling for each variable. Features are now in the range [0, 1]  # should I do Z-score normalization instead?
# continuous_preprocessed = scaler.fit_transform(train[cfg.continuous_features])
# continuous_preprocessed = pd.DataFrame(continuous_preprocessed, columns=[*cfg.continuous_features], dtype='float32')  # convert numpy array back to pandas dataframe
# train[cfg.continuous_features] = continuous_preprocessed[cfg.continuous_features]  # replace the original values with the preprocessed values


# scaler = MinMaxScaler()
# fn_train = cfg.params['train_partition']
# train = pd.read_csv(fn_train, sep=",")
# # perform min-max scaling for each variable. Features are now in the range [0, 1]  # should I do Z-score normalization instead?
# continuous_preprocessed = scaler.fit_transform(train[cfg.continuous_features])
# continuous_preprocessed = pd.DataFrame(continuous_preprocessed, columns=[*cfg.continuous_features], dtype='float32')  # convert numpy array back to pandas dataframe
# train[cfg.continuous_features] = continuous_preprocessed[cfg.continuous_features]  # replace the original values with the preprocessed values
# #train.to_csv("train_tmp.csv",sep=",")
# IDs_list = train['PATNO'].to_numpy()
# train_IDs = IDs_list
# train_generator = DataGenerator3D_Gene(list_IDs = train_IDs, metadata_pd = train, csv_wPaths = fn_train,  shuffle=False , **cfg.params)
#
# fn_val = cfg.params['val_partition']
# val = pd.read_csv(fn_val, sep=",")
# continuous_preprocessed = scaler.fit_transform(val[cfg.continuous_features])
# continuous_preprocessed = pd.DataFrame(continuous_preprocessed, columns=[*cfg.continuous_features], dtype='float32')  # convert numpy array back to pandas dataframe
# val[cfg.continuous_features] = continuous_preprocessed[cfg.continuous_features]  # replace the original values with the preprocessed values
# #val.to_csv("val_tmp.csv",sep=",")
# IDs_list = val['PATNO'].to_numpy()
# val_IDs = IDs_list
# val_generator = DataGenerator3D_Gene(list_IDs = val_IDs, metadata_pd = val, csv_wPaths = fn_val,  shuffle=False, **cfg.params)
#
# fn_test = cfg.params['test_partition']
# test = pd.read_csv(fn_test, sep=",")
# continuous_preprocessed = scaler.fit_transform(test[cfg.continuous_features])
# continuous_preprocessed = pd.DataFrame(continuous_preprocessed, columns=[*cfg.continuous_features], dtype='float32')  # convert numpy array back to pandas dataframe
# test[cfg.continuous_features] = continuous_preprocessed[cfg.continuous_features]
# #test.to_csv("test_tmp.csv",sep=",")
# IDs_list = test['PATNO'].to_numpy()
# test_IDs = IDs_list
# test_generator = DataGenerator3D_Gene(test_IDs, test, fn_test, **cfg.params)

# scaler = MinMaxScaler()
# fn_all = cfg.params['all']
# full = pd.read_csv(fn_all,sep=",")
# continuous_preprocessed = scaler.fit_transform(full[cfg.continuous_features])
# continuous_preprocessed = pd.DataFrame(continuous_preprocessed, columns=[*cfg.continuous_features], dtype='float32')  # convert numpy array back to pandas dataframe
# full[cfg.continuous_features] = continuous_preprocessed[cfg.continuous_features]  # replace the original values with the preprocessed values# continuous_preprocessed = scaler.fit_transform(train[cfg.continuous_features])
# IDs_list = full['PATNO'].to_numpy()
# full.to_csv("full_tmp.csv",sep=",")



# train val split here !!!
# Need to split on both the PD/Control grouping but also on the gene grouping
fn_train = cfg.params['train']
train = pd.read_csv(fn_train,sep=",")
train['stratify_col'] = train['Group_numeric'].astype(str) + '_' + train['Gene_Subtype_NoPRKN'].astype(str)
#train, val = train_test_split(train, test_size=0.25, stratify=train['stratify_col'], random_state=42)
#train, val = train_test_split(train, test_size=0.25, stratify=train['stratify_col'], random_state=42)

# train.to_csv("train_tmp.csv",sep=",")
# val.to_csv("val_tmp.csv",sep=",")
# test.to_csv("test_tmp.csv",sep=",")

fn_test = cfg.params['test_partition']
test = pd.read_csv(fn_test,sep=",")

# IDs
# train_IDs = train['PATNO'].to_numpy()
test_IDs = test['PATNO'].to_numpy()
# val_IDs = val['PATNO'].to_numpy()


# Count the occurrences of 'Gene_subgroup' in the training set
#train_gene_subgroup_counts = train['Gene_Subtype_NoPRKN'].value_counts()
#print("Train Gene Subgroup Counts:\n", train_gene_subgroup_counts)

# Count the occurrences of 'Gene_subgroup' in the test set
#val_gene_subgroup_counts = val['Gene_Subtype_NoPRKN'].value_counts()
#print("\nVal Gene Subgroup Counts:\n", val_gene_subgroup_counts)

# Count the occurrences of 'Gene_subgroup' in the test set
test_gene_subgroup_counts = test['Gene_Subtype_NoPRKN'].value_counts()
print("\nTest Gene Subgroup Counts:\n", test_gene_subgroup_counts)


# train_generator = DataGenerator3D_Gene(list_IDs = train_IDs, metadata_pd = train, csv_wPaths = fn_train,  shuffle=False , **cfg.params)
test_generator = DataGenerator3D_Gene(test_IDs, test, fn_test, **cfg.params)
# val_generator = DataGenerator3D_Gene(list_IDs = val_IDs, metadata_pd = val, csv_wPaths = fn_train,  shuffle=False, **cfg.params)

########################## K-Fold
n_splits = 5


# StratifiedKFold for splitting the training data into train/val sets in a stratified manner
skf = StratifiedKFold(n_splits=n_splits)




# Print a batch from the training generator
#X_train, y_train = train_generator.__getitem__(0)
#print("Training Data Batch - Features (X_train):")
#print(X_train[0].shape)
#print(len(X_train[1]))
#print(len(X_train[2]))
#print("Training Data Batch - Labels (y_train):")
#print(y_train.shape)

# for batch in test_generator:
#     print(batch)
#     break



# -----------------------------------
#             CALLBACKS
# -----------------------------------

# ---------- LR SCHEDULER -----------
print("[INFO] using a 'cosine' learning rate decay with periodic 'restarts'...")
n_cycles = 16
t_mul = 1
m_mul = 0.75
#total_steps = len(partition['training'][fold])//cfg.params['batch_size']*cfg.train_params['n_epochs']
# total_steps = len(train_IDs)//cfg.params['batch_size']*cfg.train_params['n_epochs']
# sch = CosineDecayRestarts(initial_learning_rate=cfg.train_params['learning_rate'], first_decay_steps=total_steps//int(n_cycles), t_mul=float(t_mul), m_mul=float(m_mul), alpha=0.0)
# learning_rates = [sch(step) for step in range(total_steps)]  # Plot learning rates
# plt.figure()
# plt.plot(learning_rates)

# Simpler scheduler as keep having nan for cosine decay restarts lr scheduler

# initial_learning_rate = cfg.train_params['learning_rate']  # Your initial learning rate
# decay_rate = 0.96  # Factor by which the learning rate decreases
# decay_steps = total_steps // 10  # How often to apply decay (e.g., every 10% of total steps)
#
# sch = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=initial_learning_rate,
#     decay_steps=decay_steps,
#     decay_rate=decay_rate,
#     staircase=True  # If True, the learning rate decays in discrete steps
#)


# -----------------------------------
#            BUILD MODEL
# -----------------------------------

print("[INFO] building model for fold {} ...".format(fold))

# ------ (A.1) IMAGE ENCODER -------
base_network = ENCODER.build(reg=l2(0.00005), shape=cfg.params['dim_3D'])
absolute_diff = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))  # compute the absolute difference between tensors

# Create CTP encoders
inputs, outputs, skip = [], [], []
for w in range(cfg.params['timepoints']):
    # Create inputs and get model outputs
    i = Input(shape=(*cfg.params['dim_3D'], 1))
    o = base_network(i)
# Append results
inputs.append(i)
outputs.append(o[0])

# Concatenate latent vectors and skip connections
imaging_encoded = Concatenate(axis=1)(outputs)


# ----- (A.2) CLINICAL ENCODER -----
# Import metadata and create embeddings
# Embed categorical data with no ordinal relationship, using a fixed dim for all features
embed_dim = 1573 #392 #2016 #384
C_inputs = []
C_embedding_outputs = []
for i in range(len(cfg.categorical_features['names'])):
    categorical_i = Input(shape=(1, ),  name=cfg.categorical_features['names'][i]) #dtype='int32',
    categorical_i_ = Embedding(input_dim=cfg.categorical_features['categories'][i], output_dim=embed_dim)(categorical_i)  # output_dim=cfg.transformer_params['projection_dim_3D']
    C_inputs.append(categorical_i)
    C_embedding_outputs.append(categorical_i_)
categorical_inputs = Concatenate(axis=1)(C_embedding_outputs)

# Numerical feature tokenizer: Continuous inputs are transformed to tokens (embeddings) instead of used as-is
N_inputs = []
N_embedding_outputs = []
for feature_name in cfg.continuous_features:
    continuous_i = Input(shape=(1, ), dtype='float32', name=feature_name)
    continuous_i_ = Dense(embed_dim, activation='relu')(continuous_i)
    N_inputs.append(continuous_i)
    N_embedding_outputs.append(tf.expand_dims(continuous_i_, axis=1))
continuous_inputs = Concatenate(axis=1)(N_embedding_outputs)

# Concatenate numerical (a.k.a. continuous) and categorical features
metadata_encoded = Concatenate(axis=1)([continuous_inputs, categorical_inputs])



# -------- (B) MULTIMODAL FUSION --------


# ------- SELF-ATTENTION METADATA -------
# Add a positional encoding
sequence_length = metadata_encoded.shape[1]  # inputs are of shape: (batch_size, n_features, filter_size)
positions = tf.range(start=0, limit=sequence_length, delta=1)
embedded_positions = Embedding(input_dim=sequence_length, output_dim=embed_dim)(positions)
metadata_encoded += embedded_positions

# Add CLStoken
metadata_encoded = AddCLSToken()(metadata_encoded)

# Explicit TransformerEncoder layer
num_layers = 1
num_heads = 8
dropout_rate = 0.2
inputTensor = metadata_encoded
proj_input_metadata = None  # TODO: fix naming conventions so that for loop works
for _ in range(num_layers):
    attention_output, attention_scores = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate)(inputTensor, inputTensor, return_attention_scores=True)
    proj_input_metadata = LayerNormalization(epsilon=1e-6)(inputTensor + attention_output)  # res connection provides a direct path for the gradient, while the norm maintains a reasonable scale for outputs


# ------- SELF-ATTENTION IMAGING -------
sequence_length = imaging_encoded.shape[1]  # inputs are of shape: (batch_size, n_features, filter_size)
positions = tf.range(start=0, limit=sequence_length, delta=1)
embedded_positions = Embedding(input_dim=sequence_length, output_dim=embed_dim)(positions)
imaging_encoded += embedded_positions

# Add CLStoken
imaging_encoded = AddCLSToken()(imaging_encoded)

# Explicit TransformerEncoder layer
num_layers = 1
num_heads = 8
dropout_rate = 0.2
inputTensor = imaging_encoded
proj_input_imaging = None  # TODO: fix naming conventions so that for loop works
for _ in range(num_layers):
    attention_output, attention_scores = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate)(inputTensor, inputTensor, return_attention_scores=True)
    proj_input_imaging = LayerNormalization(epsilon=1e-6)(inputTensor + attention_output)  # res connection provides a direct path for the gradient, while the norm maintains a reasonable scale for outputs


# ------- CROSS-ATTENTION METADATA -------
# Co-attention: Query - Imaging; Key & Value - Metadata  TODO: for loop for num_layers
A_attention_output, A_attention_scores = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate)(proj_input_imaging, proj_input_metadata, return_attention_scores=True)  # A_attention_output of shape [None, 33, 384]
A_proj_input = LayerNormalization(epsilon=1e-6)(proj_input_imaging + A_attention_output)
A_proj_output = Sequential([Dense(embed_dim, activation=tf.nn.gelu), Dropout(dropout_rate), ])(A_proj_input)
A_x = LayerNormalization(epsilon=1e-6)(A_proj_input + A_proj_output)  # shape [None, 33, 384]

# Reduce output sequence through pooling layer
# cross_imaging = GlobalMaxPooling1D()(A_x)  # aggregate via max-pooling
cross_imaging = A_x[:, 0]  # extracting cls token

# Co-attention: Query - Metadata; Key & Value - Imaging  TODO: for loop for num_layers
B_attention_output, B_attention_scores = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate)(proj_input_metadata, proj_input_imaging, return_attention_scores=True)  # B_attention_output of shape [None, 10, 384]
B_proj_input = LayerNormalization(epsilon=1e-6)(proj_input_metadata + B_attention_output)
B_proj_output = Sequential([Dense(embed_dim, activation=tf.nn.gelu), Dropout(dropout_rate), ])(B_proj_input)
B_x = LayerNormalization(epsilon=1e-6)(B_proj_input + B_proj_output)  # shape [None, 10, 384]

# Reduce output sequence through pooling layer
# cross_metadata = GlobalMaxPooling1D()(B_x)  # aggregate via max-pooling
cross_metadata = B_x[:, 0]  # extracting cls token

# Concatenate CLS tokens
features = Concatenate()([cross_imaging, cross_metadata])


# ------- (C) OUTCOME PREDICTION ---------
#Stack an MLP before the last layer
mlp_hidden_units_factors = [2, 1]  # MLP hidden layer units, as factors of the number of inputs
mlp_hidden_units = [factor * features.shape[-1] for factor in mlp_hidden_units_factors]
for units in mlp_hidden_units:
    features = Dense(units, activation='relu')(features)
    features = Dropout(0.2)(features)

# Classify outputs
#mRS_prediction = Dense(1, activation='sigmoid')(features)  # should I try with softmax?
PD_prediction = Dense(1, activation='sigmoid')(features)

# # Build model and plot its summary
# PD_model = Model(inputs=[inputs, C_inputs, N_inputs], outputs=PD_prediction) #C_inputs
# print("[INFO] printing model summary...")
# PD_model.summary()
# tf.keras.utils.plot_model(mRS_model, show_shapes=True, rankdir="LR")


def build_PD_model(inputs, C_inputs, N_inputs, PD_prediction):
    # Build model and plot its summary
    PD_model = Model(inputs=[inputs, C_inputs, N_inputs], outputs=PD_prediction) #C_inputs
    print("[INFO] printing model summary...")
    PD_model.summary()

    return PD_model



# -----------------------------------
#            TRAIN MODEL
# -----------------------------------

# Define hyperparameters to tune
# param_grid = {
#     'learning_rate': [1e-5, 1e-4],                 # Different learning rates
#     'epochs': [100],                           # Number of epochs
#     'batch_size': [1, 2, 4],
# }

param_grid = {
    'learning_rate': [1e-5, 1e-4, 1e-3],                 # Different learning rates
    'epochs': [10, 100, 150],                           # Number of epochs
    'batch_size': [2, 4],
}

PD_model = build_PD_model(inputs, C_inputs, N_inputs, PD_prediction)

# Define loss - Focal loss: improved version of x-entropy loss that tries to handle class imbalance by down-weighting easy examples and focusing on hard ones
# focal_loss = BinaryFocalLoss(gamma=2)  # reduction='mean', from_logits=False
binary_crossentropy = tf.keras.losses.BinaryCrossentropy()

# Define optimizer
#adam_optimizer = optimizers.Adam(learning_rate=sch) #optimizers.Adam(learning_rate=1e-5)
# adam_optimizer = optimizers.Adam(learning_rate=cfg.train_params['learning_rate'], decay=cfg.train_params['weight_decay'])

# Compile model
#PD_model.compile(loss=binary_crossentropy, optimizer=adam_optimizer, metrics=['accuracy'])

est_score = 0
best_params = {}

# Loop through all combinations of hyperparameters
for learning_rate in param_grid['learning_rate']:
    for epochs in param_grid['epochs']:
        for batch_size in param_grid['batch_size']:
            print(f"Training model with learning_rate={learning_rate}, epochs={epochs}, batch_size={batch_size}")

            fold_scores = []
            results = []

            # Perform k-fold cross-validation
            for fold, (train_idx, val_idx) in enumerate(skf.split(train, train['stratify_col'])):
                results = []
                print(f"Fold {fold + 1}/{n_splits}")

                # Create train and validation sets for this fold
                train_fold = train.iloc[train_idx]
                train_IDs = train_fold['PATNO'].to_numpy()

                val_fold = train.iloc[val_idx]
                val_IDs = val_fold['PATNO'].to_numpy()

                # Initialize data generators
                train_generator = DataGenerator3D_Gene(list_IDs=train_IDs, metadata_pd=train_fold, csv_wPaths=fn_train, shuffle=False, **cfg.params)
                val_generator = DataGenerator3D_Gene(list_IDs=val_IDs, metadata_pd=val_fold, csv_wPaths=fn_train, shuffle=False, **cfg.params)

                # Create and compile model
                PD_model = build_PD_model(inputs, C_inputs, N_inputs, PD_prediction)
                binary_crossentropy = tf.keras.losses.BinaryCrossentropy()
                total_steps = len(train_fold) // batch_size * epochs
                sch = CosineDecayRestarts(initial_learning_rate=learning_rate, first_decay_steps=total_steps // int(n_cycles), t_mul=float(t_mul), m_mul=float(m_mul), alpha=0.0)
                adam_optimizer = optimizers.Adam(learning_rate=sch)
                PD_model.compile(loss=binary_crossentropy, optimizer=adam_optimizer, metrics=['accuracy'])

                # Train the model
                reweigh = args.reweigh
                print("Reweigh choice: " + str(reweigh))

                class_weights = {0: 1.45, 1: 3.22}

                if reweigh:
                    print("Reweigh Was Chosen")
                    H = PD_model.fit(x=train_generator, validation_data=val_generator, epochs=epochs, batch_size=batch_size, class_weight= class_weights)
                    # Save the model for this fold
                    model_weights_path = f'{cfg.params["resultsPath"]}model_weights_lr{learning_rate}_epochs{epochs}_batch{batch_size}_fold{fold + 1}_reweigh_TrainTestVal.h5'
                    PD_model.save_weights(model_weights_path)
                    print(f"[INFO] Model weights saved for fold {fold + 1}")

                    # Evaluate the model on the validation set
                    val_predictions, val_true_labels, patient_ids = [], [], []
                    for i in range(len(val_generator)):
                        X_val, y_true = val_generator[i]

                        patient_id_batch = val_generator.get_current_IDs() 
                        y_pred = PD_model.predict(X_val)
                        y_pred_binary = (y_pred >= 0.5).astype(int)

                        del X_val

                        val_predictions.append(y_pred)
                        val_true_labels.append(y_true)
                        patient_ids.append(patient_id_batch)

                        for j in range(len(patient_id_batch)):
                            patient_id = patient_id_batch[j]
                            true_label = y_true[j]
                            predicted_label = y_pred[j]
                            predicted_binary = y_pred_binary[j]

                            # Append data to results list
                            results.append({
                                'Patient ID': patient_id,
                                'True Label': true_label,
                                'Predicted Label': predicted_label,
                                'Predicted Binary': predicted_binary,
                            })

                    results_df = pd.DataFrame(results)
                    results_df.to_csv(f'{cfg.params["resultsPath"]}predictions_val_fold_{fold + 1}_lr_{learning_rate}_epochs_{epochs}_batchSize_{batch_size}_trainTestVal.csv', index=False)
                    print("Results of val set for fold " +str(fold + 1)+ " saved")
                    results.clear()

                    val_predictions = np.concatenate(val_predictions, axis=0)
                    val_true_labels = np.concatenate(val_true_labels, axis=0)
                    y_pred_binary = (val_predictions >= 0.5).astype(int)

                    # Confusion matrix for this fold
                    cm_binary = confusion_matrix(val_true_labels, y_pred_binary)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
                                yticklabels=['Class 0', 'Class 1'])
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    plt.title(f'Confusion Matrix for Fold {fold + 1}')
                    cm_filename = f"{cfg.params['resultsPath']}confusion_matrix_lr{learning_rate}_epochs{epochs}_batch{batch_size}_fold_{fold + 1}_reweigh_trainTestVal.svg"
                    plt.savefig(cm_filename, format='svg', dpi=300)
                    plt.close()
                    print(f"[INFO] Confusion matrix saved for fold {fold + 1}")

                    # Append validation accuracy to fold scores
                    score = PD_model.evaluate(val_generator)
                    fold_scores.append(score[1])  # Assuming score[1] is accuracy

                    # Plot the loss curves for this fold
                    plt.figure(figsize=(10, 6))
                    plt.plot(H.history['loss'], label='Training Loss')
                    plt.plot(H.history['val_loss'], label='Validation Loss')
                    plt.title(f'Loss Curves for Fold {fold + 1}')
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.legend()
                    loss_curve_filename = f"{cfg.params['resultsPath']}loss_curves_lr{learning_rate}_epochs{epochs}_batch{batch_size}_fold_{fold + 1}_reweigh_trainTestVal.svg"
                    plt.savefig(loss_curve_filename, format='svg', dpi=300)
                    plt.close()
                    print(f"[INFO] Loss curve saved for fold {fold + 1}")

                    del PD_model
                    del train_generator
                    del val_generator
                    tf.keras.backend.clear_session()

                else:
                    print("No Reweigh")
                    H = PD_model.fit(x=train_generator, validation_data=val_generator, epochs=epochs,
                                     batch_size=batch_size)
                    # Save the model for this fold
                    model_weights_path = f'{cfg.params["resultsPath"]}model_weights_lr{learning_rate}_epochs{epochs}_batch{batch_size}_fold{fold + 1}.h5'
                    PD_model.save_weights(model_weights_path)
                    print(f"[INFO] Model weights saved for fold {fold + 1}")

                    # Evaluate the model on the validation set
                    val_predictions, val_true_labels = [], []
                    for i in range(len(val_generator)):
                        X_val, y_true = val_generator[i]
                        y_pred = PD_model.predict(X_val)
                        val_predictions.append(y_pred)
                        val_true_labels.append(y_true)

                    val_predictions = np.concatenate(val_predictions, axis=0)
                    val_true_labels = np.concatenate(val_true_labels, axis=0)
                    y_pred_binary = (val_predictions >= 0.5).astype(int)

                    # Confusion matrix for this fold
                    cm_binary = confusion_matrix(val_true_labels, y_pred_binary)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
                                yticklabels=['Class 0', 'Class 1'])
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    plt.title(f'Confusion Matrix for Fold {fold + 1}')
                    cm_filename = f"confusion_matrix_lr{learning_rate}_epochs{epochs}_batch{batch_size}_fold{fold + 1}.svg"
                    plt.savefig(cm_filename, format='svg', dpi=300)
                    plt.close()
                    print(f"[INFO] Confusion matrix saved for fold {fold + 1}")

                    # Append validation accuracy to fold scores
                    score = PD_model.evaluate(val_generator)
                    fold_scores.append(score[1])  # Assuming score[1] is accuracy

                    # Plot the loss curves for this fold
                    plt.figure(figsize=(10, 6))
                    plt.plot(H.history['loss'], label='Training Loss')
                    plt.plot(H.history['val_loss'], label='Validation Loss')
                    plt.title(f'Loss Curves for Fold {fold + 1}')
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.legend()
                    loss_curve_filename = f"loss_curves_lr{learning_rate}_epochs{epochs}_batch{batch_size}_fold{fold + 1}.svg"
                    plt.savefig(loss_curve_filename, format='svg', dpi=300)
                    plt.close()
                    print(f"[INFO] Loss curve saved for fold {fold + 1}")



            # Calculate the average score across all folds
            avg_score = sum(fold_scores) / len(fold_scores)
            print(f"Average validation score for learning_rate={learning_rate}, epochs={epochs}, batch_size={batch_size}: {avg_score}")

            # Update the best parameters if this combination is better
            if avg_score > est_score:
                est_score = avg_score
                best_params = {
                    'learning_rate': learning_rate,
                    'epochs': epochs,
                    'batch_size': batch_size
                }

print("Best parameters found: ", best_params)
print("Best validation score: ", est_score)

# # Plot the training loss and validation loss curves
# plt.figure(figsize=(10, 6))
# plt.plot(H.history['loss'], label='Training Loss')
# plt.plot(H.history['val_loss'], label='Validation Loss')
# plt.title('Loss Curves')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# # Save the figure as an SVG file
# plt.savefig('loss_curves_200Epochs_dropped70PD.svg', format='svg', dpi=300)  # Save as SVG format
# plt.close()
#
# # Save model weights
# PD_model.save_weights(
#     cfg.params['resultsPath'] + f'{fold}_multimodal_cross-attention_model_weights_changedArch_1573_dropped70PD')
# print("[INFO] model saved to disk")
#
# # Save training history
# np.save(cfg.params['resultsPath'] + f'{fold}_clinical_trainHistoryDic_changedArch_1573_dropped70PD.npy', H.history)
#
# # Load model weights
# PD_model.load_weights(
#     cfg.params['resultsPath'] + f'{fold}_multimodal_cross-attention_model_weights_changedArch_1573_dropped70PD')
#
# total_steps = len(train_IDs) // batch_size * epochs
# predictions = PD_model.predict(test_generator, steps=int(len(test_IDs) / cfg.params['batch_size']))
# np.savez_compressed(
#     cfg.params['resultsPath'] + f'{fold}_multimodal_cross-attention_changedArch_1573_dropped70PD',
#     pred=predictions)


# -----------------------------------
#          EVALUATE MODEL
# -----------------------------------

#print(test_IDs)
# total_steps = len(train_IDs)//cfg.params['batch_size']*cfg.train_params['n_epochs']
# predictions = PD_model.predict(test_generator, steps=int(len(test_IDs)/cfg.params['batch_size']))
# np.savez_compressed(cfg.params['resultsPath'] + f'{fold}_multimodal_cross-attention_changedArch_1573_dropped70PD', pred=predictions)
#
# print(f'Fold {fold} - Predictions')
# print(predictions)


# ###############################
# # CONFUSION MATRIX
# ############################
# predictions, true_labels = [], []
# for i in range(len(test_generator)):
#     X, y_true = test_generator[i]  # Get batch
#     y_pred = PD_model.predict(X)  # Get predictions
#     predictions.append(y_pred)
#     true_labels.append(y_true)
#
#
# predictions = np.concatenate(predictions, axis=0)
# true_labels = np.concatenate(true_labels, axis=0)
#
# y_true_binary = true_labels
# y_pred_binary = (predictions >= 0.5).astype(int)
#
# cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
#
#
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix for Binary Classification')
# plt.savefig('PD_Control_ConfusionMatrix_changedArch_1573_dropped70PD.svg', format='svg', dpi=300)  # Save as SVG format
# plt.close()
#
# ##########################
# # CATEGORICAL CONFUSION MATRIX
# ###########################
#
# # Extract true and predicted categorical data
# y_true_categorical = np.concatenate([X[1] for X, y in test_generator], axis=0)  # Categorical true labels
# y_pred_categorical = np.concatenate([PD_model.predict(X)[1] for X, y in test_generator],
#                                     axis=0)  # Predicted categorical labels
#
# # Iterate over categorical variables for confusion matrices
#
# for i, feature_name in cfg.categorical_features['names']:
#     y_true_cat = y_true_categorical[:, i]  # True labels for categorical feature
#     y_pred_cat = np.argmax(y_pred_categorical[:, i], axis=1)  # Predicted labels for categorical feature
#
#     cm_cat = confusion_matrix(y_true_cat, y_pred_cat)
#
#     # Plot confusion matrix for each categorical variable
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm_cat, annot=True, fmt='d', cmap='Blues')
#     plt.title(f'Confusion Matrix for {feature_name}')
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.savefig(f'PD_Control_ConfusionMatrix_{feature_name}_changedArch_1573_dropped70PD.svg', format='svg')  # Save as SVG format
#     plt.close()



