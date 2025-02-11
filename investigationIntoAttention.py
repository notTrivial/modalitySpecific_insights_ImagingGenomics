import numpy as np
import pandas as pd
import tensorflow as tf
import os
import pickle
import random
import config_file_allVariants as cfg
import tensorflow.keras.backend as K
from sklearn.preprocessing import MinMaxScaler

#from Code.pheno_binaryClassifier_testSetInvesigations import predictions_binary
#from Code.sfcn_pd import test_IDs
from modules import ENCODER, AddCLSToken, DECODER
from data_generator_clinical import DataGenerator3D_Gene
# from lr_scheduler import CosineDecayRestarts
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras import Model, optimizers, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Concatenate, Input, Dense, Lambda, Embedding, Dropout, LayerNormalization, \
    MultiHeadAttention
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
import SimpleITK as sitk


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
parser.add_argument('-fold', type=int, help='numeric fold number starting at 0')
parser.add_argument("-train_file", type=str, required=True, help="Path to the training file")
parser.add_argument("-val_file", type=str, required=True, help="Path to the validation file")
parser.add_argument("-test_file", type=str, required=True, help="Path to the test file")
parser.add_argument("-results_path", type=str, required=True, help="Path to save the results")
parser.add_argument("-model", type=str, required=True, help="Path to model")

args = parser.parse_args()

# Use the arguments in your configuration
params = {
    'train_partition': args.train_file,
    'val_partition': args.val_file,
    'test_partition': args.test_file,
    'resultsPath': args.results_path,
}

# args = parser.parse_args()
print("Parameters received:", params)
cfg.params.update(params)

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
# print("[INFO] loading patient dictionary...")
# with open(cfg.params['dictFile_3D'], 'rb') as output:
#    partition = pickle.load(output)

# Data augmentation: horizontal/vertical flip and Gaussian noise?

# Calling training generator
# train_generator = DataGenerator3D_CTP(partition['training'][fold], metadata_csv, mRs_scores, shuffle=True, **cfg.params)
# val_generator = DataGenerator3D_CTP(partition['testing'][fold], metadata_csv, mRs_scores, shuffle=True, **cfg.params)
# test_generator = DataGenerator3D_CTP(partition['testing'][fold], metadata_csv, mRs_scores, shuffle=False, **cfg.params)

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
fn_train = cfg.params['train_partition']
train = pd.read_csv(fn_train, sep=",")

fn_val = cfg.params['val_partition']
val = pd.read_csv(fn_val, sep=",")
# train['stratify_col'] = train['Group_numeric'].astype(str) + '_' + train['Gene_Subtype_NoPRKN'].astype(str)
# train, val = train_test_split(train, test_size=0.25, stratify=train['stratify_col'], random_state=42)
# train, val = train_test_split(train, test_size=0.25, stratify=train['stratify_col'], random_state=42)

# train.to_csv("train_tmp.csv",sep=",")
# val.to_csv("val_tmp.csv",sep=",")
# test.to_csv("test_tmp.csv",sep=",")

fn_test = cfg.params['test_partition']
test = pd.read_csv(fn_test, sep=",")

# IDs
train_IDs = train['PATNO'].to_numpy()
test_IDs = test['PATNO'].to_numpy()
val_IDs = val['PATNO'].to_numpy()

# Count the occurrences of 'Gene_subgroup' in the training set
# train_gene_subgroup_counts = train['Gene_Subtype_NoPRKN'].value_counts()
# print("Train Gene Subgroup Counts:\n", train_gene_subgroup_counts)

# Count the occurrences of 'Gene_subgroup' in the test set
# val_gene_subgroup_counts = val['Gene_Subtype_NoPRKN'].value_counts()
# print("\nVal Gene Subgroup Counts:\n", val_gene_subgroup_counts)

# Count the occurrences of 'Gene_subgroup' in the test set
# test_gene_subgroup_counts = test['Gene_Subtype_NoPRKN'].value_counts()
# print("\nTest Gene Subgroup Counts:\n", test_gene_subgroup_counts)


# train_generator = DataGenerator3D_Gene(list_IDs = train_IDs, metadata_pd = train, csv_wPaths = fn_train,  shuffle=False , **cfg.params)
# test_generator = DataGenerator3D_Gene(test_IDs, test, fn_test, **cfg.params)
# val_generator = DataGenerator3D_Gene(list_IDs = val_IDs, metadata_pd = val, csv_wPaths = fn_train,  shuffle=False, **cfg.params)

########################## K-Fold
n_splits = 5

# StratifiedKFold for splitting the training data into train/val sets in a stratified manner
skf = StratifiedKFold(n_splits=n_splits)

# Print a batch from the training generator
# X_train, y_train = train_generator.__getitem__(0)
# print("Training Data Batch - Features (X_train):")
# print(X_train[0].shape)
# print(len(X_train[1]))
# print(len(X_train[2]))
# print("Training Data Batch - Labels (y_train):")
# print(y_train.shape)

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
# total_steps = len(partition['training'][fold])//cfg.params['batch_size']*cfg.train_params['n_epochs']
total_steps = len(train_IDs)//cfg.params['batch_size']*cfg.train_params['n_epochs']
sch = CosineDecayRestarts(initial_learning_rate=cfg.train_params['learning_rate'], first_decay_steps=total_steps//int(n_cycles), t_mul=float(t_mul), m_mul=float(m_mul), alpha=0.0)
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
# )


# -----------------------------------
#            BUILD MODEL
# -----------------------------------

print("[INFO] building model for fold {} ...".format(fold))

# ------ (A.1) IMAGE ENCODER -------
base_network = ENCODER.build(reg=l2(0.00005), shape=cfg.params['dim_3D'])
absolute_diff = Lambda(
    lambda tensors: K.abs(tensors[0] - tensors[1]))  # compute the absolute difference between tensors

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
embed_dim = 1573  # 392 #2016 #384
C_inputs = []
C_embedding_outputs = []
for i in range(len(cfg.categorical_features['names'])):
    categorical_i = Input(shape=(1,), name=cfg.categorical_features['names'][i])  # dtype='int32',
    categorical_i_ = Embedding(input_dim=cfg.categorical_features['categories'][i], output_dim=embed_dim)(
        categorical_i)  # output_dim=cfg.transformer_params['projection_dim_3D']
    C_inputs.append(categorical_i)
    C_embedding_outputs.append(categorical_i_)
categorical_inputs = Concatenate(axis=1)(C_embedding_outputs)

# Numerical feature tokenizer: Continuous inputs are transformed to tokens (embeddings) instead of used as-is
# N_inputs = []
# N_embedding_outputs = []
# for feature_name in cfg.continuous_features:
#     continuous_i = Input(shape=(1, ), dtype='float32', name=feature_name)
#     continuous_i_ = Dense(embed_dim, activation='relu')(continuous_i)
#     N_inputs.append(continuous_i)
#     N_embedding_outputs.append(tf.expand_dims(continuous_i_, axis=1))
# continuous_inputs = Concatenate(axis=1)(N_embedding_outputs)

# Concatenate numerical (a.k.a. continuous) and categorical features
metadata_encoded = categorical_inputs  # Concatenate(axis=1)([continuous_inputs, categorical_inputs])

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
    attention_output, attention_scores = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim,
                                                            dropout=dropout_rate)(inputTensor, inputTensor,
                                                                                  return_attention_scores=True)
    proj_input_metadata = LayerNormalization(epsilon=1e-6)(
        inputTensor + attention_output)  # res connection provides a direct path for the gradient, while the norm maintains a reasonable scale for outputs

# ------- SELF-ATTENTION IMAGING -------
print(imaging_encoded.shape)
sequence_length = imaging_encoded.shape[1]  # inputs are of shape: (batch_size, n_features, filter_size)
print(sequence_length)
print('LENGHT OF SEQ')
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
    attention_output, attention_scores = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim,
                                                            dropout=dropout_rate)(inputTensor, inputTensor,
                                                                                  return_attention_scores=True)
    proj_input_imaging = LayerNormalization(epsilon=1e-6)(
        inputTensor + attention_output)  # res connection provides a direct path for the gradient, while the norm maintains a reasonable scale for outputs

# ------- CROSS-ATTENTION METADATA -------
# Co-attention: Query - Imaging; Key & Value - Metadata  TODO: for loop for num_layers
A_attention_output, A_attention_scores = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim,
                                                            dropout=dropout_rate)(proj_input_imaging,
                                                                                  proj_input_metadata,
                                                                                  return_attention_scores=True)  # A_attention_output of shape [None, 33, 384]
A_proj_input = LayerNormalization(epsilon=1e-6)(proj_input_imaging + A_attention_output)
A_proj_output = Sequential([Dense(embed_dim, activation=tf.nn.gelu), Dropout(dropout_rate), ])(A_proj_input)
A_x = LayerNormalization(epsilon=1e-6)(A_proj_input + A_proj_output)  # shape [None, 33, 384]

# Reduce output sequence through pooling layer
# cross_imaging = GlobalMaxPooling1D()(A_x)  # aggregate via max-pooling
cross_imaging = A_x[:, 0]  # extracting cls token

# Co-attention: Query - Metadata; Key & Value - Imaging  TODO: for loop for num_layers
B_attention_output, B_attention_scores = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim,
                                                            dropout=dropout_rate)(proj_input_metadata,
                                                                                  proj_input_imaging,
                                                                                  return_attention_scores=True)  # B_attention_output of shape [None, 10, 384]
B_proj_input = LayerNormalization(epsilon=1e-6)(proj_input_metadata + B_attention_output)
B_proj_output = Sequential([Dense(embed_dim, activation=tf.nn.gelu), Dropout(dropout_rate), ])(B_proj_input)
B_x = LayerNormalization(epsilon=1e-6)(B_proj_input + B_proj_output)  # shape [None, 10, 384]

# Reduce output sequence through pooling layer
# cross_metadata = GlobalMaxPooling1D()(B_x)  # aggregate via max-pooling
cross_metadata = B_x[:, 0]  # extracting cls token

# Concatenate CLS tokens
features = Concatenate()([cross_imaging, cross_metadata])

# ------- (C) OUTCOME PREDICTION ---------
# Stack an MLP before the last layer
mlp_hidden_units_factors = [2, 1]  # MLP hidden layer units, as factors of the number of inputs
mlp_hidden_units = [factor * features.shape[-1] for factor in mlp_hidden_units_factors]
for units in mlp_hidden_units:
    features = Dense(units, activation='relu')(features)
    features = Dropout(0.2)(features)

# Classify outputs
# mRS_prediction = Dense(1, activation='sigmoid')(features)  # should I try with softmax?
PD_prediction = Dense(1, activation='sigmoid')(features)


# # Build model and plot its summary
# PD_model = Model(inputs=[inputs, C_inputs, N_inputs], outputs=PD_prediction) #C_inputs
# print("[INFO] printing model summary...")
# PD_model.summary()
# tf.keras.utils.plot_model(mRS_model, show_shapes=True, rankdir="LR")


def build_PD_model(inputs, C_inputs, PD_prediction):  # N_inputs
    # Build model and plot its summary
    PD_model = Model(inputs=[inputs, C_inputs], outputs=PD_prediction)  # C_inputs #N_inputs
    print("[INFO] printing model summary...")
    PD_model.summary()

    return PD_model


# -----------------------------------
#            TRAIN MODEL
# -----------------------------------



PD_model = build_PD_model(inputs, C_inputs, PD_prediction) #N_inputs

# Define loss - Focal loss: improved version of x-entropy loss that tries to handle class imbalance by down-weighting easy examples and focusing on hard ones
# focal_loss = BinaryFocalLoss(gamma=2)  # reduction='mean', from_logits=False
binary_crossentropy = tf.keras.losses.BinaryCrossentropy()

# Define optimizer
# adam_optimizer = optimizers.Adam(learning_rate=sch) #optimizers.Adam(learning_rate=1e-5)
adam_optimizer = optimizers.Adam(learning_rate=cfg.train_params['learning_rate'], decay=cfg.train_params['weight_decay'])

# Compile model
PD_model.compile(loss=binary_crossentropy, optimizer=adam_optimizer, metrics=['accuracy'])

est_score = 0
best_params = {}

reweigh = args.reweigh
fold = args.fold
fold = fold

# train_generator = DataGenerator3D_Gene(list_IDs=train_IDs, metadata_pd=train, csv_wPaths=fn_train, shuffle=True,
#                                        **cfg.params)
# val_generator = DataGenerator3D_Gene(list_IDs=val_IDs, metadata_pd=val, csv_wPaths=fn_val, shuffle=False, **cfg.params)
test_generator = DataGenerator3D_Gene(list_IDs=test_IDs, metadata_pd=test, csv_wPaths=fn_test, shuffle=False,
                                      **cfg.params)  # False
model_file = args.model
PD_model.load_weights(model_file)#'/media/gdagasso/TOSHIBA EXT/PD_geneticSubtype_fusionModel/Cross-Attention/Results/ADNI_PPMI_CV/10_fold/fold_3/model_weights_lr1e-05_epochs100_batch2_reweigh0_fold3_epoch05_best_val_loss0.4433_801010Split.h5') #'/media/gdagasso/TOSHIBA EXT/PD_geneticSubtype_fusionModel/Cross-Attention/Results/ADNI_PPMI_CV/fold_5/model_weights_lr1e-05_epochs100_batch2_fold5_ppmi_adni_602020Split.h5')
target_model = PD_model

predictions = target_model.predict(test_generator)
predictions_binary = predictions
predictions_binary[predictions_binary < 0.5] = 0
predictions_binary[predictions_binary > 0.5] = 1

true_labels = test['Group_numeric'].values
# -----------------------------------------
#          ANALYZE MODEL ATTENTION SCORES
# -----------------------------------------
# def pullPats_from_generator(test_generator, y_labels, label, column):
#     labels_0 = test_generator[0][y_labels == label]  # Get samples where y_label == 0
#     labels_1 = test_generator[0][y_labels == 1]  # Get samples where y_label == 1
#
labels_0 = true_labels[true_labels == 0]
labels_1 = true_labels[true_labels == 1]

# Note: target_model is your already trained model from which you want to extract the attention scores from

# To compute attention score, you need to
# 1. Create a new model
##### MULTI_HEAD_ATTENTION_1 IS FOR THE IMAGES AND MULTI_HEAD_ATTENTION IS FOR THE METADATA USE SAME LOGIC FOR CROSS-ATTENTION STUFF!!!!
self_attention = Model(inputs=target_model.input, outputs=[target_model.get_layer('multi_head_attention_2').output])  # double-check the name of the layer

# 2. Get the new outputs using your data

# self_probs = self_attention.predict(test_generator)
#
# # THIS PROBABLY NEEDS TO BE DONE IN A ONE AT A TIME WAY HERE
# print(self_probs[1].shape)
# attention_map = np.mean(self_probs[1], axis=1)
# attention_map = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map))  # normalization
#
#
# # Self probs will be a list of two arrays, since the MultiHeadAttention layer has two outputs (attention_output, attention_scores). Just extract the second one.
#
#
# # Upsample attention scores back to MRI space
# base_network_d = DECODER.build(reg=l2(0.00005), shape=cfg.params['dim_3D'])
# i = Input(shape=(*cfg.transformer_params['projection_dim_3D'],1))
# o = base_network_d(i)
#
# decoded_attention_output = o.predict(attention_map)
#
# print(decoded_attention_output.shape)
#
# # Reshape to match the MRI image dimensions if necessary
# decoded_attention_output = decoded_attention_output.reshape(193, 229, 193, 1)

# Make sure the generator is yielding data correctly
# j = 0
# for i in test_generator:
#
#     # Generate self_probs for the current batch (of size 2)
#     self_probs = self_attention.predict(i)
#     #print(self_probs[])
#     print(self_probs[0][0].shape)
#     print(self_probs[0][1].shape)
#     print('HELLOOOOOOO')
#
#     # self_probs is a list of two arrays (attention_output, attention_scores)
#     # Extract the attention scores (second output)
#     attention_scores = self_probs[0][1]
#     print("Attention scores shape:", attention_scores.shape)
#     attention_map = np.mean(attention_scores, axis=1)
#     attention_map = (attention_map - np.min(attention_map)) / (
#                 np.max(attention_map) - np.min(attention_map))  # normalization
#
#     print("Normalized Attention Map:", attention_map)
#
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(attention_map[0], cmap="viridis", xticklabels=False, yticklabels=False)
#     plt.title("Attention Map for SNP Data")
#     plt.xlabel("SNP Positions")
#     plt.ylabel("SNP Positions")
#     plt.show()


def plot_attention_geno_scores(self_attention1, IDs, PD, fn, label , fold_num, layer="self"):
    columns_to_keep = ['rs114138760', 'rs76763715', 'rs75548401', 'rs2230288', 'rs823118', 'rs4653767', 'rs10797576', 'rs34043159', 'rs6430538', 'rs353116', 'rs1955337', 'rs4073221', 'rs12497850', 'rs143918452', 'rs1803274', 'rs12637471', 'rs34884217', 'rs34311866', 'rs11724635', 'rs6812193', 'rs356181', 'rs3910105', 'rs4444903', 'rs78738012', 'rs2694528', 'rs9468199', 'rs8192591', 'rs115462410', 'rs199347', 'rs1293298', 'rs591323', 'rs2280104', 'rs13294100', 'rs10906923', 'rs118117788', 'rs329648', 'rs76904798', 'rs33949390', 'rs34637584', 'rs11060180', 'rs11158026', 'rs8005172', 'rs2414739', 'rs11343', 'rs14235', 'rs4784227', 'rs11868035', 'rs17649553', 'rs12456492', 'rs55785911', 'rs737866', 'rs174674', 'rs5993883', 'rs740603', 'rs165656', 'rs6269', 'rs4633', 'rs2239393', 'rs4818', 'rs4680', 'rs165599','APOE4_binary']
    generator = DataGenerator3D_Gene(list_IDs=IDs, metadata_pd=PD, csv_wPaths=fn, shuffle=False,
                                          **cfg.params)
    overall_attention_map = None
    total_samples = 0

    for batch in generator:
        # Generate self_probs for the current batch
        self_probs = self_attention1.predict(batch)

        # Extract attention scores (second output from self_probs)
        print(self_probs[0][0].shape)
        print(self_probs[0][1].shape)
        print(self_probs[0][1])
        attention_scores1 = self_probs[0][1]  # Shape: (batch_size, num_heads, seq_len, seq_len)

        # Aggregate across attention heads (mean over heads)
        batch_attention_map = np.mean(attention_scores1, axis=1)  # Shape: (batch_size, seq_len, seq_len)

        # Accumulate the batch attention maps
        if overall_attention_map is None:
            overall_attention_map = np.sum(batch_attention_map, axis=0)  # Initialize with the first batch
        else:
            overall_attention_map += np.sum(batch_attention_map, axis=0)  # Add current batch maps

        # Update total samples processed
        total_samples += batch_attention_map.shape[0]

    # Average the attention map across all samples
    overall_attention_map /= total_samples

    # Normalize the overall attention map for visualization
    overall_attention_map = (overall_attention_map - np.min(overall_attention_map)) / (
        np.max(overall_attention_map) - np.min(overall_attention_map))

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(overall_attention_map, cmap="viridis", xticklabels=columns_to_keep, yticklabels=columns_to_keep,)
    plt.title("Overall Attention Map for SNP Data")
    plt.xlabel("SNPs")
    plt.ylabel("SNPs")
    #plt.show()
    plot_filename = f"{cfg.params['resultsPath']}label_{label}_fold_{fold_num}_layer_{layer}_801010Split.svg"
    plt.savefig(plot_filename)
    plt.close()

    del generator

def plot_crossAttention_geno_scores(self_attention1, IDs, PD, fn, label , fold_num, layer="cross_2"):
    columns_to_keep = ['rs114138760', 'rs76763715', 'rs75548401', 'rs2230288', 'rs823118', 'rs4653767', 'rs10797576', 'rs34043159', 'rs6430538', 'rs353116', 'rs1955337', 'rs4073221', 'rs12497850', 'rs143918452', 'rs1803274', 'rs12637471', 'rs34884217', 'rs34311866', 'rs11724635', 'rs6812193', 'rs356181', 'rs3910105', 'rs4444903', 'rs78738012', 'rs2694528', 'rs9468199', 'rs8192591', 'rs115462410', 'rs199347', 'rs1293298', 'rs591323', 'rs2280104', 'rs13294100', 'rs10906923', 'rs118117788', 'rs329648', 'rs76904798', 'rs33949390', 'rs34637584', 'rs11060180', 'rs11158026', 'rs8005172', 'rs2414739', 'rs11343', 'rs14235', 'rs4784227', 'rs11868035', 'rs17649553', 'rs12456492', 'rs55785911', 'rs737866', 'rs174674', 'rs5993883', 'rs740603', 'rs165656', 'rs6269', 'rs4633', 'rs2239393', 'rs4818', 'rs4680', 'rs165599','APOE4_binary']
    generator = DataGenerator3D_Gene(list_IDs=IDs, metadata_pd=PD, csv_wPaths=fn, shuffle=False,
                                          **cfg.params)
    overall_attention_map = None
    total_samples = 0

    for batch in generator:
        # Generate self_probs for the current batch
        self_probs = self_attention1.predict(batch)

        # Extract attention scores (second output from self_probs)
        print(self_probs[0][0].shape)
        print(self_probs[0][1].shape)
        print(self_probs[0][1])
        attention_scores1 = self_probs[0][1]  # Shape: (batch_size, num_heads, seq_len, seq_len)

        # Aggregate across attention heads (mean over heads)
        batch_attention_map = np.mean(attention_scores1, axis=1)  # Shape: (batch_size, seq_len, seq_len)

        # Accumulate the batch attention maps
        if overall_attention_map is None:
            overall_attention_map = np.sum(batch_attention_map, axis=0)  # Initialize with the first batch
        else:
            overall_attention_map += np.sum(batch_attention_map, axis=0)  # Add current batch maps

        # Update total samples processed
        total_samples += batch_attention_map.shape[0]

    # Average the attention map across all samples
    overall_attention_map /= total_samples

    # Normalize the overall attention map for visualization
    overall_attention_map = (overall_attention_map - np.min(overall_attention_map)) / (
        np.max(overall_attention_map) - np.min(overall_attention_map))

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(overall_attention_map, cmap="viridis", xticklabels=columns_to_keep, yticklabels=['Token 1', 'Token 2'])
    plt.title("Overall Attention Map for SNP Data")
    plt.xlabel("SNPs")
    plt.ylabel("Image Tokens")
    #plt.show()
    plot_filename = f"{cfg.params['resultsPath']}cross_geno_label_{label}_fold_{fold_num}_layer_{layer}_602020Split.svg"
    plt.savefig(plot_filename)
    plt.close()

    del generator

def plot_crossAttention_img_scores(self_attention1, IDs, PD, fn, label , fold_num, layer="cross_3"):
    columns_to_keep = ['rs114138760', 'rs76763715', 'rs75548401', 'rs2230288', 'rs823118', 'rs4653767', 'rs10797576', 'rs34043159', 'rs6430538', 'rs353116', 'rs1955337', 'rs4073221', 'rs12497850', 'rs143918452', 'rs1803274', 'rs12637471', 'rs34884217', 'rs34311866', 'rs11724635', 'rs6812193', 'rs356181', 'rs3910105', 'rs4444903', 'rs78738012', 'rs2694528', 'rs9468199', 'rs8192591', 'rs115462410', 'rs199347', 'rs1293298', 'rs591323', 'rs2280104', 'rs13294100', 'rs10906923', 'rs118117788', 'rs329648', 'rs76904798', 'rs33949390', 'rs34637584', 'rs11060180', 'rs11158026', 'rs8005172', 'rs2414739', 'rs11343', 'rs14235', 'rs4784227', 'rs11868035', 'rs17649553', 'rs12456492', 'rs55785911', 'rs737866', 'rs174674', 'rs5993883', 'rs740603', 'rs165656', 'rs6269', 'rs4633', 'rs2239393', 'rs4818', 'rs4680', 'rs165599','APOE4_binary']
    generator = DataGenerator3D_Gene(list_IDs=IDs, metadata_pd=PD, csv_wPaths=fn, shuffle=False,
                                          **cfg.params)
    overall_attention_map = None
    total_samples = 0

    for batch in generator:
        # Generate self_probs for the current batch
        self_probs = self_attention1.predict(batch)

        # Extract attention scores (second output from self_probs)
        print(self_probs[0][0].shape)
        print(self_probs[0][1].shape)
        print(self_probs[0][1])
        attention_scores1 = self_probs[0][1]  # Shape: (batch_size, num_heads, seq_len, seq_len)

        # Aggregate across attention heads (mean over heads)
        batch_attention_map = np.mean(attention_scores1, axis=1)  # Shape: (batch_size, seq_len, seq_len)

        # Accumulate the batch attention maps
        if overall_attention_map is None:
            overall_attention_map = np.sum(batch_attention_map, axis=0)  # Initialize with the first batch
        else:
            overall_attention_map += np.sum(batch_attention_map, axis=0)  # Add current batch maps

        # Update total samples processed
        total_samples += batch_attention_map.shape[0]

    # Average the attention map across all samples
    overall_attention_map /= total_samples

    # Normalize the overall attention map for visualization
    overall_attention_map = (overall_attention_map - np.min(overall_attention_map)) / (
        np.max(overall_attention_map) - np.min(overall_attention_map))

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(overall_attention_map, cmap="viridis", xticklabels=['Token 1', 'Token 2'], yticklabels=columns_to_keep)
    plt.title("Overall Attention Map for SNP Data")
    plt.xlabel("Image Tokens")
    plt.ylabel("SNPs")
    #plt.show()
    plot_filename = f"{cfg.params['resultsPath']}cross_img_label_{label}_fold_{fold_num}_layer_{layer}_602020Split.svg"
    plt.savefig(plot_filename)
    plt.close()

    del generator

# All People
#test_PD = test[test['Group_numeric'] == 0]
#test_PD = test_PD.reset_index(drop=True)
#test_IDs = test_PD["PATNO"]
#print(test_PD.shape)
#print(test_IDs.shape)
#test_generator = DataGenerator3D_Gene(list_IDs=test_IDs, metadata_pd=test_PD, csv_wPaths=fn_test, shuffle=False,
#                                      **cfg.params)
plot_attention_geno_scores(self_attention, test_IDs, test, fn_test, "ALL" , fold, "cross")
#plot_crossAttention_img_scores(self_attention, test_IDs, test, fn_test, "ALL" , fold, "crossImg")

# PD People
test_PD = test[test['Group_numeric'] == 0]
test_PD = test_PD.reset_index(drop=True)
test_IDs = test_PD["PATNO"]
print(test_PD.shape)
print(test_IDs.shape)
#test_generator = DataGenerator3D_Gene(list_IDs=test_IDs, metadata_pd=test_PD, csv_wPaths=fn_test, shuffle=False,
#                                      **cfg.params)
plot_attention_geno_scores(self_attention, test_IDs, test_PD, fn_test, 0 , fold, "cross")
#plot_crossAttention_img_scores(self_attention, test_IDs, test, fn_test, 0, fold, "crossImg")

# Control People
test_PD = test[test['Group_numeric'] == 1]
test_PD = test_PD.reset_index(drop=True)
test_IDs = test_PD["PATNO"]
# test_generator = DataGenerator3D_Gene(list_IDs=test_IDs, metadata_pd=test_PD, csv_wPaths=fn_test, shuffle=False,
#                                       **cfg.params)
plot_attention_geno_scores(self_attention, test_IDs, test_PD, fn_test,1,fold, "cross")
#plot_crossAttention_img_scores(self_attention, test_IDs, test, fn_test, 1 , fold, "crossImg")

# # GBA People
# test_PD = test[test['Gene_Subtype'] == 3]
# test_PD = test_PD.reset_index(drop=True)
# test_IDs = test_PD["PATNO"]
# # test_generator = DataGenerator3D_Gene(list_IDs=test_IDs, metadata_pd=test_PD, csv_wPaths=fn_test, shuffle=False,
# #                                       **cfg.params)
# #plot_attention_geno_scores(self_attention, test_IDs, test_PD,fn_test,"GBA",fold, "cross")
# plot_crossAttention_img_scores(self_attention, test_IDs, test, fn_test, "GBA" , fold, "crossImg")
#
# # LRRK People
# test_PD = test[test['Gene_Subtype'] == 2]
# test_PD = test_PD.reset_index(drop=True)
# test_IDs = test_PD["PATNO"]
# # test_generator = DataGenerator3D_Gene(list_IDs=test_IDs, metadata_pd=test_PD, csv_wPaths=fn_test, shuffle=False,
# #                                       **cfg.params)
# # plot_attention_geno_scores(self_attention, test_IDs, test_PD,fn_test,"LRRK2",fold, "cross")
# plot_crossAttention_img_scores(self_attention, test_IDs, test, fn_test, "LRRK2" , fold, "crossImg")
#
# # APO People
# test_PD = test[test['Gene_Subtype'] == 1]
# test_PD = test_PD.reset_index(drop=True)
# test_IDs = test_PD["PATNO"]
# # #test_generator = DataGenerator3D_Gene(list_IDs=test_IDs, metadata_pd=test_PD, csv_wPaths=fn_test, shuffle=False,
# #                                       **cfg.params)
# # plot_attention_geno_scores(self_attention, test_IDs, test_PD,fn_test,"APO",fold, "cross")
# plot_crossAttention_img_scores(self_attention, test_IDs, test, fn_test, "APO" , fold, "crossImg")

    # Upsample attention scores back to MRI space

    # i_input = Input(shape=(cfg.transformer_params['projection_dim_3D'], 1))
    # base_network_d = DECODER.build(i_input, reg=l2(0.00005))  # , shape=cfg.params['dim_3D']
    # o = base_network_d(i_input)
    #
    # # Predict with the base network
    # decoded_attention_output = o.predict(attention_map)
    #
    # # Print the shape of the decoded attention output for debugging
    # print(f"Decoded attention output shape for instance {test_IDs[j]}: {decoded_attention_output.shape}")
    #
    # # Reshape to match the MRI image dimensions if necessary
    # decoded_attention_output = decoded_attention_output.reshape(193, 229, 193, 1)
    #
    # decoded_att_img = sitk.GetImageFromArray(decoded_attention_output)
    # output_attention_path = str(test_IDs[j]) + "self_attention_map.nii"  # Example: change the path as needed
    #
    # # Write the attention map to the disk
    # sitk.WriteImage(decoded_att_img, output_attention_path)
    # j = j+1

    # Optionally, save or process the attention map as needed
    # Example: save_attention_map(decoded_attention_output, instance_id=f"{i}-{j}")







