from numpy.random import seed

seed(1)
import tensorflow as tf

tf.random.set_seed(1)
import random

random.seed(1)
import numpy as np
# np.random.seed(1)
import pandas as pd
import tensorflow as tf
import SimpleITK as sitk
from tensorflow.keras.utils import to_categorical
from sklearn.impute import SimpleImputer



class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, batch_size, dim, shuffle, filename, column,dim_g, geno_file):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.filename = filename
        self.column = column
        self.on_epoch_end()
        self.dim_g = dim_g
        self.geno_file = geno_file
        self.current_IDs = list_IDs

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        self.current_IDs = list_IDs_temp
        # Generate data
        X_geno, y = self.__data_generation(list_IDs_temp)

        return [X_geno], y

    def get_current_IDs(self):
        # Access the IDs for the current batch
        return self.current_IDs

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        #X_pheno = np.empty((self.batch_size, *self.dim, 1))
        X_geno = np.empty((self.batch_size, self.dim_g, 1))
        y = np.empty((self.batch_size), dtype=int)  # label

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # load based on the ID from csv filter by ID
            dataset = pd.read_csv(self.filename,sep=",")
            dataset = dataset[dataset['PATNO'] == ID]
            # print(ID)
            #path = dataset['FULL_FILE_PATH'].values
            # print(path)
            #itk_img = sitk.ReadImage(path)
            #np_img = sitk.GetArrayFromImage(itk_img)
            #X_pheno[i,] = np.float32(np_img.reshape(self.dim[0], self.dim[1], self.dim[2], 1))
            X_geno[i,] = self.__process_geno(ID, self.geno_file).reshape((self.dim_g, 1)) #61 SNPs from PPMI and 1 for APO status
            y[i,] = dataset["Group_numeric"].values

        return X_geno, y  # This line will take care of outputing the inputs for training and the labels


    # Processing genotype data per ID
    def __process_geno(self, ID, geno_file):
        #dataframe = pd.read_csv(geno_file,sep="\t")#, sep="\t", header=0)#, index_col=2)
        dataframe = geno_file
        #print(dataframe.size)
        #print(dataframe.columns)
        data = dataframe[dataframe['PATNO'] == ID]
        #print(data.shape)
        data = data.drop(columns=['PATNO','FULL_FILE_PATH','Age','Sex_Numeric','Group_numeric','lrrk2','gba','Gene_Subtype','stratify_col','strat_col'])
        #print(data.shape)
        snp_data_ID = data.to_numpy()
        #print(snp_data_ID.shape)
        # dataframe = dataframe.drop(columns=['CHROM', 'POS', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT'])
        # SI = SimpleImputer(missing_values=9, strategy='most_frequent')
        # # knn_imputer = KNNImputer(missing_values='9',n_neighbors=100, weights='uniform', metric='nan_euclidean')
        # tmp = SI.fit_transform(dataframe)
        # tmp = tmp / 2.0
        # raw_data = tmp
        #
        # # Transpose the DataFrame
        # transposed_df = raw_data.transpose()
        #
        # # Reset the index to create a new 'subjectID' column
        # transposed_df.reset_index(inplace=True)
        #
        # # Rename the 'index' column to 'subjectID'
        # transposed_df.rename(columns={'index': 'subjectID'}, inplace=True)
        #
        # # Display the resulting DataFrame
        # print(transposed_df)
        # # print(dataframe_T.head)
        #
        # # Impute with KNN/ Simple
        # SI = SimpleImputer(missing_values=9, strategy='most_frequent')
        # # knn_imputer = KNNImputer(missing_values='9',n_neighbors=100, weights='uniform', metric='nan_euclidean')
        # tmp = SI.fit_transform(transposed_df)
        # tmp = tmp / 2.0
        # raw_data = tmp
        #
        # snp_data_ID = raw_data.to_numpy()

        return snp_data_ID
