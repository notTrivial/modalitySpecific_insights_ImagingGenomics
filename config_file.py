# Transformer configuration
transformer_params = {'n_layers': 1,
                      'n_heads': 8,
                      'projection_dim_2D': 1536,
                      'projection_dim_3D': 1573}

# Clinical metadata
categorical_features = {'names': ['GAB_coded_binary', 'PRKN_Coded_binary', 'APOE4_binary', 'LRRK2_Coded_binary'], 'categories': [2, 2, 2, 2]} #{'names': ['Sex', 'Hypertension', 'Smoking', 'Atrial_Fibrillation', 'mTICI'], 'categories': [2, 2, 2, 2, 3]}
continuous_features = ['Genetic_PRS_PRSp90'] #'Age',
target_feature = {'name': ['Group_numeric'], 'categories': [2]}  # mRs90_binary = 0-2 (good), 3-6 (severe)
#target_feature_OR = {'name': ['mRs90'], 'categories': 7}  # mRs90 = 0-6  -- Note: Minimum value needs to be 0!

# Training configuration
train_params = {'n_epochs': 100,
                'learning_rate': 0.002,
                'weight_decay': 0.0001,
                'ckpt_path': None}

# Data generator parameters
params = {#'imagePath': '/home/kimberly/PycharmProjects/2022Stroke/Datasets/IA_subgroup/',
          #'dictFile_3D': '/home/kimberly/PycharmProjects/2022Stroke/Datasets/IA_subgroup/partition_10FCV_IA_3D_ALL70.pickle',
          'clinicalFile': '/media/gdagasso/TOSHIBA EXT/PD_geneticSubtype_fusionModel/Data/PRS_PD_Control_geneticSubtype_freesurfer.csv',  # PATIENT_ID column must exist
          'train':'/media/gdagasso/PD_geneticSubtype_fusionModel/Cross-Attention/Data/ADNI_PPMI_ALL_variants_apo_age_sex_filePath.csv', #'/media/gdagasso/TOSHIBA EXT/PD_geneticSubtype_fusionModel/Data/TRAIN_PD_controls_geneticSubtype_dropped3756.csv',#'/media/gdagasso/TOSHIBA EXT/PD_geneticSubtype_fusionModel/Data/train_tmp.csv',
          'val_partition':'/media/gdagasso/TOSHIBA EXT/PD_geneticSubtype_fusionModel/Data/VAL_PD_controls_geneticSubtype_droppedUnnecesaryCols.csv',
          #'test_partition':'/media/gdagasso/TOSHIBA EXT/PD_geneticSubtype_fusionModel/Cross-Attention/TRAIN_VAL_enoughForStats.csv',
          'test_partition':'/media/gdagasso/TOSHIBA EXT/PD_geneticSubtype_fusionModel/Cross-Attention/TEST_split_enoughForStats.csv',#/media/gdagasso/TOSHIBA EXT/PD_geneticSubtype_fusionModel/Data/TEST_PD_controls_geneticSubtype_droppedUnnecesaryCols.csv',
          'all': '/media/gdagasso/TOSHIBA EXT/PD_geneticSubtype_fusionModel/Data/ALL_PD_controls_geneticSubtype_dropped3756_droppedUnnecesaryCols.csv',
          'resultsPath': '/media/gdagasso/TOSHIBA EXT/PD_geneticSubtype_fusionModel/Cross-Attention/Results/trainTestVal_ALL_CV/',#./results/old-weights-clinical-OR/',
          'dim_3D': (193, 229, 193),
          'patch_size': (16, 16, 16),
          'batch_size': 2,
          'timepoints': 1,
          'n_classes': 2,
          'features': [continuous_features, categorical_features['names']]}
