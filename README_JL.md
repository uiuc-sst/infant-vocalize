Written by Jialu Li
### Important files ###
NN/ser.py -- 2-layer FCN model entry point
NN/fc.py -- 2-layer FCN model implementation

SER/ser_adapt_lena.py -- CNN model entry point
SER/model_py/cnn_multi_filters_v2.py -- CNN model implementation

LDA/LDA_classification.py -- LDA model entry point


### Important argument structures ###
feature-train-path: path to training features 
emo-train-path: path to training labels
feature-test-path: path to testing features 
emo-test-path: path to testing labels
load: path to pretrained models
save: path to where to save trained models
weighted_sampler: whether to use weighted sampler or not

### Extracted feature pathes and names ###
path to extracted features for infant data: /ws/ifp-05/hasegawa/jialuli3/features/merged_idp_lena_5way
path to extracted features by feature selection for infant data: /ws/ifp-05/hasegawa/jialuli3/features/idp_lena_5way
path to extracted features by feature selection for CRIED data: /ws/ifp-05/hasegawa/jialuli3/features/CRIED
path to extracted features for mother data: /ws/ifp-05/hasegawa/jialuli3/features/idp_mom

Features naming convention are data_type+feature_name+fold+normalized
For example, train_embo1_norm.h5 means normalized default training features for 1st fold of the data.
train_combined1_multiple_norm.h5 means normalized default+comp training features for 1st fold of the data.
train_embo_norm_fisher_1000.h5 means normalized top 1000 default features for Fisher scores. Use train_embo_norm_fisher_1000_padded.h5 for CNN model.


### Pretrained model names
Pretrained model names generally has the format of data_type+feature_name+fold_number+special_condition. For example,
idp_lena_5way_embo_1.pt means default features for 1st fold of idp+lena data of infant data. 
idp_lena_5way_combined1_multiple.pt means default+complementary features for the 1st fold of idp+lena data of infant data. 
idp_lena_5way_combined_fisher_1000_weighted_sampler_1.pt means selected top 1000 features for default+complementary feature set for the 1st fold of idp+lena data of infant data. 

Most of the times, the feature names and model names explain itself what the settings of the models are. 
