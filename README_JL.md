Written by Jialu Li
### Important files ###
NN/ser.py -- 2-layer FCN model entry point

NN/fc.py -- 2-layer FCN model implementation

SER/ser_adapt_lena.py -- CNN model entry point

SER/model_py/cnn_multi_filters_v2.py -- CNN model implementation

SER/ser_adapt_lena_multitask.py -- CNN model entry point for multi-tiers training

SER/model_py/cnn_multi_filters_multitask.py -- CNN model implementation for multi-tiers

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
path to extracted features for infant data: /ws/ifp-05/hasegawa/jialuli3/idp_data/features/merged_idp_lena_5way

path to extracted features by feature selection for infant data: /ws/ifp-05/hasegawa/jialuli3/idp_data/features/idp_lena_5way

path to extracted features by feature selection for CRIED data: /ws/ifp-05/hasegawa/jialuli3/idp_data/features/CRIED

path to extracted features for mother data: /ws/ifp-05/hasegawa/jialuli3/features/idp_data/idp_mom

Features naming convention are data_type+feature_name+fold+normalized

For example, train_embo1_norm.h5 means normalized default training features for 1st fold of the data.

train_combined1_multiple_norm.h5 means normalized default+comp training features for 1st fold of the data.

train_embo_norm_fisher_1000.h5 means normalized top 1000 default features for Fisher scores. Use train_embo_norm_fisher_1000_padded.h5 for CNN model.


### Pretrained model names
path to pretrained data for 2-layer FCN: /ws/ifp-05/hasegawa/jialuli3/idp_data/model_weights/NN

path to pretrained data for CNN: /ws/ifp-05/hasegawa/jialuli3/idp_data/model_weights/CNN

Pretrained model names generally has the format of data_type+feature_name+fold_number+special_condition. For example,

idp_lena_5way_embo_1.pt means default features for 1st fold of idp+lena data of infant data. 

idp_lena_5way_combined1_multiple.pt means default+complementary features for the 1st fold of idp+lena data of infant data. 

idp_lena_5way_combined_fisher_1000_weighted_sampler_1.pt means selected top 1000 features for default+complementary feature set for the 1st fold of idp+lena data of infant data. 

### Most of the times, the feature names and model names explain itself what the settings of the models are. ###

### Labeling scheme ###
#### Classification models used in Speech communication paper
Infant vocalization: CRY 0, FUS 1, LAU 2, BAB 3, SCR 4

Mother vocalization: IDS 0, ADS 1, PLA 2, RHY 3, LAU 4, WHI 5

#### Multitask learning: label consists of 5 digis

first digit: speaker diarization label, SIL 0, CHN 1, FAN 2, MAN 3, CXN 4, MIX 5

second digit: CHN label, SIL 0, CRY 1, FUS 2, LAU 3, BAB 4, SCR 5

third digit: FAN label, SIL 0, IDS 1, FAN 2, LAU 3, PLA 4

fourth digit: MAN label, SIL 0, IDS 1, MAN 2, LAU 3, SNG 4, PLA 5

fifth digit: CXN label, SIL 0, IDS 1, CXN 2, LAU 3, SNG 4, PLA 5

For example, 12000 means infant is fussing, 51100 means infant is crying while mom is talking to infant with motherese. Note that if first digit is 0, then the rest of the digits will be zeros(00000). 00000 is the unique label for silence.
