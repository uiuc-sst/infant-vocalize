Classify infant vocalizations as
babble, cry, fuss, laugh, hic.

Author: Yijia Xu <yijia.xu11@gmail.com>, 2017

The best classifier was a pretty simple architecture:
linear discriminant analysis, applied to a vector of OpenSmile features.

Example of data cleaning and feature extraction
`make_balance_dataset.py`
`make_tfrecords_FBANK.py`

Feature selection
`feature_selection.py`

CNN model
`CNNmodel.py`

Train and evaluate CNN
`train_prosody.py`
`eval.py`

Train and evaluate LDA
`LDA_test_reliability.py`

Evaluate CNN+HMM
`HMM_CODE/caveneuwirth.py`

Related: transcribe mom and child speech,
https://github.com/yijiaxu3/speech-segmentation-transcription
