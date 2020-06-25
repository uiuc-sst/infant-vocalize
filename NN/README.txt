ser.py -- the main program
fc.py -- architecture of the emotion classifier
loss.py -- deepFmeasure loss
util.py -- utility file containing helper functions necessary for the model to run

Under model folder,
idp_mom.pt -- pretrained model weights for mom vocalization
lena_idp_4way.pt -- 4 way classifiers (CRY, FUS, LAU, BAB) for child vocalization
lena_idp_5way.pt -- 5 way classifiers  (CRY, FUS, LAU, BAB, SCR) for child vocalization

To run the model, simply setup feature-train-path, emo-train-path, feature-test-path, and emo-test-path;
then run python ser.py
