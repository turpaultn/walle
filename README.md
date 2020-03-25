# walle
Weak audio labels limitations for embeddings

This repo is the implementation of:
 "Limitations of weak labels for embedding and tagging" accepted to ICASSP 2020.
 
**Note: this is a work in progress, an extension is in progress** 
so if there are any bug or if you want details about something, do not hesitate.
 
# Data
The data used in this experiment are derived from DESED dataset.
See data_utils to see how to reproduce the data.

# 3 epxperiments
THe 3 experiments use the same network architecture, but trained differently

## Classifier
The classifier is a simple CRNN train end to end.

## Protoypical network
We train the prorotypical network by sampling some data of each class in each batch.
Some of the data are used to make the "prototypes" and the other to do the


## Reference

N. Turpault, R. Serizel, E. Vincent, "Limitations of weak labels for embedding and tagging", 
in Proc. ICASSP 2020, Barcelona, Spain.