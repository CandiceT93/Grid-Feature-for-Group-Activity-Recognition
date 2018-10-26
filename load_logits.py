import numpy as np
from sklearn.metrics import confusion_matrix

filename_logits = "./lstm_spatial_test_flipped/logits.txt"
filename_labels = "./lstm_spatial_test_flipped/labels.txt"
filename_preds = "./lstm_spatial_test_flipped/preds.txt"
motion_logits = np.genfromtxt(filename_logits, delimiter = ' ', dtype = float)
labels = np.genfromtxt(filename_labels, dtype = float)
preds = np.genfromtxt(filename_preds, dtype = float)

print motion_logits.shape
print labels.shape
print preds.shape

unique, counts = np.unique(labels, return_counts=True)
print dict(zip(unique, counts))

num_test = labels.shape[0]
original_accuarcy_flow = np.sum(preds == labels)
original_accuarcy_flow = float(original_accuarcy_flow) / float(num_test)
print("original accuarcy on optical flow image is " + str(original_accuarcy_flow))
cfm_original_flow = confusion_matrix(labels, preds)
print("original confusion matrix on optical flow image is ")
print(cfm_original_flow)