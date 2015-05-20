# kaggle-Otto-Group-Competition-NNs-Model-
The link to this Kaggle otto group competition is below:
https://www.kaggle.com/c/otto-group-product-classification-challenge

This is our team best Neural Network model we use to combine with xgb model. The single NN model can give a leaderboard of around 0.44 (logloss), and a simple linear combination of 10NNs will give around 0.43. 

The features used for this model including the followings:

1) dropout layers (p = 0.15, 0.25, 0.25)

2) ReLU as nonlineariies for each activation. (default) 

3) softmax as output

4) expotential decreasing of learning rate and momentum

5) L2 regularization

