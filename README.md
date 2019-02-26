# Sentiment-Classification
### Assignment 1.1 - COL772(Spring'19): Natural Language Processing
#### Creator: Navreet Kaur[2015TT10917]
 
#### Problem Statement:
The goal of the assignment is to build a sentiment categorization system for business reviews from annotated data. The input of the code will be a set of annotated business reviews from the website Yelp.

#### Training Data:
Training dataset of business reviews can be downloaded from this link: ```https://owncloud.iitd.ac.in/nextcloud/index.php/s/2KRyxd9XLFcnpXR```. Each data point has review text, and a rating. Ratings are floating point numbers between 1 and 5.

#### The Task:
Build a non-neural classifier that given a review, predicts its sentiment polarity.

#### Files:
1. ```knowledge.py```: Containes of in-domain stopwords, positive and negative words etc.
2. ```preprocess.py```: Methods for preprocessing the raw data.
3. ```features.py```: Methods for extracting different features from the pre-processed data.
4. ```form_matrix.py```: Methods to form feature matrix from words and the features formed using previous script.
5. ```predict.py```: Methods for building and training the model given feature and label matrix.
6. ```variables.py```: Global tunable variables for selecting the preprocessing steps, features to use, vocabulary building method, model and other hyperparameters.
7. ```train.py```: Main file to run for training the model given the data.
8. ```test.py```: For making predictions on test data.
9. ```compile.sh```: Compiling the whole code.
10. ```train.sh```: Run ```sh ./train.sh trainfile.json devfile.json model_file``` to train the model with  ```trainfile.json``` as training data and ```devfile.json``` as validation data. Model will be saved in ```model_file```
11. ```test.sh```: Run ```./test.sh model_file testfile.json outputfile.txt``` to test the trained model on ```testfile.json```. A new file ```outputfile.txt```containing a sequence of numbers between 1 and 5, representing the predictions, will created.
