
# import nltk
# nltk.download('punkt')
# import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from spellchecker import SpellChecker
import random
import joblib
import string
from nltk.corpus import stopwords
import os
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split, learning_curve,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

###################################################################
########################## preprocessing ##########################
###################################################################
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Convert tokens to lowercase
    tokens = [word.lower() for word in tokens]

    # Remove punctuation
    table = str.maketrans('', '', string.punctuation)
    tokens = [word.translate(table) for word in tokens]

    # Remove digits and single-character words
    tokens = [word for word in tokens if not (word.isdigit() or len(word) == 1)]

    # Spell correction
    # spell = SpellChecker()
    # tokens = [spell.correction(word) for word in tokens]

    # Lemmatize words
    # lemmatizer = WordNetLemmatizer()
    # tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) if word is not None else '' for word in tokens]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    return tokens
def preprocess_folder(folder_path):
    preprocessed_texts = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r') as file:
            text = file.read()
            preprocessed_text = preprocess_text(text)
            preprocessed_texts.append(preprocessed_text)
    return preprocessed_texts
def train_model(classifier_name, classifier, feature_vector_train, label, feature_vector_valid, valid_y, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    filename = classifier_name + '1_model.pkl'
    joblib.dump(classifier, filename)  # Change 'classifier_model.pkl' to your desired filename
    print("Model saved successfully.")
    accuracy = accuracy_score(valid_y, predictions)
    return accuracy, predictions
#######################################################################################################################################
"""                                                                                                   """
#######################################################################################################################################
def Hyperparam(x_test,y_test,model,parms,x_train, y_train,x):# selection for best hyperparamter
    if x==0:
        grid_search = RandomizedSearchCV(model, param_distributions=parms, n_iter=100, cv=5, random_state=42)
    else:
        grid_search = GridSearchCV(model, param_grid=parms, cv=5)
    grid_search.fit(x_train, y_train)
    # Print the best hyperparameters and accuracy score
    print("Best hyperparameters: ", grid_search.best_params_)
    print("Accuracy score: ", grid_search.score(x_test, y_test))
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
def plot_confusion_matrixM(y_test, pred):
    plt.figure(figsize=(10, 6))
    fx = sns.heatmap(metrics.confusion_matrix(y_test, pred), annot=True, fmt=".2f", cmap="GnBu")
    fx.set_title('Confusion Matrix \n')
    fx.set_xlabel('\n Predicted Values\n')
    fx.set_ylabel('Actual Values\n')
    fx.xaxis.set_ticklabels(['Medium Profit', 'Low Profit', 'Low Loss', 'High Profit', 'High Loss'])
    fx.yaxis.set_ticklabels(['Medium Profit', 'Low Profit', 'Low Loss', 'High Profit', 'High Loss'])
    plt.show()
def perform_voting(predictions):
    # Initialize an empty list to store the voted predictions
    voted_predictions = []

    # Iterate over each sample's predictions
    for i in range(len(predictions[0])):
        # Extract the predictions for the current sample from all three models
        sample_predictions = [pred[i] for pred in predictions]

        # Perform voting by taking the mode of predictions for the current sample
        voted_prediction = np.argmax(np.bincount(sample_predictions))

        # Append the voted prediction to the list
        voted_predictions.append(voted_prediction)

    return voted_predictions

# Function to plot predictions and true labels
def plot_predictions_with_labels(predictions, true_labels, model_names):
    num_samples = len(predictions[0])
    num_models = len(predictions)

    plt.figure(figsize=(10, 6))

    for i in range(num_models):
        plt.subplot(num_models, 1, i+1)
        plt.plot(range(num_samples), predictions[i], label='Predicted')
        plt.plot(range(num_samples), true_labels, label='True', linestyle='--')
        plt.xlabel('Sample Index')
        plt.ylabel('Label')
        plt.title(f'Predictions vs True Labels ({model_names[i]})')
        plt.legend()

    plt.tight_layout()
    plt.show()
#######################################################################################################################################
"""                                                     Read Data                                         """
#######################################################################################################################################
positive_texts = preprocess_folder('C:/Users/dell/Downloads/review_polarity/review_polarity/txt_sentoken/pos')
negative_texts = preprocess_folder('C:/Users/dell/Downloads/review_polarity/review_polarity/txt_sentoken/neg')
# Labels for positive and negative texts
positive_labels = [1] * len(positive_texts)
negative_labels = [0] * len(negative_texts)
# Combine texts and labels
all_texts = positive_texts + negative_texts
all_labels = positive_labels + negative_labels
# Shuffle the data
combined_data = list(zip(all_texts, all_labels))
random.shuffle(combined_data)
all_texts, all_labels = zip(*combined_data)
# Split data into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(all_texts, all_labels, test_size=0.2,
                                                                      random_state=42)
#######################################################################################################################################
"""                                                       tfidf                                         """
#######################################################################################################################################

encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_labels)
valid_y = encoder.fit_transform(test_labels)
# Join the tokens of each document into a single string
train_texts_joined = [' '.join(tokens) for tokens in train_texts]
test_texts_joined = [' '.join(tokens) for tokens in test_texts]
# tfidf_vect = TfidfVectorizer()
tfidf_vect = TfidfVectorizer(max_features=37300)
tfidf_vect.fit(train_texts_joined)
# Transform train and test texts
xtrain_tfidf = tfidf_vect.transform(train_texts_joined)
xvalid_tfidf = tfidf_vect.transform(test_texts_joined)

joblib.dump(tfidf_vect, 'tfidf_vectorizer.pkl')
print("TF-IDF Vectorizer saved successfully.")
xtrain_tfidf.data

print("Number of training samples:", len(train_texts))
print("Number of testing samples:", len(test_texts))

#######################################################################################################################################
"""                                              model training                                        """
#######################################################################################################################################
# Naive Bayes training
accuracy_nb, nb_predictions = train_model("naive_bayes", naive_bayes.MultinomialNB(alpha=0.2), xtrain_tfidf, train_y, xvalid_tfidf, valid_y)
print("Naive Bayes Accuracy: ", accuracy_nb)

# Logistic Regression training
accuracy_lr, lr_predictions = train_model("LogisticRegression", linear_model.LogisticRegression(C=10, max_iter=100, solver='newton-cg'), xtrain_tfidf, train_y, xvalid_tfidf, valid_y)
print("Logistic Regression Accuracy: ", accuracy_lr)

accuracy_svm, svm_predictions = train_model("svm", svm.SVC(kernel='linear', probability=True), xtrain_tfidf, train_y, xvalid_tfidf, valid_y)
print("SVM Accuracy: ", accuracy_svm)

accuracy_Knn, Knn_predictions = train_model("KNN", KNeighborsClassifier(n_neighbors=7,p=2,weights="distance"), xtrain_tfidf, train_y, xvalid_tfidf, valid_y)
print("KNN: ", accuracy_Knn)
#######################################################################################################################################
"""                                            confusion_matrix                                               """
#######################################################################################################################################
# Compute evaluation metrics
cm_nb = confusion_matrix(valid_y, nb_predictions)
precision_nb = precision_score(valid_y, nb_predictions)
recall_nb = recall_score(valid_y, nb_predictions)
f1_nb = f1_score(valid_y, nb_predictions)
#######################################################################################################################################
cm_lr = confusion_matrix(valid_y, lr_predictions)
precision_lr = precision_score(valid_y, lr_predictions)
recall_lr = recall_score(valid_y, lr_predictions)
f1_lr = f1_score(valid_y, lr_predictions)
#######################################################################################################################################
# Compute evaluation metrics
cm_svm = confusion_matrix(valid_y, svm_predictions)
precision_svm = precision_score(valid_y, svm_predictions)
recall_svm = recall_score(valid_y, svm_predictions)
f1_svm = f1_score(valid_y, svm_predictions)
#######################################################################################################################################
# Compute evaluation metrics
cm_knn = confusion_matrix(valid_y, Knn_predictions)
precision_knn = precision_score(valid_y, Knn_predictions)
recall_knn = recall_score(valid_y, Knn_predictions)
f1_knn = f1_score(valid_y, Knn_predictions)
#######################################################################################################################################
print("Naive Bayes:")
print("Confusion Matrix:")
print(cm_nb)
print("Precision:", precision_nb)
print("Recall:", recall_nb)
print("F1-score:", f1_nb)
#######################################################################################################################################
print("\nLogistic Regression:")
print("Confusion Matrix:")
print(cm_lr)
print("Precision:", precision_lr)
print("Recall:", recall_lr)
print("F1-score:", f1_lr)
#######################################################################################################################################
print("\nSVM:")
print("Confusion Matrix:")
print(cm_svm)
print("Precision:", precision_svm)
print("Recall:", recall_svm)
print("F1-score:", f1_svm)
#######################################################################################################################################
print("\nKNN:")
print("Confusion Matrix:")
print(cm_knn)
print("Precision:", precision_knn)
print("Recall:", recall_knn)
print("F1-score:", f1_knn)
#######################################################################################################################################
# Plot Naive Bayes confusion matrix
plot_confusion_matrix(cm_nb, title='Naive Bayes Confusion Matrix')
plot_confusion_matrix(cm_lr, title='Logistic Regression Confusion Matrix')
plot_confusion_matrix(cm_svm, title='SVM Confusion Matrix')
plot_confusion_matrix(cm_knn, title='KNN Confusion Matrix')

#######################################################################################################################################
# plot_confusion_matrix(valid_y, nb_predictions.squeeze())
# plot_confusion_matrix(valid_y, lr_predictions.squeeze())
# plot_confusion_matrix(valid_y, Knn_predictions.squeeze())
# plot_confusion_matrix(valid_y, svm_predictions.squeeze())


#######################################################################################################################################
"""                                            models Voting                                                    """
#######################################################################################################################################
all_predictions = [svm_predictions , lr_predictions , nb_predictions,Knn_predictions ]
final_predictions = perform_voting(all_predictions)
model_names = ['svm_predictions', ' LogisticRegression_pred', 'naiivbays_pred','KNN_pred']

plot_predictions_with_labels(all_predictions,valid_y , model_names)
#######################################################################################################################################
"""                                            model testing                                                    """
#######################################################################################################################################

params_L = {'C': [0.001, 0.01, 0.1, 1,10],
          'solver': ['newton-cg', 'lbfgs', 'liblinear'],
          'max_iter': [100, 500, 1000]}

params_NB = {
    'alpha': [0.1, 0.2, 0.5, 1.0, 2.0],
    'fit_prior': [True, False],  # Include fit_prior
    'class_prior': [None, [0.3, 0.7], [0.4, 0.6], [0.5, 0.5]]  # Include class_prior
}
param_grid = {'n_neighbors': [3, 5, 7],
              'weights': ['uniform', 'distance'],
              'p': [1, 2]}
params_SVM = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel type
    'gamma': ['scale', 'auto'],  # Kernel coefficient
    'degree': [2, 3, 4],  # Degree of the polynomial kernel (if kernel is 'poly')
    'coef0': [0.0, 1.0],  # Independent term in kernel function
    'shrinking': [True, False],  # Whether to use the shrinking heuristic
    'probability': [True, False]  # Whether to enable probability estimates
}
knn = KNeighborsClassifier()
lr = LogisticRegression()
nb = MultinomialNB()
svm=SVC()

# Hyperparam(xvalid_tfidf,valid_y,lr,params_L,xtrain_tfidf,train_y,1)
# Hyperparam(xvalid_tfidf, valid_y, nb, params_NB, xtrain_tfidf, train_y, 1)
# Hyperparam(xvalid_tfidf, valid_y, knn, param_grid, xtrain_tfidf, train_y, 1)
# Hyperparam(xvalid_tfidf, valid_y, svm, params_SVM, xtrain_tfidf, train_y, 1)



#######################################################################################################################################
                                                # LogisticRegression

"""'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
'C': [0.001, 0.01, 0.1, 1, 10],
Best hyperparameters:  {'C': 10, 'max_iter': 100, 'solver': 'saga'}
Accuracy score:  0.83
Best hyperparameters:  {'C': 10, 'max_iter': 500, 'solver': 'saga'}
Accuracy score:  0.8625
Best hyperparameters:  {'C': 10, 'max_iter': 100, 'solver': 'newton-cg'}
Accuracy score:  0.83
Best hyperparameters:  {'C': 10, 'max_iter': 100, 'solver': 'liblinear'}
Accuracy score:  0.8625
Best hyperparameters:  {'C': 10, 'max_iter': 100, 'solver': 'newton-cg'}
Accuracy score:  0.885
Best hyperparameters:  {'C': 10, 'max_iter': 100, 'solver': 'liblinear'}
Accuracy score:  0.84
Best hyperparameters:  {'C': 1, 'max_iter': 100, 'solver': 'newton-cg'}
Accuracy score:  0.845
Best hyperparameters:  {'C': 10, 'max_iter': 100, 'solver': 'newton-cg'}
Accuracy score:  0.8775
Best hyperparameters:  {'C': 10, 'max_iter': 100, 'solver': 'newton-cg'}
Accuracy score:  0.86
"""
#######################################################################################################################################
                                                        # NB
"""                                                          
Best hyperparameters:  {'alpha': 0.5, 'class_prior': None, 'fit_prior': False}
Accuracy score:  0.8075
Best hyperparameters:  {'alpha': 0.5, 'class_prior': None, 'fit_prior': False}
Accuracy score:  0.83
"""
#######################################################################################################################################



