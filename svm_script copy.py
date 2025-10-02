#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from svm_source import *
from sklearn import svm
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from time import time
import csv
import os

scaler = StandardScaler()

import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')

#%%
###############################################################################
#               Toy dataset : 2 gaussians
###############################################################################

n1 = 200
n2 = 200
mu1 = [1., 1.]
mu2 = [-1./2, -1./2]
sigma1 = [0.9, 0.9]
sigma2 = [0.9, 0.9]
X1, y1 = rand_bi_gauss(n1, n2, mu1, mu2, sigma1, sigma2)

plt.show()
plt.close("all")
plt.ion()
plt.figure(1, figsize=(15, 5))
plt.title('First data set')
plot_2d(X1, y1)

X_train = X1[::2]
Y_train = y1[::2].astype(int)
X_test = X1[1::2]
Y_test = y1[1::2].astype(int)

# fit the model with linear kernel
clf = SVC(kernel='linear')
clf.fit(X_train, Y_train)

# predict labels for the test data base
y_pred = clf.predict(X_test)

# check your score
score = clf.score(X_test, Y_test)
print('Score : %s' % score)

# display the frontiere
def f(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf.predict(xx.reshape(1, -1))

plt.figure()
frontiere(f, X_train, Y_train, w=None, step=50, alpha_choice=1)

# Same procedure but with a grid search
parameters = {'kernel': ['linear'], 'C': list(np.linspace(0.001, 3, 21))}
clf2 = SVC()
clf_grid = GridSearchCV(clf2, parameters, n_jobs=-1)
clf_grid.fit(X_train, Y_train)

# check your score
print(clf_grid.best_params_)
print('Score : %s' % clf_grid.score(X_test, Y_test))

def f_grid(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_grid.predict(xx.reshape(1, -1))

# display the frontiere
plt.figure()
frontiere(f_grid, X_train, Y_train, w=None, step=50, alpha_choice=1)

#%%
###############################################################################
#               Iris Dataset
###############################################################################

iris = datasets.load_iris()
X = iris.data
X = scaler.fit_transform(X)
y = iris.target
X = X[y != 0, :2]
y = y[y != 0]

# split train test (say 25% for the test)
# You can shuffle and then separate or you can just use train_test_split 
#whithout shuffling (in that case fix the random state (say to 42) for reproductibility)
X, y = shuffle(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
###############################################################################
# fit the model with linear vs polynomial kernel
###############################################################################

#%%
# Q1 Linear kernel

# fit the model and select the best hyperparameter C
parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 200))}
clf_linear = GridSearchCV(SVC(), parameters, cv=5)
clf_linear.fit(X_train, y_train)

# compute the score

print('Generalization score for linear kernel: %s, %s' %
      (clf_linear.score(X_train, y_train),
       clf_linear.score(X_test, y_test)))

#%%
# Q2 polynomial kernel
Cs = list(np.logspace(-3, 3, 5))
gammas = 10. ** np.arange(1, 2)
degrees = np.r_[1, 2, 3]

# fit the model and select the best set of hyperparameters
parameters = {'kernel': ['poly'], 'C': Cs, 'gamma': gammas, 'degree': degrees}
clf_poly = GridSearchCV(SVC(), parameters, cv=5)
clf_poly.fit(X_train, y_train)


print(clf_poly.best_params_)
print('Generalization score for polynomial kernel: %s, %s' %
      (clf_poly.score(X_train, y_train),
       clf_poly.score(X_test, y_test)))


#%%
# display your results using frontiere (svm_source.py)

def f_linear(xx):
    return clf_linear.predict(xx.reshape(1, -1))

def f_poly(xx):
    return clf_poly.predict(xx.reshape(1, -1))

plt.ion()
plt.figure(figsize=(15, 5))
plt.subplot(131)
plot_2d(X, y)
plt.title("iris dataset")

plt.subplot(132)
frontiere(f_linear, X, y)
plt.title("linear kernel")

plt.subplot(133)
frontiere(f_poly, X, y)

plt.title("polynomial kernel")
plt.tight_layout()
plt.draw()

#%%
###############################################################################
#               Face Recognition Task
###############################################################################
"""
The dataset used in this example is a preprocessed excerpt
of the "Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

  _LFW: http://vis-www.cs.umass.edu/lfw/
"""

####################################################################
# Download the data and unzip; then load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4,
                              color=True, funneled=False, slice_=None,
                              download_if_missing=True)
# data_home='.'

# introspect the images arrays to find the shapes (for plotting)
images = lfw_people.images
n_samples, h, w, n_colors = images.shape

# the label to predict is the id of the person
target_names = lfw_people.target_names.tolist()

####################################################################
# Pick a pair to classify such as
# names = ['Donald Rumsfeld', 'Colin Powell']
names = ['Donald Rumsfeld', 'Colin Powell']

idx0 = (lfw_people.target == target_names.index(names[0]))
idx1 = (lfw_people.target == target_names.index(names[1]))
images = np.r_[images[idx0], images[idx1]]
n_samples = images.shape[0]
y = np.r_[np.zeros(np.sum(idx0)), np.ones(np.sum(idx1))].astype(int)

#teacher's code (bad version)
#y = np.r_[np.zeros(np.sum(idx0)), np.ones(np.sum(idx1))].astype(np.int)

# plot a sample set of the data
plot_gallery(images, np.arange(12))
plt.show()

#%%
####################################################################
# Extract features

# features using only illuminations
X = (np.mean(images, axis=3)).reshape(n_samples, -1)

# Scale features
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

#%%
####################################################################
# Split data into a half training and half test set

indices = np.random.permutation(X.shape[0])
train_idx, test_idx = indices[:X.shape[0] // 2], indices[X.shape[0] // 2:]
X_train, X_test = X[train_idx, :], X[test_idx, :]
y_train, y_test = y[train_idx], y[test_idx]
images_train, images_test = images[
    train_idx, :, :, :], images[test_idx, :, :, :]

####################################################################
# Quantitative evaluation of the model quality on the test set

#%%
# Q4
print("--- Linear kernel ---")
print("Fitting the classifier to the training set")
t0 = time()

# fit a classifier (linear) and test all the Cs
Cs = 10. ** np.arange(-5, 6)
scores = []
for C in Cs:
    clf = SVC(kernel="linear", C=C)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))

ind = np.argmax(scores)
print("Best C: {}".format(Cs[ind]))

plt.figure()
plt.plot(Cs, scores)
plt.xlabel("Parametres de regularisation C")
plt.ylabel("Scores d'apprentissage")
plt.xscale("log")
plt.tight_layout()
plt.show()
print("Best score: {}".format(np.max(scores)))

print("Predicting the people names on the testing set")
t0 = time()


#%%
# let's calculate the error
errors = []
for C in Cs:
    clf = SVC(kernel="linear", C=C)
    clf.fit(X_train, y_train)
    errors.append(1 - clf.score(X_test, y_test))

# look for the min of the error
best_ind = np.argmin(errors)
best_C = Cs[best_ind]
best_error = errors[best_ind]

plt.figure()
plt.plot(Cs, errors, marker="o")

# plot error
plt.scatter(best_C, best_error, color="red", s=100, zorder=5)

plt.xscale("log")
plt.xlabel("Paramètre de régularisation C")
plt.ylabel("Erreur de prédiction")
plt.title("Influence de C sur la performance")
plt.grid(True)
plt.tight_layout()
plt.show()


print("Best C: {}".format(Cs[best_ind]))
print("Best error: {}".format(np.min(errors)))
print("Best accuracy(score): {}".format(1 - np.min(errors)))
t0 = time()

#%%
# confusion matrix
# small C
clf_small = SVC(kernel="linear", C=1e-5)
clf_small.fit(X_train, y_train)
y_pred_small = clf_small.predict(X_test)

cm_small = confusion_matrix(y_test, y_pred_small, labels=clf_small.classes_)
disp_small = ConfusionMatrixDisplay(confusion_matrix=cm_small, display_labels=clf_small.classes_)

# big C
clf_large = SVC(kernel="linear", C=1e5)
clf_large.fit(X_train, y_train)
y_pred_large = clf_large.predict(X_test)

cm_large = confusion_matrix(y_test, y_pred_large, labels=clf_large.classes_)
disp_large = ConfusionMatrixDisplay(confusion_matrix=cm_large, display_labels=clf_large.classes_)

# rows
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# 0 = Donald Rumsfeld, 1 = Colin Powell)
class_names = ['Donald Rumsfeld', 'Colin Powell']

disp_small = ConfusionMatrixDisplay(confusion_matrix=cm_small,
                                    display_labels=class_names)
disp_small.plot(ax=axes[0], cmap="Blues", xticks_rotation='vertical', colorbar=False)
axes[0].set_title("Confusion matrix (C=1e-5)")
disp_large = ConfusionMatrixDisplay(confusion_matrix=cm_large,
                                    display_labels=class_names)
disp_large.plot(ax=axes[1], cmap="Blues", xticks_rotation='vertical', colorbar=False)
axes[1].set_title("Confusion matrix (C=1e5)")

plt.tight_layout()
plt.show()

#%% the same thing but in numbers
print("=== Classification report (C=1e-5) ===")
print(classification_report(y_test, y_pred_small, target_names=class_names))

print("\n=== Classification report (C=1e5) ===")
print(classification_report(y_test, y_pred_large, target_names=class_names))

#%%
# predict labels for the X_test images with the best classifier
clf = SVC(kernel="linear", C=Cs[ind])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("done in %0.3fs" % (time() - t0))
# The chance level is the accuracy that will be reached when constantly predicting the majority class.
print("Chance level : %s" % max(np.mean(y), 1. - np.mean(y)))
print("Accuracy : %s" % clf.score(X_test, y_test))

#%%
####################################################################
# Qualitative evaluation of the predictions using matplotlib

prediction_titles = [title(y_pred[i], y_test[i], names)
                     for i in range(y_pred.shape[0])]

plot_gallery(images_test, prediction_titles)
plt.show()

####################################################################
# Look at the coefficients
plt.figure()
plt.imshow(np.reshape(clf.coef_, (h, w)))
plt.show()

#%% 
# Qualitative evaluation of the predictions using matplotlib
####################################################################
# Look at the coefficients pour C=1e-5 et C=1e5
coef_small = clf_small.coef_.ravel()
coef_large = clf_large.coef_.ravel()


fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(coef_small.reshape(h, w), cmap=plt.cm.seismic, interpolation="nearest")
axes[0].set_title("Coefficients (C=1e-5)")

axes[1].imshow(coef_large.reshape(h, w), cmap=plt.cm.seismic, interpolation="nearest")
axes[1].set_title("Coefficients (C=1e5)")

plt.tight_layout()
plt.show()

#%%
# Q5

def run_svm_cv(_X, _y):
    _indices = np.random.permutation(_X.shape[0])
    _train_idx, _test_idx = _indices[:_X.shape[0] // 2], _indices[_X.shape[0] // 2:]
    _X_train, _X_test = _X[_train_idx, :], _X[_test_idx, :]
    _y_train, _y_test = _y[_train_idx], _y[_test_idx]

    _parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 5))}
    _svr = svm.SVC()
    _clf_linear = GridSearchCV(_svr, _parameters)
    _clf_linear.fit(_X_train, _y_train)

    print('Generalization score for linear kernel: %s, %s \n' %
          (_clf_linear.score(_X_train, _y_train), _clf_linear.score(_X_test, _y_test)))

print("Score sans variable de nuisance")
run_svm_cv(X, y)

print("Score avec variable de nuisance")
n_features = X.shape[1]
# We add nuisance variables
sigma = 1
noise = sigma * np.random.randn(n_samples, 300, ) 
# with gaussian coefficients of std sigma
X_noisy = np.concatenate((X, noise), axis=1)
X_noisy = X_noisy[np.random.permutation(X.shape[0])]
np.random.shuffle(X_noisy.T)

run_svm_cv(X_noisy, y)

#%%
# Q5 - addition

def run_svm_cv(_X, _y, random_state=42):
    rng = np.random.RandomState(random_state)
    indices = rng.permutation(_X.shape[0])
    train_idx, test_idx = indices[:_X.shape[0] // 2], indices[_X.shape[0] // 2:]
    X_train, X_test = _X[train_idx, :], _X[test_idx, :]
    y_train, y_test = _y[train_idx], _y[test_idx]

    parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 5))}
    svr = svm.SVC()
    clf = GridSearchCV(svr, parameters)
    clf.fit(X_train, y_train)

    return clf.score(X_test, y_test)

noise_dims = np.arange(10, 1001, 25)
test_scores = []

rng = np.random.RandomState(42)
for n_noise in noise_dims:
    noise = rng.randn(X.shape[0], n_noise)
    X_aug = np.concatenate((X, noise), axis=1)
    score = run_svm_cv(X_aug, y)
    test_scores.append(score)

# Plotting a graph
plt.figure(figsize=(10,6))
plt.plot(noise_dims, test_scores, color='red')
plt.title("Influence du nombre de variables de nuisance sur l'accuracy")
plt.xlabel("Nombre de variables de nuisance")
plt.ylabel("Accuracy sur l'échantillon de test")
plt.grid(True)
plt.show()

# %%
# Q6
#comparing time and accuracity for components from 2 to 200 

import time as tm

# all the numbers to check
components_list = list(range(2, 201, 10))

# for autosave
filename = "pca_results_autosave_copy_2.csv"
if os.path.exists(filename):
    # upload exister results
    components_done, accuracies, times = [], [], []
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            components_done.append(int(row["n_components"]))
            accuracies.append(float(row["accuracy"]))
            times.append(float(row["time_seconds"]))
    print(f"Composants {len(components_done)} chargés déjà traités")
else:
    components_done, accuracies, times = [], [], []

print("Score apres reduction de dimension (svd_solver='randomized')\n")


accuracies = []  # for test score
times = []       # for time
print("Score apres reduction de dimension (svd_solver='randomized')\n")

for n_components in components_list:
# skip what already have
    if n_components in components_done:
        continue

    print(f"Nombre de composantes PCA: {n_components}")
    
    #PCA с randomized solver - what prof asks
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
    X_pca = pca.fit_transform(X_noisy)
    
    t0 = tm.time()
    # hoping that, run_svm_cv giving back test_score
    test_score = run_svm_cv(X_pca, y)  
    elapsed = tm.time() - t0
    
    components_done.append(n_components)
    accuracies.append(test_score)  # save accuracy
    times.append(elapsed)          # save time
    
    print(f"Test score: {test_score:.3f}")
    print(f"Temps de calcul: {elapsed:.3f} secondes\n")

     # autosave
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_components", "accuracy", "time_seconds"])
        for n, acc, t in zip(components_done, accuracies, times):
            writer.writerow([n, acc, t])
    print(f"Les résultats intermédiaires sont enregistrés dans {filename}\n")


# graph
fig, ax1 = plt.subplots(figsize=(8,5))

# Précision
color = 'tab:blue'
ax1.set_xlabel('Nombre de composantes PCA')
ax1.set_ylabel('Précision', color=color)
ax1.plot(components_list, accuracies, marker='o', color=color, label='Précision')
ax1.tick_params(axis='y', labelcolor=color)

# Temps
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Temps (s)', color=color)
ax2.plot(components_list, times, marker='s', linestyle='--', color=color, label='Temps')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_yscale('log')  # logarithmic time scale

plt.title("Influence du nombre de composantes PCA sur la précision et le temps")
plt.show()
# %%
# q7

####################################################################
# Download the data and unzip; then load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4,
                              color=True, funneled=False, slice_=None,
                              download_if_missing=True)
# data_home='.'

# introspect the images arrays to find the shapes (for plotting)
images = lfw_people.images
n_samples, h, w, n_colors = images.shape

# the label to predict is the id of the person
target_names = lfw_people.target_names.tolist()

####################################################################
# Pick a pair to classify such as
# names = ['Donald Rumsfeld', 'Colin Powell']
names = ['Donald Rumsfeld', 'Colin Powell']

idx0 = (lfw_people.target == target_names.index(names[0]))
idx1 = (lfw_people.target == target_names.index(names[1]))
images = np.r_[images[idx0], images[idx1]]
n_samples = images.shape[0]
y = np.r_[np.zeros(np.sum(idx0)), np.ones(np.sum(idx1))].astype(int)

#teacher's code (bad version)
#y = np.r_[np.zeros(np.sum(idx0)), np.ones(np.sum(idx1))].astype(np.int)

# plot a sample set of the data
plot_gallery(images, np.arange(12))
plt.show()

# separating data first
indices = np.random.permutation(X.shape[0])
train_idx, test_idx = indices[:X.shape[0] // 2], indices[X.shape[0] // 2:]
X_train, X_test = X[train_idx, :], X[test_idx, :]

# normalisation just for train
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

#%%
# Q4
print("--- Linear kernel ---")
print("Fitting the classifier to the training set")
t0 = time()

# fit a classifier (linear) and test all the Cs
Cs = 10. ** np.arange(-5, 6)
scores = []
for C in Cs:
    clf = SVC(kernel="linear", C=C)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))

ind = np.argmax(scores)
print("Best C: {}".format(Cs[ind]))

plt.figure()
plt.plot(Cs, scores)
plt.xlabel("Parametres de regularisation C")
plt.ylabel("Scores d'apprentissage")
plt.xscale("log")
plt.tight_layout()
plt.show()
print("Best score: {}".format(np.max(scores)))

print("Predicting the people names on the testing set")
t0 = time()


#%%
# let's calculate the error
errors = []
for C in Cs:
    clf = SVC(kernel="linear", C=C)
    clf.fit(X_train, y_train)
    errors.append(1 - clf.score(X_test, y_test))

# look for the min of the error
best_ind = np.argmin(errors)
best_C = Cs[best_ind]
best_error = errors[best_ind]

plt.figure()
plt.plot(Cs, errors, marker="o")

# plot error
plt.scatter(best_C, best_error, color="red", s=100, zorder=5)

plt.xscale("log")
plt.xlabel("Paramètre de régularisation C")
plt.ylabel("Erreur de prédiction")
plt.title("Influence de C sur la performance")
plt.grid(True)
plt.tight_layout()
plt.show()


print("Best C: {}".format(Cs[best_ind]))
print("Best error: {}".format(np.min(errors)))
print("Best accuracy(score): {}".format(1 - np.min(errors)))
t0 = time()

#%%
# confusion matrix
# small C
clf_small = SVC(kernel="linear", C=1e-5)
clf_small.fit(X_train, y_train)
y_pred_small = clf_small.predict(X_test)

cm_small = confusion_matrix(y_test, y_pred_small, labels=clf_small.classes_)
disp_small = ConfusionMatrixDisplay(confusion_matrix=cm_small, display_labels=clf_small.classes_)

# big C
clf_large = SVC(kernel="linear", C=1e5)
clf_large.fit(X_train, y_train)
y_pred_large = clf_large.predict(X_test)

cm_large = confusion_matrix(y_test, y_pred_large, labels=clf_large.classes_)
disp_large = ConfusionMatrixDisplay(confusion_matrix=cm_large, display_labels=clf_large.classes_)

# rows
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# 0 = Donald Rumsfeld, 1 = Colin Powell)
class_names = ['Donald Rumsfeld', 'Colin Powell']

disp_small = ConfusionMatrixDisplay(confusion_matrix=cm_small,
                                    display_labels=class_names)
disp_small.plot(ax=axes[0], cmap="Blues", xticks_rotation='vertical', colorbar=False)
axes[0].set_title("Confusion matrix (C=1e-5)")
disp_large = ConfusionMatrixDisplay(confusion_matrix=cm_large,
                                    display_labels=class_names)
disp_large.plot(ax=axes[1], cmap="Blues", xticks_rotation='vertical', colorbar=False)
axes[1].set_title("Confusion matrix (C=1e5)")

plt.tight_layout()
plt.show()

#%% the same thing but in numbers
print("=== Classification report (C=1e-5) ===")
print(classification_report(y_test, y_pred_small, target_names=class_names))

print("\n=== Classification report (C=1e5) ===")
print(classification_report(y_test, y_pred_large, target_names=class_names))

#%%
# predict labels for the X_test images with the best classifier
clf = SVC(kernel="linear", C=Cs[ind])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("done in %0.3fs" % (time() - t0))
# The chance level is the accuracy that will be reached when constantly predicting the majority class.
print("Chance level : %s" % max(np.mean(y), 1. - np.mean(y)))
print("Accuracy : %s" % clf.score(X_test, y_test))

#%%
####################################################################
# Qualitative evaluation of the predictions using matplotlib

prediction_titles = [title(y_pred[i], y_test[i], names)
                     for i in range(y_pred.shape[0])]

plot_gallery(images_test, prediction_titles)
plt.show()

####################################################################
# Look at the coefficients
plt.figure()
plt.imshow(np.reshape(clf.coef_, (h, w)))
plt.show()

#%% 
# Qualitative evaluation of the predictions using matplotlib
####################################################################
# Look at the coefficients pour C=1e-5 et C=1e5
coef_small = clf_small.coef_.ravel()
coef_large = clf_large.coef_.ravel()


fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(coef_small.reshape(h, w), cmap=plt.cm.seismic, interpolation="nearest")
axes[0].set_title("Coefficients (C=1e-5)")

axes[1].imshow(coef_large.reshape(h, w), cmap=plt.cm.seismic, interpolation="nearest")
axes[1].set_title("Coefficients (C=1e5)")

plt.tight_layout()
plt.show()

#%%
# Q5

def run_svm_cv(_X, _y):
    _indices = np.random.permutation(_X.shape[0])
    _train_idx, _test_idx = _indices[:_X.shape[0] // 2], _indices[_X.shape[0] // 2:]
    _X_train, _X_test = _X[_train_idx, :], _X[_test_idx, :]
    _y_train, _y_test = _y[_train_idx], _y[_test_idx]

    _parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 5))}
    _svr = svm.SVC()
    _clf_linear = GridSearchCV(_svr, _parameters)
    _clf_linear.fit(_X_train, _y_train)

    print('Generalization score for linear kernel: %s, %s \n' %
          (_clf_linear.score(_X_train, _y_train), _clf_linear.score(_X_test, _y_test)))

print("Score sans variable de nuisance")
run_svm_cv(X, y)

print("Score avec variable de nuisance")
n_features = X.shape[1]
# We add nuisance variables
sigma = 1
noise = sigma * np.random.randn(n_samples, 300, ) 
# with gaussian coefficients of std sigma
X_noisy = np.concatenate((X, noise), axis=1)
X_noisy = X_noisy[np.random.permutation(X.shape[0])]
np.random.shuffle(X_noisy.T)

run_svm_cv(X_noisy, y)

# %%
# Q6
#comparing time and accuracity for components from 2 to 200 

import time as tm

# all the numbers to check
components_list = list(range(2, 201, 10))

# for autosave
filename = "pca_results_autosave_copy_2.csv"
if os.path.exists(filename):
    # upload exister results
    components_done, accuracies, times = [], [], []
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            components_done.append(int(row["n_components"]))
            accuracies.append(float(row["accuracy"]))
            times.append(float(row["time_seconds"]))
    print(f"Composants {len(components_done)} chargés déjà traités")
else:
    components_done, accuracies, times = [], [], []

print("Score apres reduction de dimension (svd_solver='randomized')\n")


accuracies = []  # for test score
times = []       # for time
print("Score apres reduction de dimension (svd_solver='randomized')\n")

for n_components in components_list:
# skip what already have
    if n_components in components_done:
        continue

    print(f"Nombre de composantes PCA: {n_components}")
    
    #PCA с randomized solver - what prof asks
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
    X_pca = pca.fit_transform(X_noisy)
    
    t0 = tm.time()
    # hoping that, run_svm_cv giving back test_score
    test_score = run_svm_cv(X_pca, y)  
    elapsed = tm.time() - t0
    
    components_done.append(n_components)
    accuracies.append(test_score)  # save accuracy
    times.append(elapsed)          # save time
    
    print(f"Test score: {test_score:.3f}")
    print(f"Temps de calcul: {elapsed:.3f} secondes\n")

     # autosave
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_components", "accuracy", "time_seconds"])
        for n, acc, t in zip(components_done, accuracies, times):
            writer.writerow([n, acc, t])
    print(f"Les résultats intermédiaires sont enregistrés dans {filename}\n")


# graph
fig, ax1 = plt.subplots(figsize=(8,5))

# Précision
color = 'tab:blue'
ax1.set_xlabel('Nombre de composantes PCA')
ax1.set_ylabel('Précision', color=color)
ax1.plot(components_list, accuracies, marker='o', color=color, label='Précision')
ax1.tick_params(axis='y', labelcolor=color)

# Temps
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Temps (s)', color=color)
ax2.plot(components_list, times, marker='s', linestyle='--', color=color, label='Temps')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_yscale('log')  # logarithmic time scale

plt.title("Influence du nombre de composantes PCA sur la précision et le temps")
plt.show()