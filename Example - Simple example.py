import numpy
from sklearn.svm.classes import SVC
import Features_manager
import Database_manager
from sklearn.metrics.classification import precision_recall_fscore_support, accuracy_score
from sklearn.cross_validation import KFold

# initialize database_manager
database_manager = Database_manager.make_database_manager()
# initialize feature_manager
feature_manager = Features_manager.make_feature_manager()

# recover tweets
tweets = numpy.array(database_manager.return_tweets())
labels = numpy.array(feature_manager.get_label(tweets))


# create the feature space with all available features
X,feature_names,feature_type_indexes=feature_manager.create_feature_space(tweets)

feature_type=feature_manager.get_availablefeaturetypes()

print("features types:", feature_type)
print("features names:", feature_names)
print("feature space dimension:", X.shape)

golden=[]
predict=[]
kf = KFold(len(tweets),n_folds=5, random_state=True)
for index_train, index_test in kf:

    clf = SVC(kernel="linear")

    clf.fit(X[index_train],labels[index_train])
    test_predict = clf.predict(X[index_test])

    golden=numpy.concatenate((golden,labels[index_test]), axis=0)
    predict=numpy.concatenate((predict,test_predict), axis=0)

prec, recall, f, support = precision_recall_fscore_support(
golden,
predict,
beta=1)

accuracy = accuracy_score(
golden,
predict
)

print(prec, recall, f, support)
print(accuracy)
