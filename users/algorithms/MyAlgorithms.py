from django_pandas.io import read_frame
from users.models import GLCMtableModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score

import matplotlib.pyplot as plt
class Algorithms:
    dataset = GLCMtableModel.objects.all()
    df = read_frame(dataset)
    X = df.iloc[:,1:7].values
    y = df.iloc[:,-1].values
    X_train,X_test, y_train,y_test = train_test_split(X,y, test_size=0.25, random_state=0)

    def startLogisticregression(self):
        print('Starting Logistic Regression')
        from sklearn.linear_model import LogisticRegression
        reg = LogisticRegression()
        reg.fit(self.X_train,self.y_train)
        y_pred = reg.predict(self.X_test)
        lg_acc = accuracy_score(self.y_test,y_pred)
        lg_precc= precision_score(self.y_test,y_pred)
        lg_recall = recall_score(self.y_test,y_pred)
        lg_f1Scr = f1_score(self.y_test,y_pred)
        cm = confusion_matrix(self.y_test,y_pred)
        print('Confusion Matrix ',cm)
        from sklearn.metrics import plot_confusion_matrix
        #plot_confusion_matrix(reg, self.y_test,y_pred)
        #plt.show()
        print("LG Accuracy ",lg_acc,lg_precc,lg_recall,lg_f1Scr)
        return [lg_acc,lg_precc,lg_recall,lg_f1Scr]

    def startKNN(self):
        print('Starting K Nearest Neighbors')
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier()
        knn.fit(self.X_train, self.y_train)
        y_pred = knn.predict(self.X_test)
        knn_acc = accuracy_score(self.y_test, y_pred)
        knn_precc = precision_score(self.y_test, y_pred)
        knn_recall = recall_score(self.y_test, y_pred)
        knn_f1Scr = f1_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        print('Confusion Matrix ', cm)
        from sklearn.metrics import plot_confusion_matrix
        # plot_confusion_matrix(reg, self.y_test,y_pred)
        # plt.show()
        print("KNN Accuracy ", knn_acc,knn_precc,knn_recall,knn_f1Scr)
        return [knn_acc,knn_precc,knn_recall,knn_f1Scr]

    def startNaivebayes(self):
        print('Starting NaiVe bayes')
        from sklearn.naive_bayes import GaussianNB
        nb = GaussianNB()
        nb.fit(self.X_train, self.y_train)
        y_pred = nb.predict(self.X_test)
        nb_acc = accuracy_score(self.y_test, y_pred)
        nb_precc = precision_score(self.y_test, y_pred)
        nb_recall = recall_score(self.y_test, y_pred)
        nb_f1Scr = f1_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        print('Confusion Matrix ', cm)
        from sklearn.metrics import plot_confusion_matrix
        # plot_confusion_matrix(reg, self.y_test,y_pred)
        # plt.show()
        print("Naive bayes Accuracy ", nb_acc, nb_precc, nb_recall, nb_f1Scr)
        return [nb_acc, nb_precc, nb_recall, nb_f1Scr]

    def startDecisionTree(self):
        print('Starting Decision tree')
        from sklearn.tree import DecisionTreeClassifier
        dt = DecisionTreeClassifier()
        dt.fit(self.X_train, self.y_train)
        y_pred = dt.predict(self.X_test)
        dt_acc = accuracy_score(self.y_test, y_pred)
        dt_precc = precision_score(self.y_test, y_pred)
        dt_recall = recall_score(self.y_test, y_pred)
        dt_f1Scr = f1_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        print('Confusion Matrix ', cm)
        from sklearn.metrics import plot_confusion_matrix
        # plot_confusion_matrix(reg, self.y_test,y_pred)
        # plt.show()
        print("Decision Tree Accuracy ", dt_acc, dt_precc, dt_recall, dt_f1Scr)
        return [dt_acc, dt_precc, dt_recall, dt_f1Scr]

    def startRandomForest(self):
        print('Starting Random Forest')
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier()
        rf.fit(self.X_train, self.y_train)
        y_pred = rf.predict(self.X_test)
        rf_acc = accuracy_score(self.y_test, y_pred)
        rf_precc = precision_score(self.y_test, y_pred)
        rf_recall = recall_score(self.y_test, y_pred)
        rf_f1Scr = f1_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        print('Confusion Matrix ', cm)
        from sklearn.metrics import plot_confusion_matrix
        # plot_confusion_matrix(reg, self.y_test,y_pred)
        # plt.show()
        print("Random Forest Accuracy ", rf_acc, rf_precc, rf_recall, rf_f1Scr)
        return [rf_acc, rf_precc, rf_recall, rf_f1Scr]

    def startSVM(self):
        print('Starting Support Vector Machine')
        from sklearn.svm import SVC
        cls = SVC()
        cls.fit(self.X_train, self.y_train)
        y_pred = cls.predict(self.X_test)
        svm_acc = accuracy_score(self.y_test, y_pred)
        svm_precc = precision_score(self.y_test, y_pred)
        svm_recall = recall_score(self.y_test, y_pred)
        svm_f1Scr = f1_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        print('Confusion Matrix ', cm)
        from sklearn.metrics import plot_confusion_matrix
        # plot_confusion_matrix(reg, self.y_test,y_pred)
        # plt.show()
        print("Support Vector Machine Accuracy ", svm_acc, svm_precc, svm_recall, svm_f1Scr)
        return [svm_acc, svm_precc, svm_recall, svm_f1Scr]

    def startMultiLayerPerceptron(self):
        print('Starting Multi layer Perceptron')
        from sklearn.neural_network import MLPClassifier
        cls = MLPClassifier(activation='relu')
        cls.fit(self.X_train, self.y_train)
        y_pred = cls.predict(self.X_test)
        mlp_acc = accuracy_score(self.y_test, y_pred)
        mlp_precc = precision_score(self.y_test, y_pred)
        mlp_recall = recall_score(self.y_test, y_pred)
        mlp_f1Scr = f1_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        print('Confusion Matrix ', cm)
        from sklearn.metrics import plot_confusion_matrix
        # plot_confusion_matrix(reg, self.y_test,y_pred)
        # plt.show()
        print("Multi-layer Perceptron  Accuracy ", mlp_acc, mlp_precc, mlp_recall, mlp_f1Scr)
        return [mlp_acc, mlp_precc, mlp_recall, mlp_f1Scr]

    def startDeepLearning(self):
        print("Deep learning started")
        import keras
        from keras.models import Sequential
        from keras.layers import Dense
        classifier = Sequential()
        classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=6))
        classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
        classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print(classifier.summary())
        classifier.fit(self.X_train, self.y_train, batch_size=10, nb_epoch=100)
        y_pred = classifier.predict(self.X_test)
        y_pred = (y_pred > 0.5)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(self.y_test, y_pred)
        dl_acc = accuracy_score(self.y_test, y_pred)
        dl_precc = precision_score(self.y_test, y_pred)
        dl_recall = recall_score(self.y_test, y_pred)
        dl_f1Scr = f1_score(self.y_test, y_pred)

        print("Confusion Matrix ", cm)
        print('Deep Learning ',dl_acc, dl_precc, dl_recall, dl_f1Scr)
        return [dl_acc, dl_precc, dl_recall, dl_f1Scr]

