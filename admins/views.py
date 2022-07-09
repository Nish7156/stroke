from django.shortcuts import render,HttpResponse
from django.contrib import messages
from users.models import UserRegistrationModel
from users.algorithms.MyAlgorithms import Algorithms

# Create your views here.

def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/AdminHome.html')
        elif usrid == 'Admin' and pswd == 'Admin':
            return render(request, 'admins/AdminHome.html')
        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html', {})


def AdminHome(request):
    return render(request, 'admins/AdminHome.html')


def ViewRegisteredUsers(request):
    data = UserRegistrationModel.objects.all()
    return render(request, 'admins/RegisteredUsers.html', {'data': data})


def AdminActivaUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        UserRegistrationModel.objects.filter(id=id).update(status=status)
        data = UserRegistrationModel.objects.all()
        return render(request, 'admins/RegisteredUsers.html', {'data': data})


def AdminResults(request):
    obj = Algorithms()
    lg_rslt = obj.startLogisticregression()
    knn_rslt = obj.startKNN()
    nb_result = obj.startNaivebayes()
    dt_result = obj.startDecisionTree()
    rf_result = obj.startRandomForest()
    svm_result = obj.startSVM()
    mlp_result = obj.startMultiLayerPerceptron()
    dl_result = obj.startDeepLearning()
    myDict = {'Logistic Regression': lg_rslt,
              'K-Nearest Neighbors': knn_rslt,
              'Naive Bayes': nb_result,
              'Decision Tree': dt_result,
              'Random Forest': rf_result,
              'Multi-layer Perceptron': mlp_result,
              'Support Vector Machine': svm_result,
              'Deep Learning': dl_result
              }
    return render(request, 'admins/AdminResults.html',{'myDict':myDict})