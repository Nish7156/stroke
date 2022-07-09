from django.shortcuts import render,HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel,GLCMtableModel
from django.core.files.storage import FileSystemStorage
from .utility.GetImageData import ProcessCtScanImage
import pandas as pd
from .algorithms.MyAlgorithms import Algorithms
# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})
def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})
def UserHome(request):
    return render(request, 'users/UserHome.html', {})
def UserUploadImageForm(request):
    return render(request, 'users/UserUploadForm.html',{})

def UploadImageAction(request):
    image_file = request.FILES['file']
    # let's check if it is a csv file
    if not image_file.name.endswith('.jpg'):
        messages.error(request, 'THIS IS NOT A JPG  FILE')
    fs = FileSystemStorage()
    filename = fs.save(image_file.name, image_file)
    detect_filename = fs.save(image_file.name, image_file)
    uploaded_file_url = fs.url(filename)
    obj = ProcessCtScanImage()
    data = obj.preprocess(filename)
    GLCMtableModel.objects.create(contrast=data['contrast'],dissimilarity=data['dissimilarity'],homogeneity=data['homogeneity'],
                                  energy=data['energy'],correlation=data['correlation'],ass=data['ASM'],lableclass=data['lblclass'])
    return render(request, 'users/UserUploadForm.html',{})

def UserViewData(request):
    data = GLCMtableModel.objects.all()
    #data = pd.DataFrame(data)
    #for x in data:
        #print(x.ass)
    return render(request, 'users/ViewDataset.html',{'data':data})

def UsersAlgorithms(request):
    obj = Algorithms()
    lg_rslt = obj.startLogisticregression()
    knn_rslt = obj.startKNN()
    nb_result = obj.startNaivebayes()
    dt_result = obj.startDecisionTree()
    rf_result = obj.startRandomForest()
    svm_result = obj.startSVM()
    mlp_result = obj.startMultiLayerPerceptron()
    dl_result = obj.startDeepLearning()
    myDict = {'Logistic Regression':lg_rslt,
              'K-Nearest Neighbors':knn_rslt,
              'Naive Bayes':nb_result,
              'Decision Tree':dt_result,
              'Random Forest':rf_result,
              'Multi-layer Perceptron':mlp_result,
              'Support Vector Machine':svm_result,
              'Deep Learning':dl_result
              }
    return render(request, 'users/UserAlgorithmResults.html',{'myDict':myDict})