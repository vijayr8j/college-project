import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time
from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, HttpRequest
from django.shortcuts import render, redirect
#from .forms import *
import numpy as np
import pandas as pd
import time

from django.contrib import messages
from django.shortcuts import render
from django.urls import reverse_lazy
from django.urls import reverse
from django.http import HttpResponse
from django.views.generic import (View,TemplateView,
ListView,DetailView,
CreateView,DeleteView,
UpdateView)
from . import models
from .forms import *
from django.core.files.storage import FileSystemStorage
#from topicApp.Topicfun import Topic
from liverApp.liverfun import liver

class dataUploadView(View):
    form_class = liverForm
    success_url = reverse_lazy('success')
    template_name = 'create.html'
    failure_url= reverse_lazy('fail')
    filenot_url= reverse_lazy('filenot')
    def get(self, request, *args, **kwargs):
        form = self.form_class()
        return render(request, self.template_name, {'form': form})
    def post(self, request, *args, **kwargs):
        #print('inside post')
        form = self.form_class(request.POST, request.FILES)
        #print('inside form')
        if form.is_valid():
            form.save()
            data_bgr= request.POST.get('Hepatitis_c_vir_antibody')
            data_bu=request.POST.get('Aspartate_Transaminase')
            data_ch=request.POST.get('chronic_renal_insufficenc')
            data_sc=request.POST.get('Nonalcoholic_steatohepatitis')
            data_pcv=request.POST.get('hemochromatosis')
            data_wc=request.POST.get('oxygen_saturation')


            #print (data)


            dataset=pd.read_csv("finalpree.csv",index_col=None)
            df2=dataset
            import warnings
            warnings.simplefilter(action='ignore', category=FutureWarning)
            #warnings.simplefilter(action='ignore', category=Conve)

            indep_X=df2.drop('class_attribute', 1)
            dep_Y=df2['class_attribute']
            obj=liver()
            selectk_feature=obj.selectkbest(indep_X,dep_Y)
            #rfe_feature=obj.rfeFeature(indep_X,dep_Y)

            #selectk_pca=opca(selectk_feature,dep_Y)
            #rfe_pca=pca(rfe_feature,dep_Y)

            classifier,Accuracy,report,X_test,y_test,cm=obj.svm(selectk_feature,indep_X,dep_Y)
            data_chh=classifier.predict([[data_bgr,data_bu,data_sc,data_pcv,data_wc]])

            return render(request, "succ_msg.html", {'data_bgr':data_bgr,'data_bu':data_bu,'data_sc':data_sc,'data_pcv':data_pcv,'data_wc':data_wc,'data_ch':data_ch})


        else:
            return redirect(self.failure_url)
