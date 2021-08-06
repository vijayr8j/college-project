from django import forms
from .models import *


class liverForm(forms.ModelForm):
    class Meta():
        model=liverModel
        fields=['Hepatitis_c_vir_antibody','Aspartate_Transaminase','chronic_renal_insufficenc','Nonalcoholic_steatohepatitis','hemochromatosis','oxygen_saturation']
