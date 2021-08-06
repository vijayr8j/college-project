from django.db import models

# Create your models here.
class liverModel(models.Model):

    Hepatitis_c_vir_antibody=models.FloatField()
    Aspartate_Transaminase=models.FloatField()
    chronic_renal_insufficenc=models.IntegerField()
    Nonalcoholic_steatohepatitis=models.FloatField()
    hemochromatosis=models.FloatField()
    oxygen_saturation=models.FloatField()
