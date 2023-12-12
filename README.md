# GalacticHarmony

Jacob Yoon & Brandon Dunz
For DATA6150-02

Machine Learning Journey in Space Object Classification


The data set star_classification.csv contains 100,000 different spectral objects in space. 
| Galaxies |      Stars    |  Quasars |
|----------|:-------------:|------:|
| 59,445   |  21,594       | 18,961|

The purpose of this code is to use machine learning models to accurately predict which spectral object we’re investigating based on initial spectral properties.
The variables u, g, r, i, z are photometric filters used in the SDSS system to measure the amount of light from objects in different wavelength bands. Each filter corresponds to a different color of light, with u being in the ultraviolet range, g in green, r in red, i in near-infrared, and z in infrared.
The variable redshift value is based on the increase in wavelength of light emitted by the object due to its motion away from or towards the observer. It is a measure of how much the universe has expanded since the light was emitted by the object.


# Dataset Column information

**obj_ID**: Object Identifier is a unique value that identifies the object in the image catalog used by the Sloan Digital Sky Survey (SDSS) data archive.

**alpha**: Right Ascension angle (in degrees) at J2000 epoch, which is a coordinate system commonly used in astronomy to describe the position of celestial objects in the sky.

**delta**: Declination angle (in degrees) at J2000 epoch, which is another coordinate system commonly used in astronomy to describe the position of celestial objects in the sky.

**u, g, r, i, z**: These are photometric filters used in the SDSS system to measure the amount of light from objects in different wavelength bands. Each filter corresponds to a different color of light, with u being in the ultraviolet range, g in green, r in red, i in near-infrared, and z in infrared.

**run_ID**: Run Number is used to identify the specific scan of the sky made by SDSS. Each scan covers a specific area of the sky and is assigned a unique run number.

**rereun_ID**: Rerun Number is used to specify how the image was processed, such as the version of the software or the calibration used to create the image.

**cam_col**: Camera column is used to identify the scanline within the run. Each scan is divided into multiple camera columns to cover a larger area of the sky.

**field_ID**: Field Number is used to identify each field in the scan, which is a smaller area within the camera column.

**spec_obj_ID**: Unique ID used for optical spectroscopic objects. This means that two different observations with the same spec_obj_ID must share the output class, which is either a galaxy, star, or quasar.

**class**: Object class is the classification assigned to the object based on its spectral properties, which can be a galaxy, star, or quasar.

**redshift**: Redshift value is based on the increase in wavelength of light emitted by the object due to its motion away from or towards the observer. It is a measure of how much the universe has expanded since the light was emitted by the object.

**plate**: Plate ID identifies each plate used in the SDSS spectroscopic survey. Each plate contains multiple fibers that collect the light from different objects.

**MJD**: Modified Julian Date is used to indicate when a given piece of SDSS data was taken. It is a modified version of the Julian Date, which is a way to represent dates and times in astronomy.

**fiber_ID**: Fiber ID identifies the fiber that pointed the light at the focal plane in each observation. Each fiber collects the light from a different object, allowing SDSS to observe multiple objects simultaneously.


# Dataset Header
|   | obj_ID         | alpha      | delta      | u       | g       | r       | i       | z       | run_ID | rerun_ID | cam_col | field_ID | spec_obj_ID   | class  | redshift | plate | MJD   | fiber_ID |
|---|----------------|------------|------------|---------|---------|---------|---------|---------|--------|----------|---------|----------|---------------|--------|----------|-------|-------|----------|
| 0 | 1.237661e+18   | 135.689107 | 32.494632  | 23.87882| 22.27530| 20.39501| 19.16573| 18.79371| 3606   | 301      | 2       | 79       | 6.543777e+18  | GALAXY | 0.634794 | 5812  | 56354 | 171      |
| 1 | 1.237665e+18   | 144.826101 | 31.274185  | 24.77759| 22.83188| 22.58444| 21.16812| 21.61427| 4518   | 301      | 5       | 119      | 1.176014e+19  | GALAXY | 0.779136 | 10445 | 58158 | 427      |
| 2 | 1.237661e+18   | 142.188790 | 35.582444  | 25.26307| 22.66389| 20.60976| 19.34857| 18.94827| 3606   | 301      | 2       | 120      | 5.152200e+18  | GALAXY | 0.644195 | 4576  | 55592 | 299      |
| 3 | 1.237663e+18   | 338.741038 | -0.402828  | 22.13682| 23.77656| 21.61162| 20.50454| 19.25010| 4192   | 301      | 3       | 214      | 1.030107e+19  | GALAXY | 0.932346 | 9149  | 58039 | 775      |
| 4 | 1.237680e+18   | 345.282593 | 21.183866  | 19.43718| 17.58028| 16.49747| 15.97711| 15.54461| 8102   | 301      | 3       | 137      | 6.891865e+18  | GALAXY | 0.116123 | 6121  | 56187 | 842      |

# Models and Accuracies
| Model                   | Accuracy |
|-------------------------|----------|
| Logistic Regression     | 0.9559   |
| Random Forest           | 0.9770   |
| Support Vector Machine  | 0.9615   |
| Decision Tree           | 0.9654   |
| Neural Network          | 0.9722   |
| Soft Voting Ensemble    | 0.9718   |
| Hard Voting Ensemble    | 0.9710   |


```
Model: Logistic Regression
Accuracy: 0.9559
Classification Report:
              precision    recall  f1-score   support

      GALAXY       0.96      0.96      0.96     17845
         QSO       0.95      0.88      0.91      5700
        STAR       0.95      1.00      0.97      6455

    accuracy                           0.96     30000
   macro avg       0.95      0.95      0.95     30000
weighted avg       0.96      0.96      0.96     30000


--------------------------------------------------

Model: Random Forest
Accuracy: 0.9770
Classification Report:
              precision    recall  f1-score   support

      GALAXY       0.98      0.99      0.98     17845
         QSO       0.96      0.92      0.94      5700
        STAR       0.99      1.00      1.00      6455

    accuracy                           0.98     30000
   macro avg       0.98      0.97      0.97     30000
weighted avg       0.98      0.98      0.98     30000


--------------------------------------------------

Model: Support Vector Machine
Accuracy: 0.9615
Classification Report:
              precision    recall  f1-score   support

      GALAXY       0.97      0.97      0.97     17845
         QSO       0.97      0.91      0.94      5700
        STAR       0.94      1.00      0.97      6455

    accuracy                           0.96     30000
   macro avg       0.96      0.96      0.96     30000
weighted avg       0.96      0.96      0.96     30000


--------------------------------------------------

Model: Decision Tree
Accuracy: 0.9654
Classification Report:
              precision    recall  f1-score   support

      GALAXY       0.95      0.99      0.97     17845
         QSO       0.97      0.85      0.91      5700
        STAR       1.00      1.00      1.00      6455

    accuracy                           0.97     30000
   macro avg       0.97      0.95      0.96     30000
weighted avg       0.97      0.97      0.96     30000


--------------------------------------------------

Model: Neural Network
Accuracy: 0.9722
Classification Report:
              precision    recall  f1-score   support

      GALAXY       0.98      0.98      0.98     17845
         QSO       0.96      0.93      0.94      5700
        STAR       0.97      1.00      0.98      6455

    accuracy                           0.97     30000
   macro avg       0.97      0.97      0.97     30000
weighted avg       0.97      0.97      0.97     30000


--------------------------------------------------

Model: Soft voting Ensemble
Ensemble Model Accuracy:  0.9717666666666667
Ensemble Model Classification Report: 
               precision    recall  f1-score   support

      GALAXY       0.97      0.98      0.98     17845
         QSO       0.97      0.91      0.94      5700
        STAR       0.98      1.00      0.99      6455

    accuracy                           0.97     30000
   macro avg       0.97      0.96      0.97     30000
weighted avg       0.97      0.97      0.97     30000

--------------------------------------------------

Model: Hard voting Ensemble
Ensemble Model Accuracy:  0.9710333333333333
Ensemble Model Classification Report: 
               precision    recall  f1-score   support

      GALAXY       0.97      0.98      0.98     17845
         QSO       0.97      0.91      0.94      5700
        STAR       0.97      1.00      0.99      6455

    accuracy                           0.97     30000
   macro avg       0.97      0.96      0.97     30000
weighted avg       0.97      0.97      0.97     30000
```
