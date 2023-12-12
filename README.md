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

fiber_ID: Fiber ID identifies the fiber that pointed the light at the focal plane in each observation. Each fiber collects the light from a different object, allowing SDSS to observe multiple objects simultaneously.
