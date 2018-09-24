# Gender and Ethnicity Predictor
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



Pretrained models on 60k synthesized usernames for gender using baby names/110k usernames for ethnicity using LSTM inspired by AndrewNg DeepLearning.ai which works best for Indian names with > 88% accuracy which can predict for huge datasets/scale using pandas + tensorflow(keras).

Requirements

```
Python 3.6
Pandas 
Numpy
Tensorflow 1.10
```

View in [Browser](https://github.com/devssh/GenderEthnicityDetector/blob/master/PredictGenderAndEthnicity.ipynb) or Run
```
python predict_gender_and_ethnicity.py
```

Modify the sample names in the python script with your own text and try!

You can retrain the model to work for gender and ethnicity of any country using the string_vectorizer. Need GPUs to train for worldwide.

Star this repo to add to your favorite repos or fork the code!


