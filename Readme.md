# Gender and Ethnicity Predictor
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



Pretrained models on 60k synthesized usernames for gender using baby names/110k usernames for ethnicity using LSTM inspired by AndrewNg DeepLearning.ai which works best for Indian names with > 88% accuracy which can predict for huge datasets/scale using pandas + tensorflow(keras).

Save time training it yourself by reusing this pre-trained model straight from github to do fuzzy detection on usernames. For improved performance hire a Data Scientist to improve the model or use transfer learning to retrain it. Pull requests are welcome!

# Sample Output

```
0             RajSood25
1           mukta57cute
2    burnvipulkumarfire
3      priyasubramanium
array([[male, female]]) probability
[[0.71 0.28]
 [0.11 0.88]
 [0.99 0.  ]
 [0.   0.99]]
```


```
0        RajkumarSood
1             mukta57
2           johnsmith
3         KeanuReeves
4               Elias
5    priyasubramanium
6           Devashish

array([[other, indian ethnicity]]) probability
[[0.09 0.9 ]
 [0.33 0.66]
 [0.81 0.18]
 [0.88 0.11]
 [0.96 0.03]
 [0.05 0.94]
 [0.07 0.92]]
 ```

Modify the sample names in the python script with your own text and try!

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

You can retrain the model to work for gender and ethnicity of any country using the string_vectorizer. Need GPUs to train for worldwide.

Star this repo to add to your favorite repos or fork the code!
If you have better datasets that I can train on do share!

