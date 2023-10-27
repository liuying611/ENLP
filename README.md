### Description of the Files
Root Folder: ENLP

- EDA.py: modularized functions for cleaning, statistic, and other operation needed;
- modernNN.py: modularized models, LSTM and CNN;
- workflow_dataprocessing.ipynb: steps to clean raw data, requires ```from EDA import *```;
- workflow_train_NN*.ipynb: steps to train modern neural network models, requires ```from modernNN import *```,
- workflow_usual_model.ipynb: steps to train traditional machine learning models.
- workflow_visual.ipynb: visualization of models' result, and some of the visualization work are done in the above model training notebooks.
- Visuallization[Folder]: Images
- data[Folder]: 1. the original Kaggle data [News_Category_Dataset_v3.json](https://www.kaggle.com/datasets/rmisra/news-category-dataset); 2. others are sampled data with different sizes with or without dimension reduced (PCA);
- Exploring[Folder]: All files produced to explore the dataset, models, sentiment analysis, deminision reduction and so on in the duration of improving the Project;


### Code Reference
- Author:BANNOUR CHAKER, 2021, [NLP-tfidf-word2vec-Bert_confirmed_part1](https://www.kaggle.com/code/bannourchaker/nlp-tfidf-word2vec-bert-confirmed-part1#Bag-of-Words)
- Author: BAHAA AL-DEEN KATTAN, 2020, [Modern Models-Techniques with Real Project I](https://www.kaggle.com/code/xv7d111/modern-models-techniques-with-real-project-i)
- Author: MUHAMMAD AL ATIQI, 2019, [News Classification LSTM ~65% Accuracy](https://www.kaggle.com/code/arutaki/news-classification-lstm-65-accuracy)
- Author: MOHAMED AL-KAISI, 2020, [Topic Modeling and Sentiment Analysis](https://www.kaggle.com/code/temoralkaisi/topic-modeling-and-sentiment-analysis)
- Author: AVIKUMART · LINKED, 2022, [NLP-News_articles_classif (Wordembeddings&RNN)](https://www.kaggle.com/code/avikumart/nlp-news-articles-classif-wordembeddings-rnn)
- Author: DOMINIK KLEPL, 2019, [News Category Classification](https://www.kaggle.com/code/dklepl/news-category-classification/notebook)
  