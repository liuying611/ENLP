from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('wordnet')
nltk.download('punkt')
import re
import string



class PreProcessing:
    def __init__(self, fpath, fileformat='json') -> None:
        '''
        Read Raw Data
        fpath: file path
        '''
        self.fpath = fpath
        self.fileformat = fileformat
        self.df = pd.DataFrame()
        self.final_columns = ['text', 'category']
        self.categories = []
    def get_data(self):
        '''
        combine columns and drop useless columns
        '''
        print(f"Start reading data from {self.fpath}...")
        if self.fileformat == 'json':
            self.df = pd.read_json(self.fpath, lines=True)
        elif self.fileformat == 'csv':
            self.df = pd.read_csv(self.fpath)
        else:
            print('Format not supported')
        
        # combine headline and short_description
        self.df['text'] = self.df['headline'] + ' ' + self.df['short_description']
        # drop columns that are not in final_columns
        columns_to_drop = [col for col in self.df.columns if col not in self.final_columns]
        self.df = self.df.drop(columns_to_drop, axis=1)
        return self.df
    
    def reorganize_category(self, replace_dict = None):
        '''
        Reorganize topics
        replace_dict: dictionary of topics to replace
        '''
        print(f'Reorganizing categories...')
        if replace_dict is None:
            replace_dict = {
                "HEALTHY LIVING": "WELLNESS",
                "QUEER VOICES": "GROUPS VOICES",
                "COMEDY": "ENTERTAINMENT",
                "BUSINESS": "BUSINESS & FINANCES",
                "PARENTS": "PARENTING",
                "BLACK VOICES": "GROUPS VOICES",
                "THE WORLDPOST": "WORLD NEWS",
                "STYLE": "STYLE & BEAUTY",
                "GREEN": "ENVIRONMENT",
                "TASTE": "FOOD & DRINK",
                "WORLDPOST": "WORLD NEWS",
                "SCIENCE": "SCIENCE & TECH",
                "TECH": "SCIENCE & TECH",
                "MONEY": "BUSINESS & FINANCES",
                "ARTS": "ARTS & CULTURE",
                "COLLEGE": "EDUCATION",
                "LATINO VOICES": "GROUPS VOICES",
                "CULTURE & ARTS": "ARTS & CULTURE",
                "FIFTY": "MISCELLANEOUS",
                "GOOD NEWS": "MISCELLANEOUS"}
        self.df.category = self.df.category.replace(replace_dict)
        self.categories = self.df['category'].unique()
        return self.df
    
    def cleaning(self):
        
        def strip_emoji(text):
            emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002500-\U00002BEF"  # chinese characters
                u"\U00002702-\U000027B0"
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u"\U00010000-\U0010ffff"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u200d"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\ufe0f"  # dingbats
                u"\u3030"
                                "]+", flags=re.UNICODE)
            return emoji_pattern.sub(r'', text)

        #Remove punctuations, links, mentions and \r\n new line characters
        def strip_all_entities(text):
            text = text.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower() #remove \n and \r and lowercase
            text = re.sub(r"(?:\@|https?\://)\S+", "", text) #remove links and mentions
            text = re.sub(r'[^\x00-\x7f]',r'', text) #remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
            banned_list= string.punctuation + 'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'
            table = str.maketrans('', '', banned_list)
            text = text.translate(table)
            return text

        #clean hashtags at the end of the sentence, and keep those in the middle of the sentence by removing just the # symbol
        def clean_hashtags(text):
            new_text = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', text)) #remove last hashtags
            new_text2 = " ".join(word.strip() for word in re.split('#|_', new_text)) #remove hashtags symbol from words in the middle of the sentence
            return new_text2

        #Filter special characters such as & and $ present in some words
        def filter_chars(a):
            sent = []
            for word in a.split(' '):
                if ('$' in word) | ('&' in word):
                    sent.append('')
                else:
                    sent.append(word)
            return ' '.join(sent)

        def remove_mult_spaces(text): # remove multiple spaces
            return re.sub("\s\s+" , " ", text)

        # remove stopwords
        def remove_stopwords(sentence):
            """
            Removes a list of stopwords

            Args:
                sentence (string): sentence to remove the stopwords from

            Returns:
                sentence (string): lowercase sentence without the stopwords
            """
            # List of stopwords
            stop = stopwords.words("english")
            stop.extend(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 
                        'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ha', 'wa'])
            # Sentence converted to lowercase-only
            sentence = sentence.lower()

            words = sentence.split(' ')
            no_words = [w for w in words if w not in stop]
            sentence = " ".join(no_words)

            return sentence


        def del_less_than_10_words(text):
            text = text.strip()
            if len(text.split(' ')) < 10: #remove sentences with less than 3 words
                return None
            else:
                return text

        def lemmatize_text(text):
            # Tokenize the sentence
            word_list = word_tokenize(text)
            lem = WordNetLemmatizer()
            global MAX_LEN
            # Lemmatize list of words and join
            lemmatized = [lem.lemmatize(word) for word in word_list]
            lemmatized_output = [token for token in lemmatized if token!='']
            if len(lemmatized_output) == 0:
                return 'NA'
            elif len(lemmatized_output)>MAX_LEN:
                MAX_LEN = len(lemmatized_output)
            return " ".join(lemmatized_output)
        
        print(f'Srarting cleaning data...')
        self.df['text'] = (self.df['text'].apply(strip_emoji)
                .apply(strip_all_entities)
                .apply(clean_hashtags)
                .apply(filter_chars)
                .apply(remove_mult_spaces)
                .apply(lemmatize_text)
                .apply(remove_stopwords)
                .apply(del_less_than_10_words))
        print(f'number of null values: {self.df.isnull().sum()}')
        self.df = self.df.dropna()  
        print(f'number of duplicates: {self.df.duplicated().sum()}')
        self.df = self.df.drop_duplicates()
        print(f'Cleaning data finished')
        print(f'number of rows after cleaning: {self.df.shape[0]}')
        
        return self.df
        
    def sentiment_analysis(self):
        '''
        Sentiment Analysis
        '''
        print(f'Starting sentiment analysis...')
        sid = SentimentIntensityAnalyzer()
        self.df['sa'] = self.df['text'].apply(lambda x: sid.polarity_scores(x)['compound'])
        return self.df
    
    def sample_data(self, output_size = None, top_k_category = None, balance = True):
        '''
        Sample data
        output_size: number of rows to sample
        top_k_category: number of categories to keep
        balance: whether to balance the dataset, if True, will reduce the top 3 categories to around 10000 rows
        '''
        self.top_k_category = top_k_category
        self.balance = balance
        self.output_size = output_size
        
        categories_to_keep = []
        if self.top_k_category is None:
            categories_to_keep = self.df['category'].value_counts().index.values
        else:
            categories_to_keep = self.df['category'].value_counts()[:self.top_k_category].index.values
        df_k = self.df[self.df['category'].isin(categories_to_keep)]
        if self.balance:
            # Sample 20000 rows from each category
            plt_to_remove = df_k[df_k['category'] == 'POLITICS'].sample(20000, random_state=1).index
            wln_to_remove = df_k[df_k['category'] == 'WELLNESS'].sample(10000, random_state=2).index
            ent_to_remove = df_k[df_k['category'] == 'ENTERTAINMENT'].sample(10000, random_state=3).index
            
            # Drop these rows
            df_k = df_k.drop(plt_to_remove)
            df_k = df_k.drop(wln_to_remove)
            df_k = df_k.drop(ent_to_remove)
            
        # Shuffle the DataFrame rows
        if self.output_size is not None:
            df_k = df_k.sample(self.output_size, random_state=4)
        else:
            df_k = df_k.sample(frac=1, random_state=4)
        return df_k
            
class DataAnlysisPCA:
    def __init__(self, df) -> None:
        # make sure the columns are text, category, sa
        self.df = pd.DataFrame()
        self.MAX_LEN = 0
        self.columns = ['text', 'category', 'sa']
        self.best_pca_component = 0
        if not all([col in df.columns for col in self.columns]):
            print('Columns not correct')
            return None
        else:
            self.df = df
        self.categories = self.df['category'].unique()
        
    def get_max_word_length(self, percentile = 99, plot = False):
        length_list = []
        for index, row in self.df.iterrows():
            length_list.append(len(row['text'].split(' ')))
        self.MAX_LEN = np.percentile(length_list, percentile)
        if plot:
            self.plot_word_length_distribution(length_list, percentile)
        
        return self.MAX_LEN, length_list
    
    def plot_word_length_distribution(self, length_list, percentile):
        '''
        Plot the distribution of the word length
        max_len: the max length of the word
        length_list: list of word length
        percentile: percentile of the word length
        '''
        # Plotting the histogram of the words length
        plt.figure(figsize=(6, 3))
        plt.hist(length_list, bins=50, edgecolor='black')
        plt.xlabel('Text Length')
        plt.ylabel('Frequency')
        plt.title('Distribution of Text Lengths')
        plt.axvline(x=self.MAX_LEN, color='red', linestyle='--', label=f'{percentile}th Quartile')
        plt.legend()
        plt.grid(True)
        plt.show()
        print(f"{percentile} Percentile of Text Lengths: {self.MAX_LEN}")
        
    def get_most_common_words(self, top_n_words = 50, plot = False):
        '''
        Get the most common words
        top_n_words: number of words to return
        plot: whether to plot the bar chart
        '''
        # Get the most common words
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(self.df['text'])
        word_list = vectorizer.get_feature_names()
        count_list = X.toarray().sum(axis=0)
        word_count_df = pd.DataFrame({'word': word_list, 'count': count_list})
        word_count_df = word_count_df.sort_values('count', ascending=False)
        word_count_df = word_count_df[:top_n_words]
        if plot:
            self.plot_most_common_words(word_count_df)
        df_most_common = self.build_top_words_count_matrix(top_n_words, word_count_df, plot)
        df_pca = self.dimension_reduction(df_most_common)
        return word_count_df, df_most_common, df_pca
    
    def plot_most_common_words(self, word_count_df):
        '''
        Plot the most common words
        word_count_df: dataframe of the most common words
        '''
        # Plot the most common words
        plt.figure(figsize=(6, 3))
        plt.bar(word_count_df['word'], word_count_df['count'])
        plt.xticks(rotation=90)
        plt.xlabel('Word')
        plt.ylabel('Count')
        plt.grid(True)
        plt.show()
        
    def build_top_words_count_matrix(self, top_n_words, word_count_df, plot = False):
        '''
        Vectorize the top k words
        top_k_words: number of words to vectorize
        '''
        # Vectorize the top k words
        print(f'\n Starting building CountMatrix of top {top_n_words} common words...')
        top_words_to_keep = word_count_df['word'].values[:top_n_words]
        def keep_word_in_list(text, top_words_to_keep):
            text = text.split()
            for t in text:
                if t in top_words_to_keep:
                    continue
                else:
                    text.remove(t)
            return ' '.join(text)
        df_top_words = self.df['text'].apply(lambda x: keep_word_in_list(x, top_words_to_keep))
        df_most_common = pd.DataFrame()
        for word in top_words_to_keep:
            word_count = []
            for row in df_top_words:
                if word in row.split():
                    word_count.append(1)
                else:
                    word_count.append(0)
            df_most_common[word] = word_count
        self.best_pca_component = self.get_best_PCA_component(df_most_common, plot)
        
        return df_most_common
    
    def get_best_PCA_component(self, df_most_common, plot):
        print(f'\n Starting PCA to find best component...')
        pca = PCA().fit(df_most_common)
        cumsum = pd.DataFrame(np.cumsum(pca.explained_variance_ratio_))
        best_component = cumsum[(cumsum >= .90) & (cumsum  < .95)].dropna().index[0] 
        print(f'\n total most common words: {len(df_most_common.columns)}')
        print('Best Component that keep 90% info:', best_component)
        if plot:
            plt.figure(figsize=(6, 3))
            plt.plot(np.cumsum(pca.explained_variance_ratio_))
            plt.axhline(y=.90, color = 'r', linestyle = '--', label = '0.90')
            plt.xlabel('number of components')
            plt.ylabel('cumulative explained variance')
            plt.legend()
            plt.show()
        return best_component
    
    def dimension_reduction(self, df_most_common):
        '''
        Dimension Reduction
        '''
        print(f'\n PCA: Starting reducing dimension...')
        pca = PCA(n_components=self.best_pca_component)
        reduced_features = pd.DataFrame(pca.fit_transform(df_most_common))
        df_pca = pd.concat([reduced_features, self.df], axis=1)
        return df_pca