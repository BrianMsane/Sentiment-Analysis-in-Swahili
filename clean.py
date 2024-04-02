import pandas as pd
import nltk
from nltk.corpus import stopwords
from eganetswahiliclearner.cleaner import clean_text # dedicated Swahili cleaner

# Download NLTK stopwords
nltk.download('stopwords')

class MyCleaner:

    # typos, slang, and stop-words datasets come from a Swahili published paper.
    def __init__(self, typos_file, slang_file, stopwords_file, df):
        self.typos_file = typos_file
        self.slang_file = slang_file
        self.stopwords_file = stopwords_file
        self.df = df

    def read_files(self):
        typos_data = pd.read_csv(self.typos_file)
        slang_data = pd.read_csv(self.slang_file)
        stopwords_data = pd.read_csv(self.stopwords_file)
        return typos_data, slang_data, stopwords_data

    def remove_stopwords(self, text, stopwords):
        tokens = text.split()
        filtered_tokens = [word for word in tokens if word.lower() not in stopwords]
        return ' '.join(filtered_tokens)

    def resolve_slang(self, sentence, slang_correction_map):
        resolved_sentence = sentence
        for slang, meaning in slang_correction_map.items():
            resolved_sentence = resolved_sentence.replace(slang, meaning)
        return resolved_sentence

    def correct_typos(self, sentence, typo_correction_map):
        corrected_sentence = sentence
        for typo, correction in typo_correction_map.items():
            corrected_sentence = corrected_sentence.replace(typo, correction)
        return corrected_sentence

    def correct_typos_in_dataframe(self, df, typo_correction_map, typo_column):
        df_copy = df.copy()
        df_copy[typo_column] = df_copy[typo_column].apply(lambda x: self.correct_typos(x, typo_correction_map))
        return df_copy

    def resolve_slang_in_dataframe(self, df, slang_correction_map, slang_column):
        df_copy = df.copy()
        df_copy[slang_column] = df_copy[slang_column].apply(lambda x: self.resolve_slang(x, slang_correction_map))
        return df_copy

    def final_clean(self):
        typos_data, slang_data, stopwords_data = self.read_files()

        # Combine English and Swahili stopwords
        english_stopwords = set(stopwords.words('english'))
        swahili_stopwords = set(stopwords_data['StopWords'])
        all_stopwords = english_stopwords.union(swahili_stopwords)

        df['Tweets'] = df['Tweets'].apply(lambda x: self.remove_stopwords(x, all_stopwords))

        slang_correction_map = dict(zip(slang_data['Slang'], slang_data['Meaning']))
        df = self.resolve_slang_in_dataframe(df, slang_correction_map, "Tweets")

        typo_correction_map = dict(zip(typos_data['Typo'], typos_data['Word']))
        df = self.correct_typos_in_dataframe(df, typo_correction_map, "Tweets")

        return df


def cleaner(df):
    cleaner = MyCleaner("Common Swahili Typos.csv", "Common Swahili Slang.csv", "Common Swahili Stop-words.csv", df)
    df['Tweets'] = df['Tweets'].apply(clean_text)
    df = cleaner.final_clean()
    return df
