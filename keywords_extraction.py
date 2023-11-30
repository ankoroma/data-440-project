import re
import numpy as np
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lemmatizer import Lemmatizer
from spacy.lookups import Lookups
from cosine_similarity import calculate_cosine_similarity
import calendar as cd


class KeywordsExtract:
    MODEL = spacy.load("en_core_web_lg")
    allow_types = ['PERSON', 'GPE', 'ORG', 'NORP', 'LOC', 'FAC', 'WORK_OF_ART', 'EVENT', 'LAW', 'PRODUCT']
    remove_words = ['new', 'time', 'matter', 'source', 'people', 'story', 'reuters story']
    remove_entities = ['REUTERS', 'Reuters', 'Thomson Reuters', 'CNBC']
    months = [cd.month_name[i] for i in range(1, 13)] + [cd.month_abbr[i] for i in range(1, 13)]
    lookups = Lookups()
    lemma_keep = ["data"]
    lemma_exc = MODEL.vocab.lookups.get_table("lemma_exc")
    for w in lemma_keep:
        del lemma_exc[MODEL.vocab.strings["noun"]][w]
    lookups.add_table("lemma_exc", lemma_exc)
    lookups.add_table("lemma_rules", MODEL.vocab.lookups.get_table("lemma_rules"))
    lookups.add_table("lemma_index", MODEL.vocab.lookups.get_table("lemma_index"))
    lemmatizer = Lemmatizer(lookups)

    @classmethod
    def extract_keywords(cls, title, content, title_entity_weight=4, title_noun_chunk_weight=2):
        """
        Extract keywords from the given title and content, with more weight assigned to entities and noun chunks in the title.

        Args:
            title (str): The title of the article.
            content (str): The content or description of the article.
            title_entity_weight (int, optional): The weight assigned to entities in the title. Defaults to 4.
            title_noun_chunk_weight (int, optional): The weight assigned to noun chunks in the title. Defaults to 2.

        Returns:
            df_ent_grp: The dataframe containing the entity groups.
            df_chunk_grp: The dataframe containing the noun chunk groups.
            keywords: The list of extracted keywords along with their weights.
        """
        title_txt = title + ". "
        full_content = title_txt + content
        ner_doc = cls.MODEL(full_content)
        df_ent_grp, _ = cls.extract_entities(ner_doc, title=title_txt, title_weight=title_entity_weight)
        df_chunk_grp, _ = cls.extract_noun_chunks(ner_doc, title=title_txt, title_weight=title_noun_chunk_weight)
        df_keywords = cls.filter_keywords_by_score(df_ent_grp, score_threshold=0.05, max_ents=10)
        df_chunks = cls.filter_keywords_by_score(df_chunk_grp, score_threshold=0.05, max_ents=10)
        if not df_chunks.empty:
            df_keywords = df_keywords.append(df_chunks, ignore_index=True, sort=False)
            df_keywords.sort_values(by=['total_score', 'start'], inplace=True, ascending=[False, True])
            df_keywords.drop_duplicates(subset=['entity'], inplace=True)
            df_keywords.drop_duplicates(subset=['lemma'], inplace=True)
        keywords = [{'keyword': str(r['entity']), 'weight': int(r['total_score'])} for _, r in df_keywords.iterrows()]
        return df_ent_grp, df_chunk_grp, keywords

    @classmethod
    def extract_entities(cls, ner_document, title="", title_weight=4):
        """
        Extracts entities from a given text using named entity recognition (NER).

        Args:
            cls (KeywordsExtract): The instance of the KeywordsExtract class.
            ner_document (spacy.tokens.doc.Doc): The document object obtained from applying NER to the input text.
            title (str, optional): The title of the article. Defaults to "".
            title_weight (int, optional): The weight assigned to entities in the title. Defaults to 4.

        Returns:
            (df_grp, df): A tuple containing two DataFrames. The first DataFrame contains the merged entity groups, and the second DataFrame contains the individual entities.

        Example:
            # Initialize the KeywordsExtract class
            ke = KeywordsExtract()

            # Define the input text
            text = "Apple Inc. is planning to open a new store in New York City."

            # Extract entities from the text
            df_grp, df = ke.extract_entities(ke.MODEL(text), title="")

            # Print the extracted entities
            print(df)

        Expected output:
            entity          start  end  label  lemma         ent_type  weight
        0   Apple Inc.      0      10   ORG    Apple Inc.    entity    1
        1   New York City   34     47   GPE    New York City entity    1
        """

        ents = [ent for ent in ner_document.ents if ent.label_ in cls.allow_types]
        ents = [cls.trim_tags(ent, for_type='tag') for ent in ents]
        ents = [cls.remove_email_and_punctuation(ent) for ent in ents]
        ents = [ent for ent in ents if len(ent) > 0]
        ent_list = [[cls.lemma_last_word(ent), ent.start_char, ent.end_char, ent.label_, ent.lemma_] for ent in ents]
        cols = ['entity', 'start', 'end', 'label', 'lemma']
        df = pd.DataFrame(ent_list, columns=cols)
        df = cls.filter_keywords(df)
        df['ent_type'] = 'entity'
        df['weight'] = 1
        if not df.empty:
            df['entity'] = df['entity'].str.strip()
            df['weight'] = df['start'].apply(lambda x: title_weight if (x < len(title)) else 1)
        df_grp, df = cls.filter_keywords_and_calculate_weight(df)
        if not df_grp.empty:
            df_grp = cls.merge_keywords_by_similarity(df_grp)
        return df_grp, df

    @classmethod
    def extract_noun_chunks(cls, ner_document, title="", title_weight=2):
        """
        Extracts noun chunks from a given text using spaCy's named entity recognition (NER).

        Args:
            cls (class): The class object.
            ner_document (spacy.tokens.doc.Doc): The document object obtained from applying NER to the input text.
            title (str, optional): The title of the article. Defaults to an empty string.
            title_weight (int, optional): The weight assigned to noun chunks in the title. Defaults to 2.

        Returns:
            df_grp: The dataframe containing the merged noun chunk groups.
            df: The dataframe containing the individual noun chunks.
        """
        chunks = [ch for ch in list(ner_document.noun_chunks) if (ch.root.ent_type_ == '')]
        chunks = [ch for ch in chunks if len(ch) > 0]
        chunks = [cls.trim_tags(ch) for ch in chunks]
        chunks = [cls.trim_stop_words(ch) for ch in chunks]
        chunks = [cls.remove_email_and_punctuation(ch) for ch in chunks]
        chunks = [cls.trim_entities(ch) for ch in chunks]
        chunks = [ch for ch in chunks if len(ch) > 0]
        chunks_list = [[cls.lemma_last_word(ch), ch.start_char, ch.end_char, ch.label_, ch.lemma_] for ch in chunks]
        cols = ['entity', 'start', 'end', 'label', 'lemma']
        df = pd.DataFrame(chunks_list, columns=cols)
        df = cls.filter_keywords(df)
        df['ent_type'] = 'noun_chunk'
        df['weight'] = 1
        if not df.empty:
            df['entity'] = df['entity'].str.strip()
            df = df[(df['entity'].str.len() > 3) | df['entity'].str.isupper()]
            df['weight'] = df['start'].apply(lambda x: title_weight if (x < len(title)) else 1)
        df_grp, df = cls.filter_keywords_and_calculate_weight(df)
        if not df_grp.empty:
            df_grp = cls.merge_keywords_by_similarity(df_grp)
        return df_grp, df

    @classmethod
    def filter_keywords_and_calculate_weight(cls, df):
        """
        Filters and calculates the weight of keywords in a DataFrame.

        Args:
            cls (KeywordsExtract): An instance of the KeywordsExtract class.
            df (DataFrame): A DataFrame containing the keywords and their weights.

        Returns:
            df_merge: A DataFrame containing the unique entities and their total weights, sorted by weight and start position.
            df: The original DataFrame with the filtered keywords.
        """
        _months = cls.months
        if not df.empty:
            # remove unwanted entities
            df = df[~df['entity'].isin(cls.remove_entities)]
            # remove misclassified dates
            df = df[df['entity'].apply(lambda x: re.search("\s+\d+|".join(_months) + "\s+\d+", str(x)) is None)]
            if df.empty:
                return pd.DataFrame(), df
            df1 = df.groupby(['entity'])[['weight']].sum()
            df0 = df.drop_duplicates(subset=['entity']).copy()
            if "weight" in df0.columns:
                df0.drop(['weight'], axis=1, inplace=True)
            df_merge = pd.merge(df0, df1, left_on='entity', right_index=True)
        else:
            return pd.DataFrame(), df
        df_merge.sort_values(by=['weight', 'start'], inplace=True, ascending=[False, True])
        df_merge = df_merge.reset_index(drop=True)
        return df_merge, df

    @classmethod
    def filter_keywords(cls, df):
        """
        Filter and clean the extracted keywords in a DataFrame.

        Args:
            df (DataFrame): A DataFrame containing the extracted keywords.

        Returns:
            df: The filtered DataFrame containing the cleaned keywords.
        """
        if not df.empty:
            # remove special characters
            df['entity'] = df['entity'].apply(lambda x: re.sub('\.|[\-\'\$\/\\\*\+\|\^\#\@\~\`]{2,}', '', str(x)))
            df = df[df['entity'].apply(lambda x: re.search('\(|\)|\[|\]|\"|\:|\{|\}|\^|\*|\;|\~|\|', str(x)) is None)]
            # remove unwanted words
            if not df.empty:
                df = df[df['entity'].apply(lambda x: x.lower() not in cls.remove_words)]
            # remove too long entities
            if not df.empty:
                df = df[df['entity'].apply(lambda x: len(str(x)) < 35)]
            # remove too short entities
            if not df.empty:
                df = df[df['entity'].apply(lambda x: len(str(x)) > 1)]
        return df

    @staticmethod
    def trim_tags(s, for_type='chunk', trim_tags=['PDT', 'DT', 'IN', 'CC'], punctuation=[',', '\'']):
        """
        Trim specified tags and punctuation marks from a given text.

        Args:
            s (str): The input text to be trimmed.
            for_type (str, optional): The type of text to be trimmed. It can be either 'chunk' or 'tag'. Defaults to 'chunk'.
            trim_tags (list, optional): A list of tags to be removed from the text. Defaults to ['PDT', 'DT', 'IN', 'CC'].
            punctuation (list, optional): A list of punctuation marks to be removed from the text. Defaults to [',', '\''].

        Returns:
            s1: The trimmed text.
        """
        if len(s) < 1:
            return s
        s1 = s
        if (for_type == 'chunk') and (re.search('|'.join(punctuation), s1.text) is not None):
            for i in range(len(s1) - 1, -1, -1):
                if s1[i].text in punctuation:
                    s1 = s1[i + 1:]
                    break
        if len(s1) > 0:
            s1 = s1[1:] if (s1[0].tag_ in trim_tags) else s1
        if len(s1) > 1:
            s1 = s1[:-1] if (s1[-1].tag_ in trim_tags) else s1
        return s1

    @staticmethod
    def lemma_last_word(s):
        """
        Extracts the lemma (base form) of the last word in a given text.

        Args:
            s (spacy.tokens.span.Span): The input text as a spaCy span object.

        Returns:
            txt: The lemma of the last word in the input text, with the rest of the text if applicable.
        """
        if s[-1].tag_ in ['NNS', 'NNPS']:
            lemma = KeywordsExtract.lemmatizer(s[-1].text, 'NOUN')[0]
            txt = lemma.title() if s[-1].text.istitle() else lemma
            if len(s) > 1:
                txt = s[:-1].text + " " + txt
        else:
            txt = s.text
        return txt

    @staticmethod
    def filter_keywords_by_score(df, score_threshold=0.1, max_ents=50, top_n=5):
        """
        Filters the keywords in a DataFrame based on their scores and returns the top N keywords that meet the score threshold.

        Args:
            df (DataFrame): A DataFrame containing the keywords and their scores.
            score_threshold (float, optional): The threshold for the score. Keywords with scores above this threshold will be selected. Defaults to 0.1.
            max_ents (int, optional): The maximum number of keywords to select. Defaults to 50.
            top_n (int, optional): The number of top keywords to return. Defaults to 5.

        Returns:
            ent: The filtered DataFrame containing the top N keywords that meet the score threshold.
        """
        if df.empty:
            return pd.DataFrame()
        elif len(df) > max_ents:
            df = df[:max_ents]
        cond1 = df['total_score'] / df['total_score'].sum() > score_threshold
        cond2 = df['total_score'] > 0
        ent = df[cond1 & cond2].reset_index(drop=True)
        ent = ent.iloc[:top_n]
        return ent

    @staticmethod
    def trim_stop_words(s):
        """
        Removes stop words from a given text.

        Args:
            s (spacy.tokens.doc.Doc): The input text as a spaCy document object.

        Returns:
            s1: The trimmed text with stop words removed.
        """

        if len(s) < 1:
            return s
        if len(s) == 1:
            s1 = [] if (s[0].text.lower() in STOP_WORDS) else s
        else:
            s1 = s[1:] if (s[0].text.lower() in STOP_WORDS) else s
            if len(s1) == 1:
                s1 = [] if (s[-1].text.lower() in STOP_WORDS) else s
            else:
                s1 = s1[:-1] if (s1[-1].text.lower() in STOP_WORDS) else s1
        return s1

    @staticmethod
    def trim_entities(s):
        """
        Trim unnecessary words from a spaCy `Span` object that represents an entity.
    
        Args:
            s (spaCy Span): The entity to be trimmed.
        
        Returns:
            s1: The trimmed entity.
        
        Summary:
        The `trim_entities` method is used to trim unnecessary words from a spaCy `Span` object that represents an entity. 
        It removes words from the beginning of the entity until it reaches a word that is not a noun, proper noun, adjective, or punctuation.
    
        Example Usage:
        ```python
        ke = KeywordsExtract()
        text = "Apple Inc. is planning to open a new store in New York City."
        doc = ke.MODEL(text)
        entities = [ent for ent in doc.ents if ent.label_ in ke.allow_types]
        entity = entities[0]  # Assume there is at least one entity
        trimmed_entity = ke.trim_entities(entity)
        print(trimmed_entity.text)
        ```
        Expected output:
        ```
        Inc. is planning to open a new store in New York City
        ```
        """
        if len(s) < 2:
            return s
        if s[-1].text == s.root.text:
            n = len(s) - 1
        else:
            n = len(s) - 2
        for i in range(n - 1, -1, -1):
            if not s[i].pos_ in ['NOUN', 'PROPN', 'ADJ', 'PUNCT']:
                s1 = s[i + 1:n + 1]
                break
        else:
            s1 = s[:n + 1]
        return s1

    @staticmethod
    def remove_email_and_punctuation(s):
        """
        Removes email addresses and certain punctuation marks from a given text.

        Args:
        - s (spacy.tokens.doc.Doc): The input text as a spaCy document object.

        Returns:
        - s1: The modified text with email addresses and certain punctuation marks removed.
        """

        if len(s) < 1:
            return s
        s1 = s if not s.root.like_email else []
        if len(s1) > 0:
            s1 = s1[:-1] if (s1[-1].tag_ in ['POS']) else s1
            s1 = s1[1:] if (s1[0].tag_ in ['POS']) else s1
        if len(s1) > 0:
            s1 = s1[1:] if (s1[0].pos_ in ['PUNCT']) else s1
        return s1

    @staticmethod
    def merge_keywords_by_similarity(df, th=0.4):
        """
        Merge similar keywords in a DataFrame based on their cosine similarity scores.

        Args:
            df (DataFrame): A DataFrame containing the keywords and their weights.
            th (float, optional): The threshold value for cosine similarity. Defaults to 0.4.

        Returns:
            df: The merged DataFrame containing the unique merged keywords and their total scores.
        """
        if len(df) > 1:
            m = calculate_cosine_similarity(df['entity'].tolist())
            most_sim = [[j for j in range(len(m)) if m[i, j] >= th] for i in range(len(m)) if i < 20]
            m[np.tril_indices(len(m), -1)] = 0
            m[m < th] = 0
            m[m >= th] = 1
            df['total_score'] = np.matmul(m, df['weight'].values.reshape(-1, 1))
            for idx, x in enumerate(most_sim):
                if len(x) <= 1:
                    continue
                a = df.loc[x, 'entity']
                df.loc[idx, 'entity'] = a.values[0]
            df.loc[df['total_score'] == 0.0, 'total_score'] = df.loc[df['total_score'] == 0.0, 'weight'] * 1.0
            df = df.drop_duplicates(subset=['entity']).reset_index(drop=True)
            df.sort_values(by=['total_score', 'start'], inplace=True, ascending=[False, True])
        else:
            df['total_score'] = df['weight'] if 'weight' in df.columns else 1
        return df

