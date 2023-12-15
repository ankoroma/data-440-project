# Lifespan of News Stories: A NLP Approach to Extracting Trending News

## Project Idea
Nowadays, millions of news articles and blogs are published online every day. News data is created at the rate one cannot imagine a few years ago. Social media platforms have become the main source of news online to meet the information consumption needs of internet users. However, a large amount of news with repeated, duplicated or junk contents is also created. Therefore, there is an increasing need for automatic grouping of the news based on the contents of articles. This project will explore various NLP techniques for clustering online news and extracting the trending stories over time.

## Research Questions
Define a news story: the <b>main topic</b> of the article, so focus on the headlines.
1. What is the duration of the shelf-life of news stories?
2. Among news stories covering political, environmental, social issues, violence and war, which ones tend to have a longer lasting public view and which ones tend to fade away quickly?

## Data Sources
<b>3 News Agencies (from [Kaggle](https://www.kaggle.com/notlucasp/financial-news-headlines)):</b>
- Reuters (used in current iteration)
- The Guardian
- CNBC

<b>Data Structure of Reuters Dataset:</b>
- About 32k news articles from 2018 - 2020
- Each article contains the headline, a short description, and publishing time


## Data Pipeline
| Preprocessing | Keyword Extraction | News Clustering | Visualizing Trending Stories
| ---- | --------- | --------- | ------------
| Remove Unwanted Text | NER | Keyword Vectorization | Cluster Time Series News
| Date Normalization | Noun Phrases | News Similarity | Visualize Top Trending Stories
| Lemmatization | Keyword Scoring | DBSCAN Algo. | 
|  | Keyword Filtering |  | 
|  | Postprocessing |  | 


## Preprocessing
- Removed unwanted texts
  - Non-english characters, unusual headline patterns, etc.
```python
 non_en_chars = {
        "’": "'",
        "‘": "'"
    }
    remove_from_title = ["BREAKING:", "[\-:] report", "[\-:] sources",  "[\-:] source", "source says", "Exclusive\:",
                         "Factbox\:", "Timeline[\-:]",  "Instant View\:", "Explainer\:", ": Bloomberg",
                         ": WSJ"]
    remove_if_start_with = ['close breaking news']
    replace_if_contain = ['click here to see']
```

- Date Normalization
  - M/D/Y: “Jul 18 2020”
- Lemmatization
  - Kept stopwords for noun phrase detection (“The U.K”)


## Keyword Extraction (main task)
<b>Spacy: Name Entity Recognition (NER)</b>
- Extract named entities with term frequency to reflect key points of news story (i.e., POS: PERSON, ORG, GPE, Noun phrases, etc.)
```python
allow_types = ['PERSON', 'GPE', 'ORG', 'NORP', 'LOC', 'FAC', 'WORK_OF_ART', 'EVENT', 'LAW', 'PRODUCT']
```

<b>Keyword Scoring Metric</b>
- Different weights are used depending on keyword type (entity or noun phrase)
- Keywords found in headlines weighs more than those in the content (entity = 4, noun chunks = 2, other = 1)
```python
def extract_keywords(cls, title, content, title_entity_weight=4, title_noun_chunk_weight=2):
```

<b>Keyword Filtering</b>
- Remove stopwords, special characters ,  news agency in headline/content, etc. 
```python
remove_entities = ['REUTERS', 'Reuters', 'Thomson Reuters', 'CNBC', 'reuters story']
```

<b>Postprocessing</b>
- Abbreviate long entities for visualization
```python
keywords_linking_table = {
        "United States": "US",
        "United Nations": "UN",
        "European Union": "EU",
        "United Kingdom": "UK",
        "European Central Bank": "ECB",...}
```

## News Clustering: Unsupervised
<b>Vectorization</b>
- Convert the keywords into numerical representation using sklearn’s CountVectorizer or HashingVectorizer depending on number of articles to be clustered
- Score of extracted keyword is the term frequency in vectorizer
```python
def get_vectors(str_list, hash_threshold=500):
    text = [t.replace("&", "_") for t in str_list]
    if len(text) > hash_threshold:
        vectorizer = HashingVectorizer(n_features=10000, stop_words=None)
        m = vectorizer.fit_transform(text)      
    else:
        vectorizer = CountVectorizer(stop_words=None)
        m = vectorizer.fit_transform(text)            
 ```

<b>Cosine Similarity</b>
- Measure similarity between the keywords from two news articles

<b>DBSCAN</b>
- Cluster news stories based on similarity score
- No predetermined number of stories in a cluster


## Visualizations: Timeline & Aninmated Time Series Plot

## Findings: Addressing [Research Questions](README.md#research-questions)
<b>Q1: Analyzing trending news story duration</b>
- Trade war among various countries 
  - US-China trade talks lasted for 8 months
- Boeing 737 MAX 8 aircraft crash (also lasted for 8 months)
- Covid-19 gained traction in February 2020
  - Rapid growth since then

<b>Q2: Topics with long lasting public view</b>
- Political news stories (especially on the US economy) last longer


## Challenges & Future Work
- Memory & runtime issues due to large datasets (~1hr runtime)
- Interactivity: zooming, filtering, alternative visual representations (scope of course)
- Scalability: Ensure keyword extraction and clustering algorithms scales with growing dataset
  - Integrate database systems (MongoDB, PostgreSQL)
  - Explore other news agencies or look at more recent data
  - Check for disparity in recent trending stories
- Web Application
  - Search Engine: users can search and view different timelines
  - Custom Visualization: 


## References
- Tom Nicholls & Jonathan Bright (2019) Understanding News Story Chains using Information Retrieval and Network Clustering Techniques, Communication Methods and Measures, 13:1, 43–59, DOI: 10.1080/19312458.2018.1536972
- [Towards Data Science](https://towardsdatascience.com/extract-trending-stories-in-news-29f0e7d3316a
)
- [StoryGraph](http://storygraph.cs.odu.edu) by Alexander C. Nwala
- [Kaggle Dataset](https://www.kaggle.com/notlucasp/financial-news-headlines)
- Coding Style: [PEP8](https://peps.python.org/pep-0008/)

## Run Program on Mac & Linux OS
- You might need to modify these steps accordingly for Windows
### 1. Clone or download the repository folder.
### 2. Open the Terminal and navigate to the project folder then create a virtual environment
```python
python3 -m venv venv
```
### 3. Activate the virtual environment
```python
source venv/bin/activate
```
### 4. Install Dependencies
```python
pip install -r requirements.txt
```
### 5. Download the 'en_core_web_lg' pipeline from spacy (will not work without it)
```python
python3 -m spacy download en_core_web_lg
```
### 6. Run main.py file
```python
python3 main.py
```

### 7. Optional: view the animated time series for trending news
```python
streamlit run site.py
```

### 8. Model Output
- After running main.py, a window should pop up with the visualization: Switch to full screen for a better view
- The 'viz' folder holds the .png file from the visualization and the .gif file for the animated time series plot.
- If you chose to run site.py, a streamlit site should pop up in your default browser.

---
---
---
---
---
