# Using NLP on Reddit for Feature Importances of r/Bitcoin and r/Ethereum 

## Table of Contents

    1. ReadMe ... Introduction of Problem Statement and Executive Summary of Modeling and Results 
    2. Webscrape the Data ... Notebook that contains could for scraping Reddit using PushShift API and some data cleaning
    3. EDA and Modeling ... Data cleaning, feature engineering, exploratroy data analysis, visualiations, and preliminary modeling 
    4. Hyperparameter Tuning and Results ... Hyperparameter tuning, scores and classification metrics, and results 
    5. r_Bitcoin and r_Ethereum ... presentation slide deck 

## The Backdrop  

One of the fundamental obstacles of blockchain technology adoption is the learning curve one must traverse in order to fully understand and embrace this seemingly nebulous and complicated emergent technology. And as the Federal Reserve, along with central banks around the world, continue to print more money and devalue their currencies, Bitcoin and Ethereum continue to be the best performing assets under the sun, raising even more eyebrows about what these cryptocurrencies actually are.  

As someone who is passionate about moving forward with blockchain technology and the ecosystem of protocols that are thriving underneath the mysterious "cryptocurrency" visage, I'm oftentimes asked questions like, "what the heck is a Bitcoin?", "So, it's just money coming from thin air?", "I understand what the blockchain is, but how does that give value to a currency?", or "what can I buy with a bitcoin?"

At a glance, these questions are not too hard to answer, but admittedly some of these questions are simple reasons why this technology can be so disruptive: the space is still solving these problems and challenges us to redefine, or reimagine, basis fundamental mental models such as the concept of money and what it is. Eventually reconciling the answers to these questions will be presented in a way that is more easily digestible and acceptable for the majority of people. 

For example, PayPal recently announced that customer's can now buy, sell, and hold supported cryptocurrencies in their accounts ([*source*](https://www.coindesk.com/paypal-removes-waitlist-for-new-crypto-service)), and even more importantly the Office of the Comptroller of the Currency (OCC) just announced that banks can participate in facilitating payments in stablecoins, which are ERC-20 tokens that live on Ethereum ([*source*](https://www.occ.gov/news-issuances/news-releases/2021/nr-occ-2021-2.html)). Both of these extremely positive developments shed light on some basic questions and uncertainties that most people might have about cryptocurrencies. 

Part of the challenge of understanding the discourse is the rate at which news and developments progress. The space is moving so rapidly that, yesterday's conversations become irrelevant on an increasing basis. Even I--as someone who is deeply involved with the space--have challenges keeping up with the pace and demistifying the information and tech to other people. 

In my opinion, the best way to learn about Bitcoin and Ethereum is to do what I and most participants did: dive into the rabbit hole and dig.

But where to start? 

My goal is to create a model that classifies posts between r/Bitcoin and r/Ethereum and sheds light on key words that are indicative of each subreddit. The key words will be the most important features of each subreddit, and since I'll be looking at the most recent subreddit posts, they offer insight on current events. Perhaps these key words could serve as starting points for people who are curious and want to get dirty in the rabbit hole. 

## Summary 

I used PushShift API to scrape the most recent posts from both subreddits. The scraped data contained all sorts of information, but the only features of interest were text data. The bulk of text data were the 'title' and the 'selftext.' The 'title' is simply the title of the post on each subreddit. The 'selftext' was the body of text in the post of the subreddit. After removing deleted, removed, or null values from the data, I combined the 'title' and 'selftext' columns into a 'text' column under the assumption that while titles and posts might use different languages, for the purposes of our model, we can overlook those differences in language, such as parts of speech, cases, etc. From there I cleaned the data of all special characters and capital letters. I also got rid of outliers by engineering a 'word_count' and 'status_length' column. 

I set up two pipelines. Each pipeline used a Random Forest classifier, an Ada Boost classifier, and a Bagging classifier. One of the pipelines used a CountVectorizer transformer on each of the models and the other used a TfidVectorizer transformer on the models. I ran a grid search on each of the models and selected the count vectorized Random Forest classifier for further hypertuning because it produced the highest cross validation score.  

## DataDictionary

| Feature                  | Type    | Data   |  Description                                                  |
|--------------------------|---------|--------|---------------------------------------------------------------|
| text                     | object  | Reddit | body of text including the title and body of a post           |  
| sentiment                | float64 | Reddit | score of sentiment using SentimentIntensityAnalyzer from nltk |                             
| word_count               | int64   | Reddit | number of strings separated by a space in the text            |
| status_length            | int64   | Reddit | count of all characters in a text, including spaces           |

## Results 

| model                    | F1 Score | ROC AUC Score |
|--------------------------|----------|---------------|
| Random Forest Classifier | 82.2%    | 89.9%         |

After hypertuning the parameters I was able to predict the data with 82.2% accuracy without sacrificing too much of a bias-variance trade off and yielded an F1 score of 82.2%. Given the nature of my problem statement i.e. understanding important language used for each subreddit, I did not necessarily care to optimize for any specific score or classification metric. Rather I just needed to be reasonably sure that my model was good at correctly predicting most of the subreddit entries. For this reason I am only listing my F1 and ROC AUC scores, but other scores and metrics are available in the notebooks.

## Observations and Conclusions 

My goal is to essentially construct a quick and easy vocabulary list of important features within each subreddit. My intention is this list can serve as a starting point where people could begin their journey into learning about crypto for a couple of reasons. One, each of the features are going to be distinct features for each subreddit. And two, since I'm scraping the most recent posts, the features will also be relevant topics within each communities. 

After running through my first iteration of feature importances, I realized while features that were above average at predicting r/Ethereum were key words for the domain, features that were predictive of r/Bitcoin were ambigious in subject matter. In other words, while one could learn a bunch from the key features of r/Ethereum, it is not the case for r/Bitcoin. To mitigate this shortcoming, I included these ambiguous terms in my stops words list. I did this a couple of times. Each time the feature importances varied slightly and features were still great terms within r/Ethereum and I did not lose any significant value in my model's predictive power. However, the terms were still ambigious for r/Bitcoin. 

Here is a list of extrememly pertinent ngrams and bigrams for r/Ethereum based on feature importances of my random forest model. 

| ngrams     | bigrams               |   
|------------|-----------------------|
| token      | erc20 token           |    
| gas        | smart contract        |            
| ether      | beacon chain          |     
| dapp       | cold storage          |
| staking    | blockchain technology |

While this isn't the most robust list, these terms are all really useful words. (A more robust list and more visuals are available in my notebooks). 

It is also worthy to note that 'sentiment' was also an important feature for predicting r/Ethereum, which is an interesting conclusion. Admittedly, adding more words to my stopwords list decreased the predictive power of sentiment and illuminated other important features, but it is nonetheless an interesting sidenote. 

## Limitations and Further Considerations 

It is worthwhile to use a different modeling technique for this problem statement for a couple of reasons. One, while random forest feature importances provide insight on terms that are highly predictive of my positive target (r/Ethereum), the model was not picking up on terms that were important for r/Bitcoin. Rather it was picking up on terms that were not important to r/Ethereum. In addition, for a lot of cases the model did not assign any importance to some features which I thought would be useful for our purposes. And two, models such as a Logistic Regression might be more useful for interpreting coefficients. For these reasons, I would like to run a grid search on Logistic Regression model, and using the same methodology, I would like to compare any similarities and differences in important terms. 

Learning that sentiment is an important feature of r/Ethereum predictions, I am now curious to take a deeper dive into a sentiment analysis for both communities. 

