# PokéWhom
Who's that Pokémon (type)! An NLP-based Pokémon classifier that predicts a Pokémon’s type via logistic regression trained on n-gram features extracted from Pokédex descriptions.<br>
docker run -p 8888:8888 my-jupyter <br>
http://localhost:8888 <br>
**Credits:**<br>
https://medium.com/@18bhavyasharma/setting-up-and-running-jupyter-notebook-in-a-docker-container-d2acd713ce66 <br>
https://www.youtube.com/watch?v=7-naqq9fvZE <br>
https://medium.com/analytics-vidhya/predicting-pok%C3%A9mon-type-with-the-pok%C3%A9dex-7038754dc422

## Introduction <br>
I love Pokémon, so when I came across this dataset on Kaggle, I had to make use of it. The thought process was simple - there are 18 types of Pokémon, each based on a certain element (i.e Water, Fire, Lightning, etc.) Each Pokémon is categorized into one type (with some having 2 types). In addition, each Pokémon has a written description. Here's an example from the Dataset: 
Name: Bulbasaur <br>
Type: Grass/Poison <br>
Description: A strange seed was planted on its back at birth. The plant sprouts and grows with this Pokémon. <br>
As you can see in the description, there's "clues" that imply what Bulbasaur’s type is - the word "seed," "plant," and "sprout" have associations with the word "grass." I decided to use NLP techniques to train a model on a portion of the Pokémon descriptions and type. From there, I would evaluate my implementation with a test set of the data.  <br>

## First Attempt
I'm not some sort of genius, so my first attempt was really simple. First, I imported the data from Kaggle. I extracted the labels – the Pokémon types, and removed punctation and uppercase from the Pokémon descriptions. I decided to keep things simple and only focus on the “primary” type for a  Pokémon (For example, as you see above, Bulbasaur is a Grass and Poison type. Instead of classifying this as its own type like “Grass-Poison,” I’ve decided to leave it as just a Grass type.)
<br>
Then, I split the data with 80% of the data being used for training and 20% being used for evaluation and vectorized the features and labels:
<br>
```python
x = pokemon["desc_clean"]
y = pokemon["type_primary"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4, stratify=y)

vectorizer = CountVectorizer(ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
```

After that, I encoded the labels:

```python
label_train = train_df["type_primary"]
label_test = test_df["type_primary"]

encoder = LabelEncoder()
encoder.fit(pokemon["type_primary"])

label_train_enc = encoder.transform(label_train)
label_test_enc = encoder.transform(label_test)
```

Then, I set up my model, which is a simple Logistic Regression model, to make the predictions:

```python
model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
model.fit(X_train_vec, label_train_enc)
type_pred_enc = model.predict(X_test_vec)
type_pred = encoder.inverse_transform(type_pred_enc)
```

The results came out to be… extremely inaccurate, with some types were not being predicted at all, resulting in 0 F1 scores. I think this has to do with the data that exists – basically some types are more common than others, with water type being the most common. I decided to make some changes to account for this discrepancy in hopes of getting a little more accurate. I also wanted to experiment with the model I chose because I picked with I was most familiar with, but there could be a better option.<br>

## Attempt 2
As a matter of fact, I'm not the only one to have this idea. So upon doing further research, I found some other people's explorations on how they created a Pokemon type classifier. I got some good ideas from this. I removed common words like "can" and "the" from the Pokemon descriptions. This immediatly increased the f1 score from 0.28 to 0.35.
