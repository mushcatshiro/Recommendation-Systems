[TOC]

# 01 - introduction to recsys

trying to suggest to an entity based on the other entities and its feedback.

## implicit data and explicit data

implicit examples

- user often listen to same genre of music
- item is repeatedly purchased by same user
- viewer gives up halfway on a movie

explicit

- ratings/feedback given
- complains on social media

## non personalized recommendations

made towards all users without taking personal preference into account, eg. amazon "frequently buy together" something applies to a wider crowd and you are less likely to hate the suggestion. take a movie rating dataset as an example, a naïve approach would be to get the mean rating for each movies sort them descending and take the first few. this might be flawed if there is a movie with only one rating and its 5 out of 5. we could mask movies with a threshold count to eliminate such scenario. to further improve the recommendation we could identify pairs by creating a product A product B hashmap (or similar) for all permutation including product B and product A.

## content-based recommendation

### Jaccard similarity

after we extract the genres from imdb movie's dataset, we need something that can help us to compare between movies.
$$
J(A, B) = \frac{A \cap B}{A \cup B}
$$

```python
from scipy.spatial.distance import pdist, squareform

jaccard_distances = pdist(movie_cross_table.values, metric='jaccard')

jaccard_similarity_array = 1 - squareform(jaccard_distances)

jaccard_similarity_df = pd.DataFrame(
    jaccard_similarity_array,
    index=movie_cross_table.index,
    columns=movie_cross_table.index
)
```

### text based similarities

$$
\text{TF-IDF} = \frac{\frac{\text{count of word occurences}}{\text{total words in document}}}{\log({\frac{\text{number of docs word is in}}{\text{total number of docs}}})}
$$

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df=2, max_df=0.7)
vectorized_data = vectorizer.fit_transform(df_plots['Plot'])
tfidf_df = pd.DataFrame(vectorized_data.toarray(), columns=vectorizer.get_feature_names())
tfidf_df.index = df_plots['Title']
```

$$
cos(\theta) = \frac{A\cdot B}{\abs{A}\cdot \abs{B}}
$$

```python
from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity_array = cosine_similarity(tfidf_summary_df)
cosine_similarity_df = pd.DataFrame(
    cosine_similarity_array,
    index=tfidf_summary_df.index,
    columns=tfidf_summary_df.index
)
```

### user profile recommendations

from tfidf dataframe we can extract what the user likes from past history and take a summary of the person's taste, ie which word interest the person the most, thus we obtained the user's profile. based on this user profile we then calculate against the tfidf.

```python
# datacamp uses reindex for filtering however seems like it might be outdated version of pandas...
list_of_books_read = ['The Hobbit', 'Foundation', 'Nudge']
user_books = tfidf_summary_df.reindex(list_of_books_read)
user_prof = movies_enjoyed_df.mean()
tfidf_subset_df = tfidf_df.drop(list_of_movies_enjoyed, axis=0)
similarity_array = cosine_similarity(user_prof.values.reshape(1, -1), tfidf_subset_df)
similarity_df = pd.DataFrame(
    similarity_array.T,
    index=tfidf_subset_df.index,
    columns=["similarity_score"]
)
```

## collaborative filtering

content-based recommendation works well when we have lots of information about the items, but not the user's feeling. CF can be done between users using rating data.

note when we centering or normalize for NaN values, the resulting dataframe should **<u>not</u>** be used for prediction as the value 0 means differently when we are comparing similarity (a mean score for that particular user) and calculating for predictions (user don't like).

### user-cf example

```python
user_ratings_table = user_ratings.pivot(index='userId', columns='title', values='rating')
avg_ratings = user_ratings_table.mean(axis=1)
user_ratings_table_centered = user_ratings_table.sub(avg_ratings, axis=0)
user_ratings_table_normed = user_ratings_table_centered.fillna(0)
```

### item-cf example

cosine similarity from `sklearn.metrics.pairwise` can be used to compare A to B or pairwise between matrix A.

```python
avg_ratings = user_ratings_table.mean(axis=1)
user_ratings_centered = user_ratings_table.sub(avg_ratings, axis=0)
user_ratings_centered = user_ratings_centered.fillna(0)

similarities = cosine_similarity(movie_ratings_centered)
cosine_similarity_df = pd.DataFrame(
    similarities,
    index=movie_ratings_centered.index,
    columns=movie_ratings_centered.index
)

cosine_similarity_series = cosine_similarity_df.loc['Star Wars: Episode IV - A New Hope (1977)']
ordered_similarities = cosine_similarity_series.sort_values(ascending=False)
```

### KNN

```python
# from scratch
user_similarity_series = user_similarities.loc['user_001']
ordered_similarities = user_similarity_series.sort_values(ascending=False)
nearest_neighbors = ordered_similarities[1:11].index
neighbor_ratings = user_ratings_table.reindex(nearest_neighbors)

# using sklearn
users_to_ratings.drop("Apollo 13 (1995)", axis=1, inplace=True)
target_user_x = users_to_ratings.loc[["user_001"]]
other_users_y = user_ratings_table["Apollo 13 (1995)"]
other_users_x = users_to_ratings[other_users_y.notna()]
other_users_y.dropna(inplace=True)

from sklearn.neighbors import KNeighborsRegressor

user_knn = KNeighborsRegressor(metric='cosine', n_neighbors=10)
user_knn.fit_transform(other_users_x, other_users_y)
user_user_pred = user_knn.predict(target_user_x)
```

### when to use item-cf or user-cf

#### item-cf

pros:

- consistent over time
- easier to explain
- can be pre-calculated

cons:

- very obvious suggestion

#### user-cf

pros:

- more interesting suggestion (youtube wide and deep's wide?)

cons:

- generally bested by item-cf using standard metrics

## Matrix Factorization and Validation

### dealing with sparsity

$$
Sparsity = \frac{\text{Empty Values}}{\text{Total Cells}}
$$

sparsity is bad for KNN because for a particular item with few ratings it will just return the average value of the available values. MF solves this by decompose it into product of two lower dimensional matrices then dot product them to get a full matrix

```python
sparsity_count = user_ratings_df.isnull().values.sum()
full_count = user_ratings_df.size
sparsity = sparsity_count / full_count
```

### matrix factorization

min requirement at least 1 value in every row and column. the additional column and row created is called latent features. note the dot product matrix will be slightly different from the original matrix. rank refers to how many unnamed columns and rows created by.

### svd

$$
SVD = U \cdot \Sigma \cdot V ^ T
$$

where `U` is an m x m unitary matrix (user matrix in this course), `sigma` is a diagonal matrix which represents the weight of latent feature and `V` represents an n x n unitary matrix (item matrix in this course)

```python
from scipy.sparse.linalg import svds
import numpy as np

U, sigma, Vt = svds(user_ratings_centered)
sigma = np.diag(sigma)
U_sigma = np.dot(U, sigma)
U_sigma_Vt = np.dot(U_sigma, Vt)
uncentered_ratings = U_sigma_Vt + avg_ratings.values.reshape(-1, 1)
calc_pred_ratings_df = pd.DataFrame(
    uncentered_ratings, 
    index=user_ratings_df.index,
    columns=user_ratings_df.columns
)
user_5_ratings = calc_pred_ratings_df.loc['User_5',:].sort_values(ascending=False)
```

### validating perdictions

in traditional ML we predict a single feature or column, in recsys its much random thus the holdout / test set should look a bit different. instead of separating rows, we separates a block of entries. RMSE is usually the score function used here.

```python
actual_values = act_ratings_df.iloc[:20, :100].values
avg_values = avg_pred_ratings_df.iloc[:20, :100].values
predicted_values = calc_pred_ratings_df.iloc[:20, :100].values

mask = ~np.isnan(actual_values)

print(mean_squared_error(actual_values[mask], avg_values[mask], squared=False))
print(mean_squared_error(actual_values[mask], predicted_values[mask], squared=False))
```



___

## reference

### cosine similarity and jaccard similarity
