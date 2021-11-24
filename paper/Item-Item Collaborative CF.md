[TOC]

# Item-Item Collaborative Filtering

the goal is to develop something efficient while providing good enough results. the item to item CF approach made this possible as,

- although theoretically its $O(N^2M)$ complexity, but in practice its $O(MN)$ as most customers have less purchases
- it depends only on the number of items users purchased or rated

> question, to figure out why point 2 is the case

## Extracts

- ...(on traditional CF) The components of the vector are positive for purchased or positively rated items and negative for negatively rated items
- ...(on traditional CF) multiplies the vector components by the inverse frequency (the inverse of the number of customers who have purchased or rated the item), making less well-known items much more relevant
- ...(on traditional CF) a few customers who have purchased or rated a significant percentage of the catalog, requiring O(N) processing time. Thus, the final performance of the algorithm is approximately O(M + N)
- ...(on traditional CF) Dimensionality reduction degrade recommendation quality
- ...(on cluster models) greedy cluster generation. These algorithms typically start with an initial set of segments, which often contain one randomly selected customer each. They then repeatedly match customers to the existing segments, usually with some provision for creating new or merging existing segments
- ...(on cluster models) Some algorithms classify users into multiple segments and describe the strength of
  each relationship
- ...(on I2ICF scalability) item-to-item collaborative filtering’s scalability and performance is that it creates the
  expensive similar-items table offline...online component - looking up similar items for the user’s purchases and ratings

## Algorithm

```pseudocode
for each item in product catalog, I1
	for each customer C who purchased I1
		for each item I2 purchased by customer C
			record that a customer purchased I2
	for each item I2
		compute similarity between I1 and I2
```

we should get a item to item matrix from the described algorithm. the paper describes using cosine similarity to calculate similarity however not mentioning on the final recommendation scoring.

### related articles

...from some towardsdatascience post

1. similarity function (cosine similarity with adjustments)

$$
similarity(i,j) = \frac{\sum_u^U(r_{(u,i)}-\bar{r}_u)(r_{(u,j)}-\bar{r}_u)}{\sqrt{\sum_u^Ur^2_{(u,i)}}\sqrt{\sum_u^Ur^2_{(u,j)}}}
$$

2. score function

$$
score(u,i) = \frac{\sum_j^isimilarity(i,j)(r_{u,j}-\bar{r}_j)}{\sum{_j^isimilarity(i,j)}} + \bar{r}_j
$$

___

## thoughts

on score function 

- can we count frequencies and normalize instead