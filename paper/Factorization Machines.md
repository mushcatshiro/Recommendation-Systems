[TOC]

# Factorization Machines

SVM works great as general predictor in recommendation systems however SVM do not model the interaction between variables. This could be a problem when we are dealing with highly sparse data often seen in large scale production recommendation systems. FM addresses this limitation through learning reliable parameters by modelling every interactions available in the sparse dataset. Another advantage of using FM is that it is processed in linear time and can be optimized directly. this is largely due to the model equation can be simplified from
$$
\hat{y}(x) := w_0 + \sum_{i=1}^{n}{w_ix_i} + \sum_{i=1}^{n}\sum_{j=i+1}^{n}{<v_i, v_j>x_ix_j}
\\
w_0 \in \R, \bold{w} \in \R^2, \bold{V} \in \R^{n \times k}
$$
to
$$
\hat{y}(x) := w_0 + \sum_{i=1}^{n}{w_ix_i} + \frac{1}{2}\sum_{f=1}^{k}((\sum_{i=1}^{n}v_{i，f}x_i)^2-\sum_{i=1}^{n}v_{i,f}^{2}x_i^2)
$$
the pairwise interaction has no model parameter that directly depends on two variables i.e. a parameter with both i and j index. furthermore under sparsity most of the elements in $x$ are 0 thus the sum is only required to be computed over those non-zero elements resulting in $O(k\bar{m}_D)$ complexity where $\bar{m}_D$ is 2 for two way FM (capturing all single and pairwise interaction). In a two way FM $w_0$ is the global bias, $w_i$ models the strength of the i-th variable and $\hat{w}_{i, j}$ models the interaction between the i-th and j-th variable. FM models the interaction by factorizing it instead of using its own model parameter for each pair of interaction. As we have shown that FM have a closed model equation that can be computed in linear time. Thus, the model parameters ($w_0$, $\bold{w}$ and $\bold{V}$) can be learned efficiently by gradient descent methods eg SGDs for a variety of losses. the gradient is as follows,
$$
\frac{\partial}{\partial{\theta}} \hat{y}(x) = \left\{
\begin{array}{ll}
      1 & \text{if } \theta \text{ is } w_o \\
      x_i & \text{if } \theta \text{ is } w_i \\
      x_i\sum_{j=1}^{n}{v_{j,f}x_j - v_{i,j}x_i^2} & \text{if } \theta \text{ is } v_{i,f} \\
\end{array} 
\right.
$$
the sum $v_{i,f}$ is independent of $i$ and thus can be precomputed when we are computing for $\hat{y}(x)$ which results that the gradient can be computed at constant time $O(1)$  and the remaining parameters can be completed in $O(kn)$ or $O(km(x))$ under sparsity.

## SVD vs FM

### SVD model

for both linear kernel we can see that their model equation are identical to FM model equation and for polynomial (d=2) kernel the model equation is similar to FM model equation with the exception that all interaction parameters $w_{i,j}$  are completely independent in contrast to FMs interaction parameters where they are dependent to each other.
$$
\hat{y}(x) = w_0 + \sqrt{2}\sum_{i=1}^{n}{w_ix_i} + \sum_{i=1}^{n}{w_{i,i}^2x_i^2} + \sqrt{2}\sum_{i=1}^{n}\sum_{j=i+1}^{n}{w_{i,j}^2x_ix_j}
$$


### parameter estimation under sparsity

linear and polynomial SVM both fails at sparse problem with the proof as follows. taking a sparse matrix where the feature vectors are only two elements (user and active item).

#### 1. linear SVM

with this kind of data linear SVM is simplified to the following
$$
\hat{y}(x) = w_0 + w_u + wi
$$


this corresponds to the most basic collaborative filtering models where the user and item biases are captured. the model is simple however the empirical prediction quality remains the same regardless to the dimensionality.

#### 2. polynomial SVM

taking the polynomial (d=2) from the equation shown above and applying to the described data, the equation is simplified to the following
$$
\hat{y}(x) = w_0 + \sqrt{2}{(w_u+w_i)} + {w_{u,u}^2 + w_{i,i}^2} + \sqrt{2}{w_{u,i}^2}
$$
we can drop ${w_{u,u}^2 + w_{i,i}^2}$ as it expressed the same as ${(w_u+w_i)}$ and the model equation will be same as the linear equation with an additional user-item interaction parameter. however for each interaction parameter there is at most one observation of the user-item pair and leads to the maximum margin solution to be 0. this prevents SVM from predicting with 2-way interaction and thus it is no better than the linear SVM. similar observation can be found on matrix factorization models.

in summary, FM works well with sparsity and can model higher order interaction between features.
