+++
title = "Practical aspects of Deeplearning"
author = ["Madhavan Krishnan"]
date = 2020-10-30
categories = ["Deeplearning"]
draft = false
+++

Build a system first and then iteratevely improve upon the system with
[bias and variance/ error correct](#bav)


## Numpy vs Tensor {#numpy-vs-tensor}

| Framework  | Data structure | Property  | Targets     | Automatic differentiation | Data type        |
|------------|----------------|-----------|-------------|---------------------------|------------------|
| tensorflow | tensors        | immutable | CPU/GPU/TPU | Yes                       | Numerical/String |
| numpy      | array          | mutable   | CPU         | No                        | Numerical        |

1.  A tensor is not actually a grid of values but the computations that lead to those values
2.  Different type of tensors
    -   <https://www.tensorflow.org/guide/tensor#sparse_tensors>
    -   <https://www.tensorflow.org/guide/tensor#ragged_tensors>


## Bias and Variance {#bias-and-variance}

|          | High varience | High bias    | high bias &amp; varience | low bias &amp; variance |
|----------|---------------|--------------|--------------------------|-------------------------|
| tain set | low error     | medium error | medium error             | low error               |
| val set  | high error    | medium error | high error               | low error               |

&gt; More on that in [splitting data](#splitting_data)
&gt; - Error is in respect to the  bayes error (optimal error).
&gt; - Avoidable bias is the difference between bayes error and training error.
&gt; - Its not always easy to find bayes error(human level performance) on large structured dataset.

These steps help with the [orthogonalization](#orthogonalization)

1.  `Hight Bias` ? If Yes (Try training deeper models and or Longer training period or with a different [Optimizer](#optimizers), NN architecture/hyperparameter search)
2.  `Hight Variance` ? If Yes (Try adding more data/augmenation and or [Regularization](#regularization), [Error analysis](#error_analysis))

3.  If the model now doesn't perform well on the test set, increase the validation set.


## Orthogonalization {#orthogonalization}

Tune one parameter at a time to have a clear view over its impact on the training process.


## Hyper parameter {#hyper-parameter}

by the order of importance

-   \\(\alpha\\) learning rate
-   \\(\beta\\) momentum \\(~0.9\\)
-   \\(\beta\_1 0.9, \beta\_2 0.999, \epsilon 10^{-8}\\) for Adam optimizer.
-   Number of hidden units
-   Mini-batch size
-   Number of layers
-   Learning rate decay

Grid search of optimal hyperparameter might not work for deeplearning, since not eache hyperparameter is equally weight in its importance.

Random sampling is good to initally figure out the area in the grid that gives good results; later on reduce the search space to this and try again more densly in this reduced space.

Ensure the scale of the hyperparameter is appropriate; for example \\(\alpha\\) learning rate should be in log scale.

{{< figure src="/ox-hugo/2022-07-13_22-46-16_Screenshot from 2022-06-16 21-07-40.png" >}}

```python
r = -4 * np.random.rand() # [-4, 0]
a = 10**r #[10^-4 .. 10^-0]
r, a
```

Similarly for \\(\beta\\) its of the range \\([0.9 .. 0.999]\\)

\\((1-\beta) = [0.1 .. 0.001]\\)
and `r` is the range -3, -1

```python
r = -2 * np.random.rand()-1 # [-3, 1]
a = 1- 10**r # [10^-3 .. 10^1]
r, a
```


## Local optima {#local-optima}

The chances of getting gradient 0 in a higher dimention is quite small. Its mostly going to be a ****saddle point****. (Saddle because its like on the horse)

{{< figure src="/ox-hugo/2022-07-13_22-47-07_Screenshot from 2022-06-18 16-06-07.png" >}}


## Analyse the performace {#analyse-the-performace}


### Evaluation metric {#evaluation-metric}

Use a ****single real value**** to evaluate the models performace quickly.

Following can be a error metric;
\\[
Accuracy - 0.5 \* lattency\\\\
\frac 1 {\sum\_i^{} w^{(i)}} \sum\_i^{m\_{dev}} w^{(i)} (y\_{pred}^{(i)} \neq y^{(i)})\\\\
where w is 1 for x^{(i)} is non-porn
10 if x^{(i)} is porn.
\\]
F1 score; Takes the average (Harmonic mean) of precision and recall
\\(\frac 2 {\frac 1 {Precision} + \frac 1  {Recall}}\\)

If there are N metrics you care about,

1.  use 1 as the optimizer and
2.  N-1 as satisfier (as long as it meets some accetable threshold)

Then optimize your training for the above metic

\\(J = \frac 1 {\sum\_i^{m}w^{(i)}} \sum\_i^{m}w^{(i)} l(\hat{y}^i, y^i)\\)


## Dataset {#dataset}


### Splitting {#splitting}

The dataset is usually split into three.

| Splits     | Small dataset | Large dataset |
|------------|---------------|---------------|
| Train      | 70%           | 98%           |
| Validation | 15%           | 1%            |
| Test       | 15%           | 1%            |

&gt; Note; since the 1% of a large dataset is enough to validation the model.

-   ****Ensure the validation and test set come from the same distribution****.
-   The test set is to have an unbiased performace estimation of your model (which is not mandotory).

But when the ****training data is not the same distribution as the validation and test****, the training data is split into

| Split                                 | Error (in distribution A) Training data             | Error (in distribution B) Real world data | Type                                 |
|---------------------------------------|-----------------------------------------------------|-------------------------------------------|--------------------------------------|
| Human level \\(\approx\\) Bayes error | 4%                                                  | 6%                                        |                                      |
| Train                                 | 7%                                                  | 6%                                        | Hight Bias (7%-4% is avoidable bias) |
| ****Train-val****                     | 10%                                                 |                                           | High Variance (10%-7% is vaiance)    |
| Validation                            |                                                     | 12%                                       | Distribution shift (12%-10%)         |
| Test                                  |                                                     | 12%                                       | Over fitted on the validation split  |
|                                       | Distribution shift across these two colums of error |                                           |                                      |

&gt; To find the error is due to hight varience or due to different distribution.

&gt; Note: The different in error between train-val and validation
will give the error added due to different distribution.


### Error analysis {#error-analysis}

-   Use confusion matrix on the validation set and or filtering the misslabeled images as below.

{{< figure src="/ox-hugo/2022-07-14_00-05-53_Screenshot from 2022-06-24 22-44-43.png" >}}

-   Pro tip;
    1.  Catch misslabelled ground truth is a seprate column
    2.  Random errors on large datset is fine; only systematic errors are an issue).
        -   In case of misslabelled data, its wise to look at correctly predicted classes as well.
    3.  Ensure to do the same process to your validation set as well as the test set.


### Addressing data mismach {#addressing-data-mismach}

-   Making training data more similar (data synthesis) and or collecting more data similar to validation/ test sets.


## Transfer learning {#transfer-learning}

{{< figure src="/ox-hugo/2022-07-13_22-49-14_Screenshot from 2022-06-25 15-47-41.png" >}}

When you have trained a large set of data on a deep learning model you can expect the initial layers will learn useful features that can be reused for other similar task. For example

Using the features from the model trained on large `imagenet dataset`
\\(\color{blue}{x,y}\\) to retrain on small `radiology dataset`
\\(\color{purple}{x,y}\\) by replacing the final layers with one or more
layers to predict the \\(\color{purple}{\hat{y}}\\)

Note;

-   Pretraining is on the \\(\color{blue}{x,y}\\)  and
-   Fintuning is on the \\(\color{purple}{x,y}\\) .


## Muititask learning {#muititask-learning}

{{< figure src="/ox-hugo/2022-07-13_22-50-24_Screenshot from 2022-06-25 17-07-11.png" >}}

\\(y^{(i)}\\) will be \\([0,1,1,0]\\)

\\[
J = \frac 1 {m} \sum\_{i=1}^m \sum\_{j=1}^{number of class} L(\hat{y}\_j^i, y\_j^i)
\\]
&gt; when there is missing label data for some class you should only consider summing over the classes where there is label 0/1 in the cost funtion.

where the loss funtions is same as [logistic loss](#logistic_loss).

Tips for training a multitask learning model;

1.  Ensure sharing of low features benefits in training the model recognising different tasks.
2.  Would need a big model to train well on all the task else the performace will get a hit.
3.  There should be similar amount of data for each task.


## End to end approach {#end-to-end-approach}

When you have ****enough data**** to map a complex funtion from input to output,
an end to end deeplearning approach can be used, rather than hand designed feature.

This is not always straight forward, it differs from application to applicaiton.

-   [X] Speech recognition example:

{{< figure src="/ox-hugo/2022-07-13_22-52-03_Screenshot from 2022-06-25 21-33-09.png" >}}

-   [ ] Face recognition example:

{{< figure src="/ox-hugo/2022-07-13_22-52-50_Screenshot from 2022-06-25 21-34-08.png" >}}