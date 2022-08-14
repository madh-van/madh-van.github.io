+++
title = "Notes on Natural Language Processing"
author = ["Madhavan Krishnan"]
date = 2020-10-30
tags = ["Text-analysis"]
draft = false
+++

-   <https://www.youtube.com/watch?v=_i3aqgKVNQI&list=PLkDaE6sCZn6F6wUI9tvS_Gw1vaFAx6rd6>
-   <https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa>
-   <https://analyticsindiamag.com/word2vec-vs-glove-a-comparative-guide-to-word-embedding-techniques/>
-   <https://towardsdatascience.com/word-embedding-techniques-word2vec-and-tf-idf-explained-c5d02e34d08>


## N-gram {#n-gram}

n pairs of words in sentence.


## Bleu {#bleu}

Bilingual evaluation understudy

\\[
bleu\ score = \frac {max\ number\ of\ occurances\ in\ reference} {total\ unique\ ngram}
\\]

{{< figure src="/ox-hugo/2022-07-13_22-57-09_Screenshot from 2022-06-26 21-50-58.png" >}}


### On uni and n grams {#on-uni-and-n-grams}

{{< figure src="/ox-hugo/2022-07-13_22-58-01_Screenshot from 2022-06-26 21-51-23.png" >}}


## RNN {#rnn}

{{< figure src="/ox-hugo/2022-07-13_23-20-12_Screenshot from 2022-07-03 08-47-18.png" >}}

{{< figure src="/ox-hugo/2022-07-13_23-20-54_Screenshot from 2022-06-28 22-53-01.png" >}}

\\[
h\_t = tanh(W\_{hs} h\_{t-1} + W\_{x} x\_t)
\\]

The hidden state \\(h\_t\\) is constantly rewritten at every time step causing to vanishing gradient over time. This makes the network not learn from dependencies from a longer time period.

To overcome this issue 2 types of RNN was developed.


## LSTM (Long Short Term Memory) {#lstm--long-short-term-memory}

Gate

{{< figure src="/ox-hugo/2022-07-13_23-21-17_Screenshot from 2022-07-03 11-44-52.png" >}}

Remove or Add information from the cell state.

1.  Cell state \\(C\_t\\)

{{< figure src="/ox-hugo/2022-07-13_23-21-39_Screenshot from 2022-07-03 11-45-49.png" >}}

1.  Forget gate layer (What info to remove in cell state)

{{< figure src="/ox-hugo/2022-07-13_23-22-06_Screenshot from 2022-07-03 11-20-44.png" >}}

\\(f\_t = \sigma(W\_f . [h\_{t-1}, x\_t] + b\_f)\\)

-   Output 0,1 (for each number in the cell state \\(C\_{t-1}\\))
-   Input gate (What info to store in cell state)

{{< figure src="/ox-hugo/2022-07-13_23-22-24_Screenshot from 2022-07-03 11-49-48.png" >}}

3.1) \\(i\_t = \sigma(W\_i . [h\_{t-1}, x\_t]+ b\_i)\\) decides which input shall pass

3.2) \\(\tilde{C} = tanh(W\_c . [h\_{t-1}, x\_t]+ b\_c)\\) creates vector for cell state to be added into.

1.  Update the cell state

{{< figure src="/ox-hugo/2022-07-13_23-22-36_Screenshot from 2022-07-03 11-46-40.png" >}}

\\[ C\_{t} = f\_t \* C\_{t-1} + i\_t \* \tilde{C}
 \\]

1.  Output

{{< figure src="/ox-hugo/2022-07-13_23-22-49_Screenshot from 2022-07-03 11-56-01.png" >}}

\\[
o\_t = \sigma(W\_o . [h\_{t-1}, x\_t] + b\_o)\\\\
h\_t = o\_t \* tanh(C\_t)
\\]


### Variants of LSTM {#variants-of-lstm}

1.  Adds peepholes (context for the gates on the current cell state \\(C\_t\\)).

{{< figure src="/ox-hugo/2022-07-13_23-23-01_Screenshot from 2022-07-03 12-03-43.png" >}}

\\[
f\_t = \sigma(W\_f . [\boxed{C\_t}, h\_{t-1}, x\_t] + b\_f)\\\\
i\_t = \sigma(W\_i . [\boxed{C\_t}, h\_{t-1}, x\_t]+ b\_i) \\\\
o\_t = \sigma(W\_o . [\boxed{C\_t}, h\_{t-1}, x\_t] + b\_o)\\\\
\\]

1.  Coupled input and forget gates

{{< figure src="/ox-hugo/2022-07-13_23-23-22_Screenshot from 2022-07-03 12-08-50.png" >}}

\\[
C\_t = f\_t \* C\_{t-1} + \boxed{(1-f\_t)} \* \tilde{C}
\\]


## GRU (Gated Recurrent Unit) {#gru--gated-recurrent-unit}

1.  GRU (Gated Recurrent Unit)

2.  Merge forget and Input gate
3.  Cell state and Hidden state.

{{< figure src="/ox-hugo/2022-07-13_23-23-37_Screenshot from 2022-07-03 12-25-12.png" >}}

\\[
z\_t = \sigma(W\_z . [h\_{t-1}, x\_t]) \\\\
r\_t = \sigma(W\_r . [h\_{t-1}, x\_t])\\\\
\tilde{h}\_t = tanh(W . [r\_t \* h\_{t-1}, x\_t])\\\\
h\_t = (1-z\_t) \* h\_{t-1} + z\_t \* \tilde{h\_t}
\\]

1.  BiLSTM

{{< figure src="/ox-hugo/2022-07-13_23-24-00_Screenshot from 2022-07-03 15-32-23.png" >}}

-   Reference
    -   <https://medium.com/analytics-vidhya/rnn-vs-gru-vs-lstm-863b0b7b1573>
    -   <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>
    -   <https://distill.pub/2016/augmented-rnns/>


## <span class="org-todo todo TODO">TODO</span> Embeddings from language models (EL-mo) {#embeddings-from-language-models--el-mo}

{{< figure src="/ox-hugo/2022-07-13_23-24-37_Screenshot from 2022-07-03 20-27-54.png" >}}

or

{{< figure src="/ox-hugo/2022-07-13_23-24-46_Screenshot from 2022-07-03 20-34-04.png" >}}

-   word representations are from entire input sentence.
-   predicts the next char from inputs seen so far.
-   cost funtion: maximizes log likelihood of forward and backward prediction.

Note;

use this

{{< figure src="/ox-hugo/2022-07-13_23-25-22_Screenshot from 2022-07-03 20-36-03.png" >}}

or

{{< figure src="/ox-hugo/2022-07-13_23-25-41_Screenshot from 2022-07-03 20-36-38.png" >}}

-   Reference

<https://paperswithcode.com/method/elmo>
<https://arxiv.org/pdf/1802.05365v2.pdf>
<https://iq.opengenus.org/elmo/>
<https://indicodata.ai/blog/how-does-the-elmo-machine-learning-model-work/>


## Attention {#attention}

Attention \\(\alpha\\) is weighted sum of activation ouput;

where  \\(\sum\_{\grave{t}} \alpha^{<1,\grave{t}>} = 1\\) (ofcouse with softmax :P) and the context \\(C = \sum\_{\grave{t}} \alpha^{<1,\grave{t}>} a^{<\grave{t}>}\\)

Here \\(a^{\grave{t}}\\) is the activation from bidirectional rnn (from both the forward and backward).

the attention is computed with

{{< figure src="/ox-hugo/2022-07-13_23-27-26_Screenshot from 2022-07-04 21-20-00.png" >}}

\\[
 e^<t,{\grave{t}}> = W
 \\]

Time complexity is \\(t\_x t\_y\\) where \\(t\_x\\) is the input length and \\(t\_y\\) the output time length. Since for every output prediction we are calculating the attention \\(\alpha\\) for every input to generate the context \\(C\\).

<https://arxiv.org/pdf/1502.03044v2.pdf>


## Transformers {#transformers}

<https://www.tensorflow.org/text/tutorials/transformer>
<http://jalammar.github.io/illustrated-transformer/>


### Bert model {#bert-model}

Similar to ELmo, Bert also provides a contextual representation of the word embedding, but looks at the sentence all at once (with out having to concatenate forward and backward contextural hidden state vectors).

<https://medium.com/analytics-vidhya/understanding-the-bert-model-a04e1c7933a9>
<https://www.youtube.com/watch?v=xI0HHN5XKDo>


## Audio data {#audio-data}

Preprocessed to get the spectrogram (a visual representation of audio).

{{< figure src="/ox-hugo/2022-07-13_23-28-11_Screenshot from 2022-07-04 22-09-59.png" >}}

x axis time
y axis small change in air pressure.

-   when the input length is higher than output length

{{< figure src="/ox-hugo/2022-07-13_23-28-24_Screenshot from 2022-07-04 22-00-09.png" >}}

-   A simple trigger word detection model.

{{< figure src="/ox-hugo/2022-07-13_23-28-46_Screenshot from 2022-07-04 22-09-06.png" >}}