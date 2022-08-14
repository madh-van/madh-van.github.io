+++
title = "Notes on Computer Vision"
author = ["Madhavan Krishnan"]
date = 2020-10-30
tags = ["Computer-Vision"]
draft = false
+++

[Reference](https://www.youtube.com/watch?v=ArPaAX_PhIs&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF))

Convolution Neural Network vs Fully connected Neural Network

-   Translation invarient
-   Sparsity of connection
-   Sharing of parameter

    {{< figure src="/ox-hugo/2022-07-13_22-54-59_Screenshot from 2022-06-25 22-01-20.png" >}}


## Image processing edge detection: {#image-processing-edge-detection}

{{< figure src="/ox-hugo/2022-07-13_23-04-12_Screenshot from 2022-06-25 22-04-49.png" >}}

Grey scale input \\(\*\\) kernal/filter (3x3 filter)

{{< figure src="/ox-hugo/2022-07-13_23-04-37_Screenshot from 2022-06-25 22-08-36.png" >}}

&gt; 1) shape of the output is \\(6 - 3 + 1 = 4\\)

&gt; 2) channel vise convolution operation

&gt; 3) like wise move the filter position by 1 on x axis

&gt; 4) repeate 3) until the end and then one step down like below.

{{< figure src="/ox-hugo/2022-07-13_23-05-00_Screenshot from 2022-06-25 22-30-44.png" >}}

&gt; 5) follow the above step till the end.

{{< figure src="/ox-hugo/2022-07-13_23-05-13_Screenshot from 2022-06-25 22-33-25.png" >}}

&gt; Technically this is called ****cross-correlation**** and convolution in technical terms will have 2 step preseding (but not used in practice).
&gt; 1) Mirror of filter Horizontally and then
&gt; 2) Mirror of filter Verfically.

{{< figure src="/ox-hugo/2022-07-13_23-06-04_Screenshot from 2022-06-26 10-54-05.png" >}}

----


### Vertical edge detector {#vertical-edge-detector}

{{< figure src="/ox-hugo/2022-07-13_23-06-52_Screenshot from 2022-06-25 22-37-16.png" >}}


### Horizontal edge detector {#horizontal-edge-detector}

{{< figure src="/ox-hugo/2022-07-13_23-07-20_Screenshot from 2022-06-25 22-42-26.png" >}}


### Sobel/Schar filter {#sobel-schar-filter}

{{< figure src="/ox-hugo/2022-07-13_23-07-35_Screenshot from 2022-06-25 22-50-41.png" >}}


## Learnable filter {#learnable-filter}

These parameters can learn through backpropagation/gradient decent the necessary filters that can better map the complex funtion.

{{< figure src="/ox-hugo/2022-07-13_23-08-02_Screenshot from 2022-06-25 22-51-52.png" >}}

&gt; usually the filter sizes are odd, 1x1, 3x3, 5x5, 7x7x (comes from computer vision letrature)

Athough with input with more than 1 channel, the channel size of the kernal will match the input channel.

{{< figure src="/ox-hugo/2022-07-13_23-08-38_Screenshot from 2022-06-26 11-49-22.png" >}}

Likewise there will be many such filters learning multiple low level features with learnable parameter.

{{< figure src="/ox-hugo/2022-07-13_23-09-04_Screenshot from 2022-06-26 12-00-39.png" >}}

&gt; Note; the output channel will be based on the number of filter.

{{< figure src="/ox-hugo/2022-07-13_23-09-18_Screenshot from 2022-06-26 12-59-47.png" >}}

The number of parameters = \\(n \* (k ^2 \* c + 1)\\)

-   n is the number of filter
-   k is the kernel size
-   c is the channel size

filter: k x k x n_c (number of channel from the input)
activation: n_h x n_w x n_f (number of filter)
weights: k x k x c x n_f
bias: 1 x 1 x 1 x n_f


## Padding {#padding}

-   Avoids strinking of output
-   Input in the edges are given priority

{{< figure src="/ox-hugo/2022-07-13_23-10-01_Screenshot from 2022-06-25 23-05-12.png" >}}

-   ****Valid convolution****: no padding (where p = 0)
-   ****Same convolution**** : padding to provide output shape same as input (where \\(p = \frac {f-1} 2\\)).

\\[
\lfloor {\frac {input size + (2 \* padding) - filter size} {strides} + 1}\rfloor
\\]

&gt; If not an integer, should round it down (floor)


## Strides {#strides}

{{< figure src="/ox-hugo/2022-07-13_23-14-04_Screenshot from 2022-07-13 23-13-23.png" >}}


## Pooling {#pooling}

-   Max pooling for filter size 2 and stride 2.

{{< figure src="/ox-hugo/2022-07-13_23-14-51_Screenshot from 2022-06-26 16-54-02.png" >}}

&gt; Note;
&gt; 1) Max pool operation is done on a per channel basis, so for \\(n\\) dimentions, the output will also be \\(n\\).
&gt; 2) There is no learnable parameter for this layer.

Like wise there is average pooling (although its not used widely)
Its can be used at the deeper layers where say 7x7x1000 -&gt; 1x1x1000
commonly used values are for; f=2,  s=2 and f=3, s=2.


## Fully connected {#fully-connected}

{{< figure src="/ox-hugo/2022-07-13_23-15-32_Screenshot from 2022-06-26 19-38-43.png" >}}


## Different volumes {#different-volumes}

{{< figure src="/ox-hugo/2022-07-13_23-16-07_Screenshot from 2022-06-26 21-04-07.png" >}}

{{< figure src="/ox-hugo/2022-07-13_23-16-11_Screenshot from 2022-06-26 21-08-19.png" >}}


## 1x1 convolution {#1x1-convolution}


## Inception {#inception}

{{< figure src="/ox-hugo/2022-07-13_23-16-42_Screenshot from 2022-06-29 14-29-27.png" >}}

{{< figure src="/ox-hugo/2022-07-13_23-16-46_Screenshot from 2022-06-29 14-31-54.png" >}}


## <span class="org-todo todo TODO">TODO</span> sliding window with convolution {#sliding-window-with-convolution}

Image classification

Object localization
Object detection

Image segmentation
Instance segmentation

YOLO, Image segmentation
6 Dof pose estimation methods

Style transfer