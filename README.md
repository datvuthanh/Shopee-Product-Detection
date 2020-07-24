# Shopee Product Detection
* Kaggle kernel has a lot of limitation: cannot use ModelCheckpoint, the amount of data that can be trained & training time is limited etc
* Anyway, this is only baseline for anyone who wants to train a model uses TPU at Kaggle kernel
* To get a great score, we must use stacking-ensemble method (only for competition, not for real-life)
![](https://scontent.fhan5-1.fna.fbcdn.net/v/t1.0-9/104756059_581915206028557_4469050410893255072_n.jpg?_nc_cat=109&_nc_sid=730e14&_nc_ohc=78c28WfAhYwAX92hd4F&_nc_ht=scontent.fhan5-1.fna&oh=c6fa725e8bac7ec144a03bdea75a3248&oe=5F23AC2F)

![](https://miro.medium.com/max/820/1*Y-629BmgDNFpLumnklJyaA.png)
# Efficient-Net
In May 2019, Google published both a very exciting paper and source code for a newly designed CNN called EfficientNet, that set new records for both accuracy and computational efficiency. Here’s the results of EfficientNet, scaled to different block layers (B1, B2, etc) vs. most other popular CNN’s.

![](https://miro.medium.com/max/985/1*nQ5HYZ1xiIGn092Y5H5SIQ.jpeg)

As the image shows, EfficientNet tops the current state of the art both in accuracy and in computational efficiency. How did they do this?

## Model scaling
They learned that CNN’s must be scaled up in depth, width, and input image resolution together to improve the performance of the model. The scaling method is named compound scaling and suggests that instead of scaling only one model attribute out of depth, width, and resolution; strategically scaling all three of them together delivers better results.

There is a synergy in scaling depth, width and image-resolution together, and after an extensive grid search derived the theoretically optimal formula of “compound scaling” using the following co-efficients:

Depth = 1.20
Width = 1.10
Resolution = 1.15
Depth simply means how deep the networks is which is equivalent to the number of layers in it. Width simply means how wide the network is. One measure of width, for example, is the number of channels in a Conv layer whereas Resolution is simply the image resolution that is being passed to a CNN.

In other words, to scale up the CNN, the depth of layers should increase 20%, the width 10% and the image resolution 15% to keep things as efficient as possible while expanding the implementation and improving the CNN accuracy. This compound scaling formula is used to scale up the EfficientNet from B0-B7

**Swish Activation**
![](https://miro.medium.com/max/1400/0*EhAHcCmGOzQUgQ0k)

ReLu works pretty well but it got a problem, it nullifies negative values and thus derivatives are zero for all negative values. There are many known alternatives to tackle this problem like leaky ReLu, Elu, Selu etc., but none of them has proven consistent.

Google Brain team suggested a newer activation that tends to work better for deeper networks than ReLU which is a Swish activation. They proved that if we replace Swish with ReLu on InceptionResNetV2, we can achieve 0.6% more accuracy on ImageNet dataset.

> Swish(x) = x * sigmoid(x)

There are other things like MBConv Block etc. If you want to know more details, you can read the articles in reference below

### Make a submission with TTA
![](https://preview.ibb.co/kH61v0/pipeline.png)
* TTA is simply to apply different transformations to test image like: rotations, flipping and translations.
* Then feed these different transformed images to the trained model and average the results to get more confident answer
