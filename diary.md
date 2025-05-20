14/05/2025:  
-   Found out that batch size greatly affects GPU speed up over CPU.  
-   Currently I have a modified copy of the CIFAR10 pytorch notebook.  
    I modified it to understand how to train a image classifier on a GPU.  
-   Now that I have the bare minimum working notebook, I want to clean things up.  
    I am going to use the code I wrote in my [cnn_from_scratch](https://github.com/MauroAbidalCarrer/CNN_from_scratch) to put all the boiler plate code in seperate modules.  
    This way I can also get the live view of the charts of the metrics over epochs during training.  
    I considered using some higher level library like pyorch lightning but I figured I would first get my hands "dirty" then move on to those kind of libraries.  

17/05/2025:
-   Realized that there was an error in the `Trainer.metrics_of_dataset` function.  
    Turns out that it was actually expressing the accuracy as the mean of the sums of correct outputs per batch.  
-   Switched from  conv+relu+fc+relu to conv+relu+batchNorm+fc+relut+softmax type of model.  
    Switched from SGD (with momentum even if it's not written in the name) to Adam.
-   Came across this very [interesting notebook](https://www.kaggle.com/code/kmldas/cifar10-resnet-90-accuracy-less-than-5-min).  
    I still need to (better) understand why/how the res net work for classification tho.  
-   I read the (first, classification half of) VGG paper to understand why are conv blocks are used.  
    From what I understand, conv blocks are stacks of small (in the paper 3x3 stride 1 and pad 1 to maintain the same width/height of the input) conv layers.    
    They essentially "emulate" what a wider single conv layer would do.  
    Let's take the example of the paper of a block of 3 3x3, stride:1, pad:1 convs layer vs a 7x7 stide:2 both with input channel count = output channel count.  
    The conv block comes with these added benefeats:
    -   Less parameters the conv  block would have 3*(3\*\*2\*C\*\*2) (C squared because Cin = Cout) = 27C\*\*2 against 7\*\*2C\*\*2 = 49C\*\*2.  
        This is a 1.8x decrease in size.
    -   More non linearity since we have two more relu layers.  
    -   Better regularization/generalization as the 3*3 is a form of decomposition of the 7x7.  
        The way I understand this is that the representations that a block can learn cannot be as tied to the training data as the single 7x7 conv layer.  

    Note that the conv block and wide conv layer have the same receptieve 7x7 field.  
    I don't understand why the width/height and channel count respectively decrease and increase in between blocks instead of decrease by layer.  
    I asked chatGPT and it said that it's to preserve spatial information which sounds weird since the input will either zero or same padded...  
-   Started to read the Network in Network (NiN) article of the Dive into deep learning and found out about the lazy 2d conv in pytorch.

19/05/2025:
-   Read the dive into deep learning article of resnet.  
    Interstingly enough they say that the reason for the skip connection creation is not the vanishing/exploding gradient problem but rather non nested function groups.  
    Meaning that bigger networks can't necessarily do what smaller networks can do.  
    That sounds odd tbh.  
    One of the comments points that out and a response says that this simply what the original paper says.  
    I will try to read it tomorrow.  
-   Abondend the ida of doing the leaffliction project, instead I will "simply" train a model on a (hopefully) big data set and then try to reimplement some interpretation paper.  

20/05/2025:
-   Read the paper on the resnet architecture.  
    It was interesting.  
    It does in fact, state that the probleme it tries to solve is the "degradation problem" and not the "vanishing/exploding gradients".  
    The most interesting thing I learned is that not only does the resnet arcitecture allows for deeper models it also allows them to be a LOT thinner than VGG architecutre.  
    In the paper, they use the resnet-34 (34 layers) and VGG-19(you can guess what the 19 means...).  
    While the resnet-34 has more layers it takes 18% of the FLOPs that VGG-19 takes.  
    that is mostly because the conv layers need a lot less channels.  
    And also because there is only on fully connected layer at the end of resnet-35 against 3 of output size 4096 for VGG-19.  
-   Ok, now I can finally start reimplementing the CIFAR10 90% in 5min kaggle notebook.  
