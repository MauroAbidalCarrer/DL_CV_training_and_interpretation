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


21/05/2025:
-   Read the web version of [DNNs always grok and here is why](https://imtiazhumayun.github.io/grokking/).  
    It was very interesting, hopefully I can reimplement the Local complexity measure.  
    Ideally I could use it as input for some sort of training policy.  
-   I finished the first reimplementation of the kaggle notebook but the model is super slow.  
    Then the remote machine crashed.  
    So hopefully the issue comes from the machine and not the code.  
    I ran the code on another machine and it still is super slow: one hour for two epochs...  
    Which is odd since the notebook is called "CIFAR10 90% in 5 mins".  
    Now I actually hope that there is something wrong with my code.  
    If not, it means that I will have to pay for a better, more expensive, GPU.  
    I fixed the kaggle notebook by replacing the code cell that downloaded CIFAR10 from fast.ai by a cell that downloads it using torchvision.datasets.CIFAR10.  
    Then I ran it on kaggle using a P100 GPU.  
    It trained the model in 2 mins(wtf?!!).  
    I downloaded the notebook on the schools computer and addded it to the repos.  
    I pull the notebook from the repos onto a vastai instance with a 4090.  
    I runs faster than my reimplementation: 2 epochs in 13 minutes.  
    But that's nothing compared to the 8 epochs in two minutes.  
    So either I switch my workflow from vastai to kaggle OR I search for a simple opti trick.  
    I looked for FFT 2D conv but I couldn't find an pytorch API reference for it.  
    Also the forums seem to suggest that the benefits of using FFT for convolution emerge when using much larger filters amd inputs.  
    I ran the same notebook on a A100 vastai instance and the 8 epochs training took one minutes.  
    Damn...
    I tested the notebook on a Tesla V100 and it ran in 1min16s but it costs 28¢/h instead of the ~1$/h for the A100.  
    So I'll defenetly be using that going forward.  
-   Tommorow I will try to understand why my trainer implementation is slower than the kaggle notebooks implementation.  
    And Then I will have to add in all the other features like learning rate scheduling.  

22/05/2025:
-	I updated the setup_linux_machine repo to increase productivity.  
	I added an aliases.zsh file that contains all the aliases I already had + `p` and `amp` git aliases.  
	I might also use a repo I found that manages the .ssh/config file automatically.  
	It looks like I will do anything to not work on the "main quest" of this repo lol.  
    I tried the vastai cli and the vastai-ssh-config package I found online but I coulnd't make them work so gave up on that.  
-   Now the "real" work of the day begins, I am going to try to find out why my code is slower than the original one.  
    Turns out the model was simply not on the GPU I just had forgot to add a .cuda() call to its declaration.  