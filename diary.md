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
    Nevertheless, my training still runs two times slower than the original one (2m15s instead 1min15s).  
    Even tho this is not a very big diff for me right now, it's worth investigating into it to learn from it now.  
    It also turned out to be a simple reason: 
    I would recompute the outputs of the model with `no_grad` on the training set where the kaggle notebook uses the outputs of the training loop.  
    According to chatGPT, this is the conventional way of doing it.  
    I set out to update the Trainer implementation... and it took me an embarassing amount of time(hours).  
    I ended up with a convoluted solution, which I actually threw away for a simpler uglier solution but at least it works with minimal modifs.  
    Now I can start adding the "fancy" features.  
    Namely, lr scheduling, weight decay and gradient clipping.  
    Done.

23/05/2025:
-   I read this [toturial/explanation web page](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html) on how to find the right learning rate.  
    Very intereseting.
    KTs:
    - To find the learnging rate:
        - Start training with a small learning rate.
        - At each iteration, register the smoothed exponentially weighed averaged loss and muiltiply it by some hyper parameter for the next iteration.
        - The loss will decrease then increase.
        - Once it increases to four fold that of the minimum registed loss stop the "learning rate search"(that expression comes from me).
        - The web page states that the learnig rate with the lowest loss divided by 10 should be taken.  
          The reason for taking a learning rate smaller than the one with the lowest loss instead of taking the later is because of the exponentially weighed averaging: it makes the loss rise later.  
          I thought it was kinda weird to take the learning rate with the lowest loss since it is preceeded by other training steps that have already deacreased the loss in the previous iterations.  
          I asked why not train the model from start for each learning rate to chatGPT.  
          It said that while it would be accurate, it would be a lot more compute intensive and that searching for the learning rate in a single training run is good enough.  
          It also said that sometimes the learning rate with the highest loss difference (compared to the previous iteration) is chosen.  
          That makes a lot more sense.  
-   Then I read this [toturial/explanation web page](https://sgugger.github.io/the-1cycle-policy.html#the-1cycle-policy) of the same author on one cycle policy.  
    This is a learning rate scheduler.  
    In fact this is the page I wanted to read in the first place. However it was refering to the previous page which looked important.  
    I didn't learn THAT much mostly because there wasn't much content on the page.  
    KTs:
    - The scheduling is comprised of three parts:
        - A linear ascent of the learning rate from a low value (usually 10x lower that the value found by the lr finder) to the "normal" value, the one find by the learning rate finder.    
          This is know as the warm-up phase and this is what allows us to take a higher learning rate than if we didn't go through this warm up phase.   
          The ascent takes ~half ot the training.  
        - A slightly shorter long linear descent from the found learning rate back to the low value.
        - Yet another linear, very short descent to ~100x less than the starting value (so ~1000x less that the one found by the LR finder).  
    I don't really understand what the warm up is so I will have to into that another day.  
-   I read a reddit blog post that asked why the learning rate was used and why some papers say that it is usefull while others say that it is not.  
    The only response that seemed to make sense is that it prevents the Adam optimizer from accumulating early mostly random gradients in its momentum.  


25/05/2025:
-   I watched this [video] (https://www.youtube.com/watch?v=KOKvSQQJYy4&ab_channel=Tunadorable) on the warmup of learning rate.  
    KTs:
    - New terms:
        - loss sharpness: How much the learning rate changes for a given change in parameters.
        - trainging/loss catapults: The oscillations of the loss during training.
    - There are two reasons for the success of learning rate: 
        1.  The one discussed in the reddit post, it prevents Adam from accumulating noisy initial gradients in its momentum.  
        1.  The loss landscape is at first very sharp.  
            -   If we use a normal decaying learning rate policy/schedule we will essentially find the closest crevasse in the landscape.  
                This is not ideal since crevasses in sharp landscapes are usually overfit regions.
            -   If we first warmup the learning rate, the model will escape this sharp region and arrive at a flatter region.  
                This is great for two reasons:
                -   Flat regions are usually regions of generalization.
                -   They allow us to use higher learning rate.
        > Warning: I passed the "sharp to flat LR warm up theory" to chatGPT to confirm and it denied it once out of three times...
    - The warmup is not always necessary since the loss landscape is not always sharp it's mostly used on large models or for trainings with large batch sizes.  
    KT from chatGPT:  
    - The warm-up also helps the batch/layernorm layers to initialize their parameters similarly to the way warm up helps to initialize the momentum of Adam.  
-   With all of that out of the way, I can start looking into the interpretation papers/blog posts.  
-   I started reading the [Deep Networks Always Grok and Here is Why](https://arxiv.org/pdf/2402.15555) paper.  
    It's pretty verbose in it's introduction.  
    I just got started, but I learned a few new mathematical terms.

26/05/2025:
-   I finished reading the paper.  
    KTs:
    -   Terms:
        -   Delayed generalization: aka Grokking  
        -   Adversirial robustness: Output resistance/invariance to perturbations in the input.  
        -   Delayed robustness: Grokkiong/generalization on adversarial samples (so this seems like Gorkking premium basically)  
        -   Piecewise function: A function that is partitioned into several intervals on which the function may be defined differently.
        -   circuit: A subgraph of the network(a combination of activated neurons) (This actually comes from another paper but it's reused here.).  
            Such circuit can be simplified, when using a continuous piecewise linear activation functions sucha as relu, to a single affine transformation.  
        -   Partition/region: Shape in the input space of a circuit.  
        -   Local Complexity: A measure of the number of different output partition/regions/splines in neighborhood in the input.  
            I will explain it more clearly later.  
        -   circular Matrix: A square matrix in which all rows are composed of the same elements.  
            Each row is rotated one element to the right relative to the preceding row.
        -   Training interpolation: Point at which the model as reached very low training error/loss (usually when it has overfitted?).  
    -   All DNNs architectures are a form of MLP, that is true for CNNs and Transformers.     
        I assume it's also true for other less well known architectures but this doesn't really matter anyway.    
        In fact, a conv layer is a specifc case of a simble linear layer, it's a circular Matrix.  
    -   All DNNs are continuous piecewise linear functions/continuous piecewise affine splines.  
        Yes, the paper uses different expressions for (what seems to be) the same thing.  
    -   The density of these pieces/partition/regions in a subregion of the input space can be expressed as local complexity(LC).  
        The LC of a point (like a training/test/validation sample) expresses:
        -   how many regions there are close to it
        -   how "non linear" the output is in that input region (ssuming that the regions have diffrent linear functions).  
        -   how complex the output is in that input region
        It is presented as a new progress measure on an equal foot with loss, accuracy ect.
        To measure the LC of a layer in the neighbood of a point:
        1.  P points are projected on a sphere of radius r centerd on the point.
            This composes a cross-polytopal.  
        1.  We pass the points/vertices through the linear layer.  
        1.  Vertices from differnt sides of the neurons hyperplanes will have different signs.  
        1.  Vertices with different signs than the one they are connected to by edges of the cross-polytopal increase the LC measure.  
        1.  Then we pass the points through the layers up to the next linear layer.   
        1.  Repeat until we reach the end of the model.  
            I believe that the originally convex Polytope will potentially loose over time its convexity.  
        It's still unclear to me tho by exactly how much does the LC increases by sign changes.  
    -   The training or test LC is the LC of the training and testing points respectively.  
        When they increase, it means that the number of linear regions/complexity/non-linearity around their respective points increases.
        Simillarly, when they decrease, it means that the number of linear regions/complexity/non-linearity around their respective points decreases.
    -   During training the training LC follows this dynamic: 
        1.  A descent, this one "is subject to the network parameterization as well as initialization"(extraact from the paper) and may not always occur.  
        1.  An ubiquitous(always happens) ascent that lasts until trainging interpolation.  
            ```
            The training LC may be higher for training points than for test points.
            Indicating an accumulation of non-linearities around training data compared to test data.
            ```
            Sligthly modified extract from the paper.
        1.  A second (if there was a first) descent of training AND test points.  
            The linear regions migrate away from training points, this is kinda obvious since training LC decreases.  
            During this phase, the authors have discovered  a less obvious phenomenom through spline cam:  
            `The regions tend to migrate away from the training+testing points and toward the decision boundry.`  
            **This is when the grokking happens.**  
            Interstingly enough, this happens before generalization can be recognized through train/test loss/accuracy.  
    -   According to section four here is how grokking is affected to hyperparameters:
        -   Parameterization(number of parameters): 
            Increasing parametrization whether through widdening or deepning,
            "hastens region migration, therefore makes grokking happen earlier".
        -   Weight decay: 
            ```
            Weight decay does not seem to have a monotonic behavior as it can (added the can for clarity) both
            delays and hastens region migration, based on the amount of weight decay.
            ```
        -   Batch Normalization: Always blocks grokking (I tried reading the Appendix to understand why but I got lazy).
        -   Activation functions: Was only tested on Relu And Gelu and both lead to grokking.  
        -   Dataset size: 
            Scalling good generizable training data hastens grokking.  
            On the other hand, scalling training set tha contains data that needs to be morized, slows down grokking.  
            This is demostrated in the paper by training an MLP on MNIST and randomizing a defined fraction of the training sample labels.  
    -   The paper also makes a connection between spline and circuits theory:  
        -   Each partition is a circuit.  
        -   Moving from one adjacent partition to the other corresponds to turning on or off a single neuron.  

27/05/2025:
-   Now I will try to use first the [spline cam code](https://github.com/AhmedImtiazPrio/splinecam) and then the [local complexity code](https://github.com/AhmedImtiazPrio/grok-adversarial/).  