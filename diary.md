14/05/2025:  
-   Found out that batch size greatly affects GPU speed up over CPU.  
-   Currently I have a modified copy of the CIFAR10 pytorch notebook.  
    I modified it to understand how to train a image classifier on a GPU.  
-   Now that I have the bare minimum working notebook, I want to clean things up.  
    I am going to use the code I wrote in my [cnn_from_scratch](https://github.com/MauroAbidalCarrer/CNN_from_scratch) to put all the boiler plate code in seperate modules.  
    This way I can also get the live view of the charts of the metrics over epochs during training.  
    I considered using some higher level library like pyorch lightning but I figured I would first get my hands "dirty" then move on to those kind of libraries.  