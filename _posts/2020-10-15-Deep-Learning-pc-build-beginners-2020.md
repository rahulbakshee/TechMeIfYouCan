![nn]({{ '/images/2020-10-15-neural_net.png' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*[Unsplash credits](https://unsplash.com/collections/7516247/network)*

{: class="table-of-content"}
* TOC
{:toc}

---
Hello World !!! 

The field of Data Science or Artificial Intelligence is challanging because of it's `Disruptive` nature. Every year hundreds of GOOD papers/techniques are open sourced which claim to be state-of-the-art and it's very hard to keep up with all of them. I find a combination of theory and practice to be very effective for learning new things. **`Quickly skim through the paper and jump on to the code to feel how it works. And of course this is an iterative process.`**

## What is Deep Learning
Deep Learning is a subset of Machine Learning where you deal with complex problems like Image & Text using different Neaural Network Architectures. So, there will be lots of neurons in lots of hidden layers (millions and maybe billions of parameteres) and the job would be to get the optimal weights for these neurons.

## Why you need high compute power in Deep Learning
One needs enough compute power for faster prototyping and quicker model training. Present day CPUS are not enough for all of this. The datasets are getting larger and processing speed matters. We need to handle huge amount of data and do multiple computations simultaneously. GPUS/TPUs to the rescue...

## Cloud vs Local
`Storage` and `Compute` are the two most important things doing any kind of Data Analysis. Now you must be wodering about which compute option to choose between cloud or having your own pc. Each option has its own prose and cons. But as a `beginner` I would suggest you to first go to [Google Colab](https://colab.research.google.com/) and give it a try. Although it has some limitations on usage but it won't matter if you are a beginner.
And you will get **`free CPU/GPU/TPU`**. 

> Yesssss you read it right !!! :blush:


The **`cloud`** option is good in a sense that you don't have to worry about any setup or assembling hardwares. Just spin up some instances and you are good to go. You only have to worry about shutting them down when not in use.

Having a **`local setup`** is what I preferred because `I wanted to feel the power at my finger tips, hear the fans running at full speed at extreme loads. I wanted to have the pleasure of unboxing and assembling all parts to make one. In longer run it would be cheaper than the Cloud option. Also, my (very)old laptop was broken and I needed something new anyhow.`

I would say that ROI is better in option#1 when you don't have serious intentions to do DL work and you might leave the practice in near future. But if you are serious and will anyhow use those GPU cores to fullest, the cost of pc parts will pay off in 6 months to 1 year.

## Things to keep in mind when selecting parts
The two things I think are important for part selection
  - `Budget` could be anything for example ranging from $ 500 to $ 10,000 or more. It's up to you how much you can afford at the moment. Some people believe in spend lessnow and upgrade later whereas some people believe in wait for money to get accumulated and buy best parts in one shot.
  - `Purpose of the build` could be any of the `Deep Learning`, `Gaming`, `Editing` or a combination of them. Where you are going to keep this build? Whether you will be physically using or SSHing the build? How often you will be using it?

## My pc parts list

![all_in_one]({{ '/images/2020-10-15-all1.png' | relative_url }})

> [my pcpartpicker list](https://pcpartpicker.com/list/GVNHBc)

There are lots of brands and lots of options/configurations to choose from for a particular part. Given my little knowledge of hardware and abundant knowledge freely available on internt it was easy to get lost. But with the help of lots of articles, videos, forums and folks on the interent I was able to pull this off. 

Due to Covid-19, the availability of parts was big issue and if available in my country the prices were skyrocketing. I knew I had to spend more than I anticipated. But still I went ahead as I urgently needed something to stay in practice.

My requirements were simple. ***`performance`***. Whether it has ***`RGB`*** or not I didn't care. Below are the parts I chose, but I encourage you to do your own research and then make purchases. 


### **`GPU`**

A Graphical Processing Unit should be the first and one of the most important things to be considered when buiilding a Deep Learning rig. It has many cores to handle the high compute(matrix multiplication). AMD and NVIDIA are the two top brands in race but you might want to go with Nvidia for CUDA and cuDNN. Also, the GPU RAM should be as high as possible to have larger batches to train. When I built my pc, the Nvidia RTX 30XX series GPUs were not launched and there was no point in buying 10XX or 20XX series GPUs(costly but less performing). So I went ahead with the **`Nvidia GTX 1660 SUPER`** for hvaing the best price to perfomance ratio at that time and keeping in mind to upgrade to a 30XX series RTX graphhics card in early 2021.

### **`CPU`**

A Central Processing Unit is the heart of the build. AMD and Intel are the two most preferred brands which provide very powerful CPUs. Intel is preffered over AMD due to MKL (designed by Intel) but still I fell in love with the price to performance(cores) ratio of AMD and bought **`AMD Ryzen 3900X`**  for its 12 core & 24 threads. 

### **`CPU Cooler`**
A high performing CPU means high power consumption and high heat generated. Therefore a good quality cooler is needed to dissipate the heat generated by the 3900x. Although it comes with a wrath prism cooler, I went ahead with **`Noctua NH-D15 Chromax Black`** which is one of the best aftermarket cpu coolers.

### **`Motherboard`**

Another tough choice for me. A motherboard should be selected based on compatibility with various other parts. After much research and comparisons I went ahead with **`MSI MEG X570 Unify`** for its good heat sink, price and pure black design. 

### **`RAM`**

The more RAM the better. It is recommended that you should have RAM at least double the size of your GPU memory to avoid bottlenecking. My motherboard could support 128GB in 4 DIMM slots so I went ahead and bought 4 sticks of  **`Corsair Vengeance LPX 32GB 3000mhz c16`**  

### **`Case`**

A case is very subjective thing. Case keeps all the parts safe inside itself. That is the first thing everyone lays their eyes on. I wanted a `high airflow, decent looking` case, so I went ahead with **`Cooler Master H500 ARGB`** because the non-ARGB version was not available at that time. Whenever I upgrade to Nvidia 30XX series cards, I would also be adding **`2 Noctua NF-A14 140mm case fans`** to cool down the beast.

### **`PSU`**

One should never compromise on Power Supply Unit. It provides uninturrupted fuel to the machine to run smoothly. A general recommendation is to have 10% more watts on top of total required wattage. A 750W or 850W would have been sufficient for my build, but due to unavalability of these I had to go ahead with **`Corsair RM1000X`** which is an overkill for build right now but good for future upgrades-Nvidia Ampere 30XX series cards.


### **`SSD`**

These are the main storage for you operating system, datasets and other files. Given that my motherbaord support upto 3 M.2 NVME drives, I went ahead and bought **`Kingston A2000 1TB M.2 NVME`** and **`Western Digital Blue SN550 1TB M.2 NVME`**. These days NVME is recommended for fast boot-ups or quick data transfers. 

### **`HDD`**

I did not buy any Hard Disc Drive because first they are slow compared to SSDs and second there is a chance of failure because they are mechanical in nature. I don't need that much space right now. Storage can be upgraded in future to a **`8TB 7200RPM HDD`**.

### **`Monitor`**

A monitor has many types IPS/VA/TN and usages Coding/Gaming/Editing. From programming and coding perspective, I have heard it is recommended IPS monitor having a good PPI-Pixel per Inch. Some prefer large screen monitors for better experience while some prefer multi medium-size monitor setup for productivity. I bought **`Dell UltraSharp U2415 24" monitor`** and planning to add a **`BenQ 27" GW2780 monitor`** in near future.

### **`WebCam`**

If you are learning Computer Vision you must have an external camera or a built-in one in your laptop. This is probably an overkill but I bought **`Logitech C930e`** not just for CV but also for live meetings & streams.


### **`Keyboard`**

I did a lot of research for Mechanical vs Membrane. After years of using membrane keyboards I finally went with a Mechanical one and got **`Cosmic Byte CB-GK-16 Firefly RGB`**

### **`Mouse`**

I got a cheap used **`Dell mouse`** from my colleague because she was going to buy a new one. So I saved few bucks here. Anyway, I don't use mouse that much.


### **`Headset`**

A headset is not necessity but a good to have. Listenig to music while coding keeps me focused on the game. Given the budget I went ahead with **`HyperX Cloud Stinger`**.


### **`USB Drive`**

After all the parts are in place you need to install Operating System(s). For this you would be needing a USB Drive. Many people prefer a dual boot system having Linux(ML/DL stuff) and Windows(all other stuff). I bought a **`Sandisc Cruzer Blade 32GB USB 2.0`**


## What to do after you have built a pc

- install OS
- install ML/DL lib
- Start practicing
- start writing about it
- make video tutorials 
- enjoy the journey


## References
gamersnexus, JayzTwoCents, Linus Tech Tips, Bitwit, Paul's Hardware and countless others

[pcpartpicker](https://pcpartpicker.com/) 

[Tim Dettmers](https://timdettmers.com/)

[r/buildapc](https://www.reddit.com/r/buildapc)

[r/IndianGaming](https://www.reddit.com/r/IndianGaming/)



