# DeepQA

this is a pytorch version of IQA_DeepQA_FR 

the project is coded based on the [theano version](https://github.com/jongyookim/IQA_DeepQA_FR_release) from original author jongyookim 

the paper related to this work  is

    Jongyoo Kim and Sanghoon Lee, “Deep learning of human visual sensitivity in image quality assessment framework,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 1676–1684. 


## Prerequisites

pytorch 1.0.1

python 3.6.6


## Training DeepQA

python trainer.py

## Configuration before run

dataset/live.py

BASE_PATH = '/Path/to/your/live_datase'

LIST_FILE_NAME = '/Path/to/your/project/LIVE_IQA.txt'

./trainer.py

MODEL_SAVE_PATH ='/Path/to/your/mode_save_path'

## run

python trainer

## Performance

* our

    | database    | SRCC    | PLCC     |
    | :------------- | :----------: | -----------: |
    |  live IQA | 0.981   | 0.977   |


* original version
    
    |Database |SRCC  |PLCC  |
    |---------|:----:|:----:|
    |LIVE IQA |0.981 | 0.982|
    |CSIQ     |0.961 | 0.965|
    |TID2008  |0.947 | 0.951|
    |TID2013  |0.939 | 0.947|
    
    
    


