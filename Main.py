"""
Main file for compiling everything

training, inference everything will be run from here 

"""

# importing our personal file 
import Data_things
import model

import matplotlib.pyplot as plt



def main():
    """
    Can call function from now
    """
    #get images from  current folder
    path = "."
    dataset = Data_things.getdataset(path)
    #wrap datset into batchsz
    batchsz = 8
    datalaoder = Data_things.getdataloader(batchsz)
    # make instance of model class
    m = model.Autoenc()
    #train instance m
    model.train(m) # shoul have the training loop 

    # check prfromance on image 4
    img_number = 4
    model.evalu(m,img_number)




    
# This will call main function only when this is the main file
# when you import a python script it is not the main file
if __name__ == "__main__":
    main()