##  ___**Inference**___
1. 'Weightx4' contains the author's pre-trained weights on 4x data and 'netx4.py' defines the model architecture
2. 'checkpoint' contains weights after training the model on 6x-Div2k data
3. Run **get_Y.py** to simultaneously perform bicubic-upsampling on LR-RGB frames and extract HR-Y frames
4. Run **DWSRx4.py** to infer SR-Y frames from HR-Y frames
5. Run **YtoRGB.py** to combine the SR-Y frames with LR-RGB frames, giving SR-RGB frames

##  ___**Training**___
1. 'Weightx4' contains the author's pre-trained weights on 4x data and 'netx4.py' defines the model architecture

2. Create the following directories inside the current directory:</br> 
            **'train_checkpoints'** to save progress during training</br>
            **'train_costs'** to save epoch-wise costs in csv files</br> 
            **'Train_cropped_set'** with sub-directories: **'Train_cropped_set/HR_Y_crops'** and **'Train_cropped_set/GT_Y_crops'**</br>  
            
3. Run **generate_crops.py** with a window size of 40 and overlap of 10, to generate crops out of your HR-Y and GT-Y channels. The directory into which the crops are saved must be one of the 2 subdirectories created within 'Train_cropped_set', as described above. To obtain these channels, use **get_Y.py** with appropriate options
4. Modify hyperparameters in **training_script.py** and run it to start training
