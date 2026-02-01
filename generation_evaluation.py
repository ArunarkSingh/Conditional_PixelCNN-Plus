'''
This code is used to evaluate the FID score of the generated images.
You should at least guarantee this code can run without any error on test set.
And whether this code can run is the most important factor for grading.
We provide the remaining code,  you can't modify other code, all you should do are:
1. Modify the sample function to get the generated images from the model and ensure the generated images are saved to the gen_data_dir(line 20-33)
2. Modify how you call your sample function(line 50-55)

REQUIREMENTS:
- You should save the generated images to the gen_data_dir, which is fixed as './samples'
- If you directly run this code, it should generate images and calculate the FID score, you should follow the same format as the demonstration, there should be 100 images in 4 classes, each class has 25 images
'''
from pytorch_fid.fid_score import calculate_fid_given_paths
from utils import *
from model import * 
from dataset import *
import os
import torch
import argparse
#TODO: Begin of your code
# This is a demonstration of how to call the sample function, feel free to modify it
# You should modify this sample function to get the generated images from your model
# You should save the generated images to the gen_data_dir, which is fixed as 'samples'

# Used ChatGPT to understand the logic behind the generation evaluation function.

sample_op = lambda x : sample_from_discretized_mix_logistic(x, 5)
def my_sample(model, gen_data_dir, sample_batch_size = 25, obs = (3,32,32), sample_op = sample_op):
    # Determine device from model parameters
    device = next(model.parameters()).device
    
    for label_name, label in my_bidict.items():
        print(f"Label: {label_name}")

        # Map the integer label to a normalized float between 0 and 1.
        cond_value = float(label) / (len(my_bidict) - 1) if len(my_bidict) > 1 else 0.0
        # Create a condition tensor of shape (sample_batch_size, 1, obs[1], obs[2])
        condition = torch.full((sample_batch_size, 1, obs[1], obs[2]), cond_value, device=device)
        # Inject the condition by overriding the model's init_padding with the condition tensor.
        #model.init_padding = condition

        #generate images for each label, each label has 25 images
        sample_t = sample(model, sample_batch_size, obs, sample_op)
        sample_t = rescaling_inv(sample_t)
        save_images(sample_t, os.path.join(gen_data_dir), label=label)
    pass
# End of your code

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--ref_data_dir', type=str,
                        default="data/test", help='Location for the dataset')
    
    args = parser.parse_args()
    
    ref_data_dir = args.ref_data_dir
    gen_data_dir = os.path.join(os.path.dirname(__file__), "samples")
    BATCH_SIZE=128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(gen_data_dir):
        os.makedirs(gen_data_dir)

    #TODO: Begin of your code
    #Load your model and generate images in the gen_data_dir, feel free to modify the model

    #model = PixelCNN(nr_resnet=1, nr_filters=40, input_channels=3, nr_logistic_mix=5)
    #model = model.to(device)
    
    model = PixelCNN(nr_resnet=1, nr_filters=160, input_channels=3, nr_logistic_mix=5)
    model = model.to(device)
    # Define the checkpoint path; adjust if your checkpoint is stored elsewhere.
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'models', 'conditional_pixelcnn.pth')
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print('Trained model loaded from:', checkpoint_path)
    else:
        raise FileNotFoundError(f"Trained model checkpoint not found at {checkpoint_path}")



    model = model.eval()
    #End of your code
    
    my_sample(model=model, gen_data_dir=gen_data_dir)
    
    paths = [gen_data_dir, ref_data_dir]
    print("#generated images: {:d}, #reference images: {:d}".format(
        len(os.listdir(gen_data_dir)), len(os.listdir(ref_data_dir))))

    try:
        fid_score = calculate_fid_given_paths(paths, BATCH_SIZE, device, dims=192)
        print("Dimension {:d} works! fid score: {}".format(192, fid_score, gen_data_dir))
    except:
        print("Dimension {:d} fails!".format(192))
        
    print("Average fid score: {}".format(fid_score))
