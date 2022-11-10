from PIL import Image
import random
from transformers import pipeline
import numpy as np

def color(c):
    """Return a random RGB color

    Args:
        c (int): Just the segment number

    Returns:
        np.ndarray: a random color
    """
    n = random.randint(0,16777215)
    r,g,b = n%256,(n//256)%256,(n//(2**16))%256
    return np.array([[[r,g,b]]]).astype(np.uint8)
def segmentation(input_file_name, output_file_name = None):
    """This function can segment the image and save the mask at output_file_name

    Args:
        input_file_name (str): Path to image
        output_file_name (str, optional): Path to mask. Defaults to None.
    """
    model = pipeline("image-segmentation")
    res = model(input_file_name)
    if output_file_name is None:
        output_file_name = input_file_name + 'segmented.jpg'
    shape = (*(np.asarray(res[0]['mask']).shape),3)
    final_img = np.zeros(shape).astype(np.uint8)
    for i in range(len(res)):
        final_img += np.where(np.asarray(res[i]['mask'])[:,:,None]==0,np.zeros((1,1,3)).astype(np.uint8),color(i))
    final_img = final_img.astype(np.uint8)
    resultImage = Image.fromarray(final_img)
    resultImage.save(output_file_name)
    
 

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Automated semantic segmentation.')
    parser.add_argument('-i','--input_path', type=str,
                        help='input path/dir for the images')
    parser.add_argument('-o','--output_path', type=str,
                        help='output path/dir for the images',default = None)
    parser.add_argument('-d', '--directory',action='store_false')
   
    args = parser.parse_args()
    if args.directory:
        files = os.listdir(args.input_path)
        for file in files:
            if file.split('.')[-1] in ['jpg','jpeg','png']:
                segmentation(os.path.join(args.input_path,file),os.path.join(args.output_path,file) if args.output_path is not None else os.path.join(args.output_path,'segmented_',file))
    else:
        segmentation(args.input_path,args.output_path)
        
