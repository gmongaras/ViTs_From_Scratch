from PIL import Image
import os




def main():
    # The pathname to retrieve images
    imgPath = "data-orig"
    
    # The pathname to save the images
    savePath = "data"
    
    # The width and height of the new image
    width = 256
    height = 256
    
    
    # Iterate over all folders in the image directory
    for path in os.listdir(imgPath):
        # Construct the path
        origPath = os.path.join(imgPath, path)
        newPath = os.path.join(savePath, path)
        
        # If the new path doesn't exist, create it
        if not os.path.isdir(newPath):
            os.mkdir(newPath)
            
        # The number of files in the directory
        fileCt = 0
        
        # Iterate over all files in the data directory
        for file in os.listdir(origPath):
            # Get the full file name
            fullFile_load = os.path.join(origPath, file)
            fullFile_save = os.path.join(newPath, str(fileCt) + ".png")
            
            # Open the image
            img = Image.open(fullFile_load)
            
            # Resize the image
            img = img.resize((width, height), Image.ANTIALIAS)
            
            # Save the image
            img.convert('RGB').save(fullFile_save, "png")
            
            # Increase the file count
            fileCt += 1
    
    
    
if __name__=='__main__':
    main()