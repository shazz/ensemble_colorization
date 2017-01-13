# Colornet / ensemble_colorization

## Sources
- Fork from https://github.com/shazz/ensemble_colorization
- Fork from https://github.com/pavelgonchar/colornet

## Usage

1. Download the VGG-16 pretrained model

  ```
  cd /tmp
  curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
  sudo apt-get install git-lfs
  git lfs install
  cd -
  cd vgg/tensorflow-vgg16
  git-lfs pull
```

2. Setup the dataset structure

  * Call `./setup.sh` to create the directories

3. Download, extract and process an image data set 

  * Copy the JPG images in `dataset/original`

  * Process the images, call `./process_dataset.sh`

4. Train the 4 CNN Models

  * Modify `train.sh` to configure EPOCH, SUMMARY exports and MODEL checkpoint 
  
  * Call `./train.sh`

  * Check the results in `dataset/summary`

5. Colorize Grayscale images

  * Copy your images (grayscales or not) in `dataset/test`

  * Call `./test.sh`

