Here's a good README for your GitHub project, incorporating best practices and tailored to your project's specifics.

-----

# Image Colorization via cGANs Using ResNet-18 Backed U-Net Architecture

## Project Overview

This project explores the fascinating problem of image colorization, transforming grayscale images into vibrant colored ones. We tackle this challenge using a conditional Generative Adversarial Network (cGAN) architecture, leveraging the power of deep learning to synthesize realistic color information. Our generator is built upon a U-Net architecture, enhanced with a pre-trained ResNet-18 backbone for robust feature extraction, while the discriminator employs a PatchGAN approach to assess the realism of generated colored images.

## Features

  * **Conditional GAN (cGAN) Framework:** Utilizes a cGAN setup for stable and effective training, conditioning the generation process on the input grayscale image.
  * **ResNet-18 Backed U-Net Generator:** Employs a U-Net architecture with a ResNet-18 encoder for powerful feature extraction, followed by a decoder for upsampling and color channel prediction.
  * **PatchGAN Discriminator:** Incorporates a PatchGAN discriminator that classifies patches of the image as real or fake, promoting fine-grained realism in the generated outputs.
  * **Perceptual Loss (Optional but Recommended):** (If you've implemented this, add it. If not, consider it\!) Integrates a perceptual loss component using a pre-trained VGG network to guide the generator towards more visually pleasing and perceptually accurate colorizations.
  * **Diverse Dataset Support:** Designed to be adaptable to various image datasets (e.g., COCO sample).
  * **Evaluation Metrics:** Includes scripts for evaluating performance using metrics relevant to image quality (e.g., PSNR).

## Why ResNet-18 and U-Net?

  * **ResNet-18:** A relatively lightweight yet powerful convolutional neural network, pre-trained on ImageNet, provides excellent feature extraction capabilities. Its residual connections help mitigate vanishing gradients, allowing for deeper and more effective learning.
  * **U-Net:** Renowned for image-to-image translation tasks, the U-Net's symmetric encoder-decoder structure with skip connections allows the decoder to access fine-grained details from the encoder, crucial for high-quality image reconstruction. Combining it with ResNet-18 enhances its ability to learn complex mappings from grayscale to color.

## Getting Started

### Prerequisites

  * Python 3.x
  * PyTorch
  * Torchvision
  * NumPy
  * Pillow
  * Matplotlib
  * Tqdm (for progress bars)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Ritesh078bct/IMAGECOLORIZATION_VIA_CGANS_USING_RESNET-18_BACKED_U-NETARCHITECTURE.git
    cd IMAGECOLORIZATION_VIA_CGANS_USING_RESNET-18_BACKED_U-NETARCHITECTURE
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

    (You'll need to create a `requirements.txt` file by running `pip freeze > requirements.txt` after installing all dependencies.)

### Dataset Preparation

1.  **Download your desired dataset.** For example, you can use a subset of ImageNet or COCO.
2.  **Organize your dataset:** The `data_loader.py` script expects images to be in a certain structure. Typically, this means a directory containing all your color images. The script will handle converting them to grayscale as needed.
    ```
    data/
    ├── train/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── val/
        ├── image_val1.jpg
        ├── image_val2.jpg
        └── ...
    ```
    (Adjust `data_loader.py` if your structure differs.)

## Usage

### Training the Model

To train the colorization model, run the `train.py` script:

```bash
python train.py --epochs 100 --batch_size 16 --lr_G 0.0002 --lr_D 0.0002 --dataset_path ./data --output_dir ./checkpoints
```

**Key Arguments:**

  * `--epochs`: Number of training epochs.
  * `--batch_size`: Batch size for training.
  * `--lr_G`: Learning rate for the generator.
  * `--lr_D`: Learning rate for the discriminator.
  * `--dataset_path`: Path to your dataset directory.
  * `--output_dir`: Directory to save model checkpoints and generated samples.
  * `--gpu_id`: (Optional) Specify GPU ID if you have multiple GPUs (e.g., `--gpu_id 0`).

### Generating Colorized Images

Once you have a trained model checkpoint, you can generate colorized images:

```bash
python generate.py --checkpoint_path ./checkpoints/generator_epoch_100.pth --input_image_path ./test_images/grayscale_input.jpg --output_image_path ./output_images/colorized_output.jpg
```

**Key Arguments:**

  * `--checkpoint_path`: Path to the trained generator checkpoint.
  * `--input_image_path`: Path to the grayscale image you want to colorize.
  * `--output_image_path`: Path to save the colorized output image.

### Evaluation

To evaluate the performance of your trained model on a validation set:

```bash
python evaluate.py --checkpoint_path ./checkpoints/generator_epoch_100.pth --dataset_path ./data/val --output_dir ./evaluation_results
```

This script will calculate metrics like PSNR and SSIM and save example colorized images from the validation set.

## Project Structure

```
.
├── data/                       # Dataset directory (e.g., train/, val/)
├── models/
│   ├── generator.py            # U-Net with ResNet-18 backbone
│   └── discriminator.py        # PatchGAN discriminator
├── utils/
│   ├── data_loader.py          # Custom dataset and data loading utilities
│   └── losses.py               # GAN loss, L1/L2 loss, (optional) perceptual loss
├── checkpoints/                # Directory to save trained models
├── output_images/              # Directory to save generated images
├── train.py                    # Main training script
├── generate.py                 # Script for colorizing single images
├── evaluate.py                 # Script for model evaluation
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── (Optional) notebooks/       # Jupyter notebooks for exploration/visualization
```

## Results
![output_image](https://github.com/user-attachments/assets/2036b4ae-511b-4b43-b8d4-9172044161b3)


### Quantitative Results

| Metric | Value |
| :----- | :---- |
| PSNR   | 2X.XX |


## Contributing

Contributions are welcome\! If you have any suggestions, bug reports, or want to contribute code, please feel free to open an issue or submit a pull request.

## Contributors
*  MD. Astafar Alam (PUR078BCT068)
*  Reshmi Jha (PUR078BCT066)
*  Ritesh Sahani (PUR078BCT068)
*  Subash Kr. Yadav (PUR078BCT087)

## Acknowledgments

 We express our sincere gratitude to Mr. Pukar Karki, Deputy Head of the Department
of Electronics and Computer Engineering, for providing us with the opportunity and re-
sources to undertake this minor project. We are deeply thankful to our project supervi-
sor, Assistant Prof. Binaylal Shrestha, for his invaluable guidance, constant encourage-
ment, and insightful feedback throughout the development of this project on ”IMAGE
COLORIZATION” using conditional Generative Adversarial Networks (cGANs).
We also extend our appreciation to all the faculty members of the Electronics and Com-
puter Department for their support and technical expertise, which greatly enriched our
learning experience. Special thanks to our classmates and friends for their constructive
suggestions and motivation. Finally, we acknowledge the unwavering support of our
families, whose encouragement kept us motivated.
This project would not have been possible without the collective contributions of ev-
eryone involved.


-----
