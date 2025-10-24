# Gray to RGB

A PyTorch project for converting grayscale images to RGB using a deep convolutional neural network based on a U-Net architecture.
The model and weights are available on Hugging Face:  
🔗 https://huggingface.co/r4mt1n/gray2rgb/tree/main

## Training Summary

- **Average PSNR:** 23.88 dB
- **Average SSIM:** 0.9112
- **Model Size:** 300 MB
- **Epochs:** 9
- **Input Size:** 64×64

## Dataset

This project uses a subset of the COCO dataset 2017 for training, validation, and testing.

```
data/
└── coco/
    ├── train/
    ├── val/
    └── test/
```

- **Name:** COCO (Common Objects in Context)
- **Source:** https://cocodataset.org

## Usage

Clone the repository and install dependencies:

```bash
git clone https://github.com/r4m-t1n/gray2rgb.git
cd gray2rgb
pip install -r requirements.txt
```

To train or test the model, open the notebooks:

- `notebooks/gray-to-rgb-pipeline.ipynb` – Training pipeline
- `notebooks/test-model.ipynb` – Model evaluation and visualization

## License

This project is released under the **MIT License**.