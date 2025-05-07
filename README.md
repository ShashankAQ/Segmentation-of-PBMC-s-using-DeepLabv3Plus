# Segmentation of PBMCs's usingDeepLabv3Plus

Training DeepLabv3 on a custom dataset of stained and unstained PBMC's

## Dataset Preview
The dataset consists of grayscale .tif images of PBMC cells obtained through flow cytometry, categorized into stained and unstained samples of 30,000 samples each. It includes various types of granulocytes and monocytes, along with exceptions such as debris and possibly lymphocytes. 

![2](https://github.com/user-attachments/assets/3fa59927-41fa-4829-a698-ab4d12659761)

## Making Predictions with the Trained Model

### 1. Download the pre-trained Model

Download the custom-trained DeepLabv3+ ResNet-101 backbone from this link : [google drive](https://drive.google.com/file/d/1PLw7U5-MrrFQJ4oZue5TvnBdQ2x6IQ3a/view?usp=sharing)

### 2. Prediction

```python
git clone https://github.com/ShashankAQ/Segmentation-of-PBMC-s-using-DeepLabv3Plus.git

```
change directory

```python
cd Segmentation-of-PBMC-s-using-DeepLabv3Plus

```
Single image: 

```python
python predict.py --image_path "enter image path" --ckpt "enter model path"
```


