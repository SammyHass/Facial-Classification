
# Face Authentication

This project uses an altered version of the [VGG Convolutional Architecture](http://www.robots.ox.ac.uk/~vgg/research/very_deep/). I decided to use this as it is reasonably light weight to train and gives fairly good results. This code is for a paper I am writing - 'Efficient Facial Authentication with Convolutional Neural Networks', the paper will be available on my GitHub in December.
## Run in anaconda env (python 3.6)
```bash
pip install -r requirements.txt
python collection.py
python aug.py
python train_test_split.py
python nn.py
python test.py
```
**Hyper Parameters Information**:
- Learing Rate: 0.01
- Decay: 1E-6
- Momentum: 0.9
- Batch Size: 32
- Training Epochs: 10
- Loss Function: MSE
- Optimizer: SGD 

**Architecture Summary**:
1. Input 224, 224, 3
2. 3x3 Conv 32 (ReLU)
3. 3x3 Conv 32 (ReLU)
4. Max Pool 2x2
5. 3x3 Conv 64 (ReLU)
6. 3x3 Conv 64 (ReLU)
7. Max Pooling 2x2 
8. Dropout 0.25
9. 3x3 Conv 128 (ReLU)
10. 3x3 Conv 128 (ReLU)
9. Flatten Layers
10. Fully Connected 512 (ReLU)
11. Dropout 0.5
12. Output 2 (softmax) 
