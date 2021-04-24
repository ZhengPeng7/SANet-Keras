# SANet-Keras
> An unofficial implementation of SANet for crowd counting in Keras==2.24 + TF==1.14.0.

---

## Paper:

+ Original_paper: [_Cao, X., Wang, Z., Zhao, Y., & Su, F. (2018). Scale Aggregation Network for Accurate and Efficient Crowd Counting. *The European Conference on Computer Vision (ECCV)*, 1–17_](http://openaccess.thecvf.com/content_ECCV_2018/html/Xinkun_Cao_Scale_Aggregation_Network_ECCV_2018_paper.html).

### Results now:

*On dataset ShanghaiTech B*

> Still far from the performance in the original paper(MAE 8.4)

|  MAE   |  MSE   | MAPE | Mean DM Distance |
| :----: | :----: | :--: | :--------------: |
| 12.41 | 20.33 | 0.11 |      4.942      |

### Dataset:

- **ShanghaiTech dataset**: [dropbox(backup on my personl google-drive)](https://drive.google.com/file/d/1ZT46P-NRiYAJ7mpG1TzL_ojByGJ33WCd/view?usp=sharing) or [Baidu Disk](<http://pan.baidu.com/s/1nuAYslz>).

### Env
`conda install cudatoolkit=10.0 cudnn=7.6.5`

`pip install -r requirements.txt`


### Training Parameters:

1. *Loss* = ssim_loss + L2

2. *Optimizer* = Adam(lr=1e-4)

3. *Data augmentation*: Flip horizontally.

4. *Patch*: No patch, input the whole image, output the same shape DM.

5. *Instance normalization*: _No IN layers_ at present, since network with IN layers is very hard to train and IN layers didn't show improvement to the network in my experiments.

6. ***Output Zeros***: The density map output may fade to zeros in 95%+ random initialization, I tried the initialization method in the original paper while it didn't work. In the past, when this happens, I just restarted the kernel and re-run. But now, I tried to train different modules(1-5) separately in the first several epochs to get relatively reasonable weights:

   ![structure_lite](images/network_structure_lite.JPG), and it worked out to greatly decrease the probability of the zero-output-phenomena. Any other question, welcome to contact [me](zhengpeng0108@gmail.com).

7. *Weights*: On SHB, got best weights in 292-th epoch(300 epochs in total), and here is the loss records:

   ![Loss_records](images/loss_records_B.jpg)

8. *Prediction example*:

   ![example](images/prediction_example.JPG)

### Run:

1. Download dataset;
2. Data generation: run the`generate_datasets.ipynb `.
3. Run the `main.ipynb` to train the model and do the test.

#### Abstraction:

1. **Network = encoder + decoder**, model plot is [here](./images/SANet.png):

   

   |   Network   |           encoder            |               decoder                |
   | :---------: | :--------------------------: | :----------------------------------: |
   | Composition |   scale aggregation module   |           conv2dTranspose            |
   |    Usage    | extract multi-scale features | generate high resolution density map |

2. Loss:

   Loss = ![loss_formular](https://latex.codecogs.com/gif.latex?L_{Euclidean}+\alpha_CL_C(L_C=1-\frac{1}{N}{\sum_x{SSIM(x)}},\alpha_C=0.001))

3. Normalization layer:

   + Ease the training process;
   + Reduce 'statistic shift problem'.
