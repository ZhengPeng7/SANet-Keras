# SANet-Keras
> Unofficial implementation of SANet for crowd counting in Keras.

---

## Paper:

+ Original_paper: [_Cao, X., Wang, Z., Zhao, Y., & Su, F. (2018). Scale Aggregation Network for Accurate and Efficient Crowd Counting. *The European Conference on Computer Vision (ECCV)*, 1–17_](http://openaccess.thecvf.com/content_ECCV_2018/html/Xinkun_Cao_Scale_Aggregation_Network_ECCV_2018_paper.html).

### Results now:

*On dataset ShanghaiTech B*

> Still far from the performance in the original paper(MAE 8.6)

|  MAE  |  MSE  | MAPE  | Mean DM Distance |
| :---: | :---: | :---: | :--------------: |
| 14.32 | 22.88 | 12.22 |      110.62      |

### Dataset:

- **ShanghaiTech dataset**: [dropbox](<https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=0>) or [Baidu Disk](<http://pan.baidu.com/s/1nuAYslz>).

### Training Parameters:

1. Loss = ssim_loss + L2

2. Optimizer = Adam(lr=1e-4)

3. Data augmentation: Flip horizontally.

4. Patch: No patch, input the whole image, output the same shape DM.

5. Instance normalization: No IN at present.

6. Pay attention to: The density map output may fade to zeros in 90% initialization, I tried the initialization method in the original paper while it didn't work. When this happens, just restart the kernel and re-run... At least, it worked for me, and got the best model in around 70-th epoch. Any other question, contact [me](zhengpeng0108@gmail.com).

7. Loss records:

   ![Loss_records](images/loss_records.png)

### Run:

1. Download dataset;
2. Data generation: run the`generate_datasets.ipynb `.
3. Run the `main.ipynb` to train the model and do the test.

#### Abstraction:

1. **Network = encoder + decoder**, model plot is [here](./images/SANet_noIN.png):

   

   |   Network   |           encoder            |               decoder                |
   | :---------: | :--------------------------: | :----------------------------------: |
   | Composition |   scale aggregation module   |           conv2dTranspose            |
   |    Usage    | extract multi-scale features | generate high resolution density map |

2. Loss:

   Loss = ![loss_formular](https://latex.codecogs.com/gif.latex?L_{Euclidean}+\alpha_CL_C(L_C=1-\frac{1}{N}{\sum_x{SSIM(x)}},\alpha_C=0.001))

3. Normalization layer:

   + Ease the training process;
   + Reduce 'statistic shift problem'.