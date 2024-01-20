# Lymph Node Segmentation

Detecting lymph nodes (LN) plays a vital role in enhancing the accuracy of cancer diagnosis and treatment procedures. However, the inherent challenges in this process arise from the low-contrast features in CT scan images, coupled with the diverse shapes, sizes, and orientations of the nodes, along with their sparse distribution. These factors make the detection step particularly difficult and contribute to a high number of false positives.

Manual examination of CT scan slices can be time-consuming, and the occurrence of false positives may distract clinicians from essential tasks. Our objective is to streamline the LN detection process by leveraging Deep Learning-based segmentation. This repository houses a collection of cutting-edge segmentation models and loss functions specifically designed for LN detection. It's important to note that this project is an ongoing effort, and we anticipate continuously adding more resources. Our ultimate aim is to introduce innovative solutions that address the challenges associated with LN detection, enhancing the efficiency and accuracy of the entire process.

## Instruction

- Run `https://github.com/tahsin314/Lymph_Node_Segmentation.git`
- Run `conda env create -f environment.yml` to create a conda environment.
- Extract `Cervical Lymph Nodes- First 100 Cases` and `Cervical Lymph Nodes- Second 121 Cases`. Keep all patient data under the same directory.
- Update your data directory in `config/config.py` files `config_params` dictionary.
- Run `data_process.py`. It will read each patient data and corresponding masks and convert to soft tissue windowed `npz` files. It will also generate lymph node labels (1-present, 0-not present) for each slice.
- You can configure parameters such as `batch size`, `learning rate`, `image dimension` from `config.py`.
- Run `train.py`. It will train a classification model on the extracted data. You can choose several models and loss functions to use for training for the `config.py` file. A description of the models and loss functions are added below.

## Models

This section describes the integrated models in this pipeline:

- **`UNet`** : The **U-Net** architecture is a convolutional neural network (CNN) designed for semantic segmentation tasks, particularly in medical image analysis. Its distinctive U-shaped structure consists of a contracting path to capture context and a symmetric expansive path for precise localization. This enables the network to effectively identify and delineate structures in images, making it especially suited for tasks such as organ and tumor segmentation. The U-Net's success lies in its ability to preserve spatial information through skip connections, facilitating accurate pixel-wise predictions while maintaining computational efficiency.
  - Paper: [Ronneberger et al., 2015][unet]
  - Code: [Pytorch-UNet][unet_code]

- **`Stepwise Feature Fusion Network`** : The SSFormer archietecture uses a pyramid Transformer encoder to extract multi-scale features and a Progressive Locality Decoder (PLD) to fuse local and global features stepwise. SSFormer achieves state-of-the-art performance on several polyp segmentation benchmarks. The paper also introduces a local feature emphasis module that can reduce the attention dispersion of Transformers and improve their detail processing ability.
  - Paper: [Wang et al, 2022][ssformer]
  - Code: [ssformer][ssformer_code]

- **`CaraNet`** : CaraNet combines a context axial reverse attention module, which detects global and local features, and a channel-wise feature pyramid module, which extracts multi-scale features, to improve the segmentation accuracy and robustness.  
  - Paper: [Lou et al, 2021][caranet]
  - Code: [CaraNet][caranet_code]

- **`FCBFormer`** : The proposed transformer-based architecture addresses DL vulnerabilities, combining a primary transformer branch for feature extraction with a secondary fully convolutional branch for full-size prediction. This approach outperforms existing methods, as evidenced by state-of-the-art results on mDice, mIoU, mPrecision, and mRecall metrics on Kvasir-SEG and CVC-ClinicDB datasets. The model also demonstrates superior generalization performance when trained on one dataset and evaluated on the other.
  - Paper: [Sanderson et al, 2022][fcbformer]
  - Code: [FCBFormer][fcbformer_code]

- **`DUCKNet`** : They proposed a custom convolutional block, *DUCK*, that combines different types of convolutions to capture features at multiple scales and resolutions. They also use a residual downsampling mechanism to preserve the original image information at each encoder level. Their main contributions are:
the DUCK block, which allows more in-depth feature selection and accurate polyp border prediction and the residual downsampling, which enables the use of the initial image information at each resolution level in the encoder segment.

  - Paper: [Dumitru et al, 2023][duck]
  - Code: [DUCK-Net][duck_code]

- **`FocusNet`** : *WIP*

- **`LiVS`** : *WIP*

## Loss Functions

*WIP*

## Features

- **`Probability Correction Strategy`** : Probability Correction Strategy (PCS) is a method to improve the segmentation of small polyps, which are often underrepresented and have low confidence in the model predictions. PCS works by normalizing the logits (pre-activation values) of the positive and negative samples according to their proportions in the image. This way, the effect of sample imbalance is reduced and the predictions are more accurate. PCS is applied only during inference, so it does not affect the training process or introduce extra computation cost.
  - Paper: [Wei et al, 2021][sam]
  - Code: [SANet][sam_code]

## Resources

- This [list][segmentation_etis] contains a list of models benchmarked with high scores on several standard medical segmentation datasets.
  
## TODO

- Return loss function outputs as dictionary. In this case, it will be easier to process loss functions that returns multiple outputs. Also, dictionary key, value pairs can be logged into `wandb` easily.

- I'm planning on reporting separate Metric Performance for very small nodules, relatively small, and small nodules to demonstrate the model's effectiveness in prediction. I might tweak the loss functions based on nodule size.

- I will conduct more experiments on weighted loss functions, such as replacing average pooling weighting with Gaussian blur, to find out which is more effective.

- Addressing data imbalance by using a weighted sampler to balance the no-nodule and nodule slices is another goal.

- I use a threshold of `0.5` to differentiate the foreground and the background. I will experiment with `precision_recall_curve` and the Youden score to find a better-fitted threshold.

- ~~Due to resource constraints, I'm still training over 2D slices. I am planning to stack adjacent slices together to form multichannel data for each mask. This will help the model understand the continuity and dependency of objects within the CT slices without being resource-exhaustive.~~

- Tuning parameters has been a major issue in my experiments. I am thinking of parameter sweeping through methods like Random Search, Grid Search, Brute force search, etc., to find the best set of parameters. I'm looking into using `Optuna`, a Python-based library designed for this task.

- Some lymph nodes appear to be predicted at a slightly different distance compared to the original ones. However, this discrepancy causes a significant drop in performance metrics, despite the visual closeness to the actual output. Perhaps we can devise a metric capable of discerning if a predicted node is in close proximity to the original one, thereby yielding a favorable score.

- Currently, I'm trying to maximize the recall score by reducing the probability threshold. This in fact, increases false positive cases and thus reduces dice scores. We need to figure out a way to reduce the false positive predictions.

[unet]: https://arxiv.org/abs/1505.04597
[unet_code]: https://github.com/milesial/Pytorch-UNet
[ssformer]: https://arxiv.org/pdf/2203.03635.pdf
[ssformer_code]: https://github.com/Qiming-Huang/ssformer
[caranet]: https://arxiv.org/ftp/arxiv/papers/2108/2108.07368.pdf
[caranet_code]: https://github.com/AngeLouCN/CaraNet
[fcbformer]: https://arxiv.org/pdf/2208.08352.pdf
[fcbformer_code]: https://github.com/ESandML/FCBFormer
[sam]: https://arxiv.org/pdf/2108.00882.pdf
[sam_code]: https://github.com/weijun88/SANet
[segmentation_etis]: https://paperswithcode.com/sota/medical-image-segmentation-on-etis
[duck]: https://arxiv.org/ftp/arxiv/papers/2311/2311.02239.pdf
[duck_code]: https://github.com/RazvanDu/DUCK-Net