## UM-PDD

## GAN based Semi-Supervised Learning Approach for Plant Disease Classification


### Problem Statement
<p align="justify">
Modern agriculture continues to search for innovative ways to improve productivity and sustainability to meet the food demands of the increasing population amidst reductions in cultivable land, climate change and political instability. Researchers have investigated various methods to incorporate precision in predictive agronomic systems to determine healthy crop yield, determine farmable area, fertilizer recommendation etc. However, developing these systems includes labelling of enormous amount of data which is costly, time consuming and requires relevant domain knowledge. There is a need to build a data efficient and high-quality predictive model using a small number of labelled images. This research proposes using semi-supervised learning approach through generative adversarial networks (GANs) for detection of plant disease in crops of an arable land. Our approach is based on stacked discriminator models with shared weights. Separate logical unsupervised and supervised discriminator are created where the supervised discriminator’s output layer is reused and fed as input to the unsupervised discriminator enabling both labelling and classification of images. 

Our study is limited to the prediction of plant disease; the technique can be further explored for other predictive applications of agronomic systems such as detection of pests by using relevant curated data. Furthermore, transparency and interpretability can be added to our models by use of deep learning visualization methods.
  </p>
  
  ### Dataset Source
 Data was retrived from Kaggle. Since it is too large, it is not pushed to the repository. Please download from the following link:
- https://www.kaggle.com/vipoooool/new-plant-diseases-dataset
