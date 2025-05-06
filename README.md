# Manga Colorization Model

This is my manga colorization model. 

The purpose of this model is to colorize black and white manga. Currently, I am training it on 30k images from Tagged Anime Illustration [Dataset] (https://www.kaggle.com/datasets/mylesoneill/tagged-anime-illustrations/data), but in the future I am going to implement segmentation into panels and colorization of real manga pages. 

There are several possible architectures I tried (still in process):
- simple autoencoder 
- unet based 
- colorization with dilation by [richzang](https://github.com/richzhang/colorization)
- colorization with fusion to classificator features (described in the [article] (https://dl.acm.org/doi/10.1145/2897824.2925974))
- implementation of GAN by [mberkay0](https://github.com/mberkay0/image-colorization), which could work with all generators listed above 

## References
https://github.com/richzhang/colorization
https://huggingface.co/Keiser41/Example_Based_Manga_Colorization
https://github.com/mberkay0/image-colorization
