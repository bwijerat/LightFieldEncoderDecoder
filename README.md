# LightFieldEncoderDecoder
CNN based auto-encoder for Light Field compression - PyTorch

This project explored the possibility of building a deep CNN autoencoder for light field compression by exploiting the small variations present in sub-regions of the light field. A branch based network architecture was incrementally improved by modifying and testing network configurations along with updating the training scheme by varying the data sampling scheme, regularization approaches and other deep learning relevant algorithms.

### Example of used synthetic Light Fields
#### Center view of 6 light fields from Container1: https://github.com/cvia-kn/lf_autoencoder_cvpr2018_code 
![Image1](https://github.com/bwijerat/LightFieldEncoderDecoder/blob/main/Report/media/LF_collage.jpg)

#### Sampling scheme for 4-branch network, sampled at 3x3 regions of the 9x9 array of sub-aperture view patches, each patch
![Image2](https://github.com/bwijerat/LightFieldEncoderDecoder/blob/main/Report/media/LF_sampling_scheme.jpg)
