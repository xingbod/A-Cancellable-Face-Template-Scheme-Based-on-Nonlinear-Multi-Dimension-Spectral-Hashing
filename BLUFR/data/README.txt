xTwo basic features were extracted from the FRGCv2 and LFW databases, respectively, provided as separate downloads (frgc.mat and lfw.mat).

The FRGC features were extracted according to Section 5.3 of our IJCB paper [1]. For the LFW features, face images were firstly aligned and cropped by 22 landmarks provided by [2], and then features were extracted similar as on FRGC.

These features are just provided for a demo. They are not state of the art features, so you are suggested to use or design a better feature representation. You may also download the HighDimLBP features provided by the authors of [2]. To extract your own features, make sure to use the image list contained in the configuration file to determine the image order.


References:

[1] Shengcai Liao, Zhen Lei, Dong Yi, Stan Z. Li, "A Benchmark Study of Large-scale Unconstrained Face Recognition." In IAPR/IEEE International Joint Conference on Biometrics, Sep. 29 - Oct. 2, Clearwater, Florida, USA, 2014.

[2] Dong Chen, Xudong Cao, Fang Wen, and Jian Sun, "Blessing of dimensionality: High-dimensional feature and its efficient compression for face verification." In IEEE Conference on Computer Vision and Pattern Recognition, 2013.
