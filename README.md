# FaceRecog

The dataset I used is `Face Place`, see `face_db.md` to get more info.

The results I get are to large to upload, see `models.md` to get more info.

This repo is my graduation project, Face Recognition. The whole code I used is in code folder.
I use Keras to design net architecture(ptorch is not finished, and will not be append), trian 
and analyse, for ConvNet visualization, I use Keras-vis, it is convient and help a lot.

I experiment SGD, RMSProp and Adam on Vanilla CNN, and found Adam > RMSProp > SGD.

I experiment Vanilla CNN, ResNet and DenseNet, and found DenseNet > ResNet > Vanilla CNN.

I reduced the params of ResNet and DenseNet, found with #params reduced, epoch to achive 90% acc is increased.

I visualize convnet by Saliency and Grad-CAM, results is in `code/analyse`.

`code/src` mainly are raw python code of my experiment.

`code/experiment` mainly are jupyter notebook files of experiment results.

`code/analyses` mainly are some summary and visualizion result.

You can explore the `code` folder to find more specific info.

If you have any questions about this, please [email](cugtyt@qq.com) me or just throw issues.