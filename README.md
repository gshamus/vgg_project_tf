# vgg_project_tf
tensorflow practice 
Grant Spellman

Matroid 12/1/2018

Below is a brief outline of the steps I took to accomplish the task. The final tensorflow for doing gender classification on image features extracted from [2] can be found in the directory ./model_1e-05_20_0.25/best_model. It can be by calling tf.saved_model.loader.load(...) with that directory.


1) Convert the VGG model in caffe to a tensorflow model.
	
To do this I used Microsofts tool MMdnn. This required installing Caffe and following the instructions at [1]. This was pretty tricky, and I would not recommend trying it on Windows (!!!).  

2) Pre-process feature extraction on dataset

Once I was able to load the VGG model in tensorflow, I made one pass over the entire dataset to extract features for each image. The features are 4096-vectors extracted before the final FC layer in the VGG model, and they would be the inputs to my gender classifier. This helped to greatly reduce the amount of computation to be done because I only needed one pass through the pre-trained VGG model. The scripts gen_data_paths.py and group_feature_data.py were helpers for organizing and accessing the dataset.  

see : run_pb_model.py

3) Train gender classifier

Using these image features, I trained a simple 3 FC layer model to do binary classification of the images. I kept the model simple to reduce the hyperparameter search space, and mostly tuned performance by tweaking the learning rate. After a sparse search, learning rates around 1e-5 seemed to perform best. 

see: gender_classifier.py

4) Results visualization and evaluation

With my trained model, I visualized my results in viz_results.ipynb. I have included viz_results.html so you can easily review some of the metrics I included to verify my model performance. 


https://blog.konpat.me/tf-connecting-two-graphs-together/















References

[1] https://github.com/Microsoft/MMdnn/tree/master/mmdnn/conversion/tensorflow
[2] http://www.robots.ox.ac.uk/~vgg/software/vgg_face/  