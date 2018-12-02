# vgg_project_tf
tensorflow practice 
Grant Spellman

12/1/2018

Below is a brief outline of the steps I took to accomplish the task. The final tensorflow for doing gender classification on image features extracted from [2] can be found in the directory ./model_1e-05_20_0.25/best_model. It can be by calling tf.saved_model.loader.load(...) with that directory.

Note: My scripts assume some parent directories that contain the converted VGG model and the entire dataset. See file path constants for each script. Namely, gen_data_paths.py which references the data set downloaded from [3], and run_pb_model.py, which references the result of step 1 below. 

1) Convert the VGG model in caffe to a tensorflow model.
	
To do this I used Microsoft's tool MMdnn. This required installing Caffe and following the instructions at [1]. This was pretty tricky, and I would not recommend trying it on Windows (!!!).  

2) Pre-process feature extraction on dataset

Once I was able to load the VGG model in tensorflow, I made one pass over the entire dataset to extract features for each image. The features are 4096-vectors extracted before the final FC layer in the VGG model, and they would be the inputs to my gender classifier. This helped to greatly reduce the amount of computation to be done because I only needed one pass through the pre-trained VGG model. The scripts gen_data_paths.py and group_feature_data.py were helpers for organizing and accessing the dataset.  

see : run_pb_model.py

3) Train gender classifier

Using these image features, I trained a simple 3 FC layer model to do binary classification of the images. I kept the model simple to reduce the hyperparameter search space, and mostly tuned performance by tweaking the learning rate. After some search, learning rates around 1e-5 seemed to perform best. 

see: gender_classifier.py

4) Results visualization and evaluation

With my trained model, I visualized my results in viz_results.ipynb. I have included viz_results.html so you can easily review some of the metrics I included to verify my model performance. 

Note: I tried to combine the two frozen tf models to have a final end to end gender classifier pipeline, but I really struggled to get this to work. My attempts are in combine_models.py, and I used the below list of sites in my search. 

https://blog.konpat.me/tf-connecting-two-graphs-together/
https://github.com/tensorflow/models/issues/1988
https://github.com/tensorflow/tensorflow/issues/20825
https://stackoverflow.com/questions/51881957/deploy-pre-trained-inception-in-tensorflowserving-fails-savedmodel-has-no-varia
https://stackoverflow.com/questions/42858785/connect-input-and-output-tensors-of-two-different-graphs-tensorflow


References

[1] https://github.com/Microsoft/MMdnn/tree/master/mmdnn/conversion/tensorflow

[2] http://www.robots.ox.ac.uk/~vgg/software/vgg_face/  

[3] https://s3.amazonaws.com/matroid-web/datasets/agegender_cleaned.tar.gz