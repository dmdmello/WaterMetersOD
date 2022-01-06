Code for training a simple object detector for the TLKWaterMeters dataset. Most of the code is defined in the notebook ObjectDetection.ipynb.

In order to run the notebook in its entirety, you will need to perform one of the following:
  1) decompress the .zip file "images_preprocessed.zip" and simply run the notebook
  2) move the original "TLKWaterMeters/images/" folder into "TLKWaterMeters/" in this project, and follow the notebook instructions for preprocessing the original images.

The model whose training is described in the notebook is saved in "training_outputs/model_pretrained/obj_rec_model.pth". To directly load it in the notebook, uncomment the code lines indicating to do so after the model is defined in the notebook.

Image with bounding box predictions sampled after some training epochs in the notebook are also saved as outputs in "training_outputs/images_bb_samples/".
