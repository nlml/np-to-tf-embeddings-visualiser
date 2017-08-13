# np-to-tf-embeddings-visualiser
Quick function to go from a dictionary of sets of (images, labels, feature vectors) to checkpoints that can be opened in Tensorboard

To run the example, just:
`python example.py`

Then navigate to the `embeds` directory created by that script and run tensorboard with:
`tensorboard --logdir=embeds`

Then head to http://localhost:6006/ and click Embeddings.