# multi_label
A project started for iMaterialist Challenge (Fashion) at FGVC5(https://www.kaggle.com/c/imaterialist-challenge-fashion-2018).
I achived 22nd (top 11%) in the end. https://www.kaggle.com/meikintom

## A multi-label multi-loss-func classifier implemented with Keras.
I change the Keras ImageGenerator a little to support multi-label datasuch as:

category: [0, 1, 0, 0] (Only one category for one object)

color: [0, 1, 1, 0] (Several colors for one object)

pattern: [0, 1, 1, 1] (Several pattern for one object)
