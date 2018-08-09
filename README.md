# multi_label
multi_label training
A multi label classifier implemented with Keras.
I change the Keras ImageGenerator a little to support multi-label datasuch as:
category: [0, 1, 0, 0] (Only one category for one object)
color: [0, 1, 1, 0] (Several colors for one object)
pattern: [0, 1, 1, 1] (Several pattern for one object)
