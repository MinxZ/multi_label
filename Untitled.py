"""
Xception_2 2048 228
"""

reduce_lr_patience = 5
patience = 10  # reduce_lr_patience+1 + 1
early_stopping = EarlyStopping(
    monitor='val_loss', patience=patience, verbose=2, mode='auto')
checkpointer = ModelCheckpoint(
    filepath=f'../models/{model_name}.h5', verbose=0, save_best_only=True)
reduce_lr = ReduceLROnPlateau(
    factor=np.sqrt(0.1), patience=reduce_lr_patience, verbose=2)

datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Compile model
optimizer = 'Adam'
lr = 1e-4  # 1-5e4
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=lr),
    metrics=[f1_score])

# Start fitting model
batch_size = 16
epoch = 1e4
steps_per_epoch = len(x_train) / batch_size / 20,
validation_steps = len(x_val) / batch_size,


"""
Xception_3 2048 228
"""
# lr

reduce_lr_patience = 5
patience = 10  # reduce_lr_patience+1 + 1
early_stopping = EarlyStopping(
    monitor='val_loss', patience=patience, verbose=2, mode='auto')
checkpointer = ModelCheckpoint(
    filepath=f'../models/{model_name}.h5', verbose=0, save_best_only=True)
reduce_lr = ReduceLROnPlateau(
    factor=np.sqrt(0.1), patience=reduce_lr_patience, verbose=2)

datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Compile model
optimizer = 'Adam'
lr = 1e-3  # 1-5e4
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(),
    metrics=[f1_score])

# Start fitting model
batch_size = 16
epoch = 1e4
steps_per_epoch = len(x_train) / batch_size / 20,
validation_steps = len(x_val) / batch_size,
