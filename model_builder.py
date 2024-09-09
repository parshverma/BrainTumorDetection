import tensorflow as tf

def build_model():
    base_model = tf.keras.applications.DenseNet201(input_shape=(150, 150, 3), include_top=False, weights='imagenet', pooling='avg')
    base_model.trainable = False
    
    inp = base_model.input
    x = tf.keras.layers.Dense(128, activation='relu')(base_model.output)
    out = tf.keras.layers.Dense(4, activation='softmax')(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
