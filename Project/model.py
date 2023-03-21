import tensorflow as tf

class Unet():

    def __init__(self):

        # Contracting path
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu")
        self.batch1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation="relu")
        self.batch2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same", activation="relu")
        self.batch3 = tf.keras.layers.BatchNormalization()
        self.conv4 = tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same", activation="relu")
        self.batch4 = tf.keras.layers.BatchNormalization()
        self.conv5 = tf.keras.layers.Conv2D(1024, (3, 3), strides=(2, 2), padding="same", activation="relu")
        self.batch5 = tf.keras.layers.BatchNormalization()

        # Middle
        self.conv6 = tf.keras.layers.Conv2D(1024, (3, 3), strides=(2, 2), padding="same", activation="relu")
        self.batch6 = tf.keras.layers.BatchNormalization()
        self.drop1 = tf.keras.layers.Dropout(0.5)

        # Expansive path
        self.deconv1 = tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding="same", activation="relu")
        self.debatch1 = tf.keras.layers.BatchNormalization()
        self.drop2 = tf.keras.layers.Dropout(0.5)
        self.concat1 = tf.keras.layers.Concatenate()
        self.conv7 = tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu")
        self.batch7 = tf.keras.layers.BatchNormalization()

        self.deconv2 = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same", activation="relu")
        self.debatch2 = tf.keras.layers.BatchNormalization()
        self.concat2 = tf.keras.layers.Concatenate()
        self.conv8 = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu")
        self.batch8 = tf.keras.layers.BatchNormalization()

        self.deconv3
        
def __call__(self, input):

    # contracting path
    c1 = self.conv1(input)
    c1 = self.batch1(c1)
    c1 = self.leakyrelu1(c1)
    c2 = self.conv2(c1)
    c2 = self.batch2(c2)
    c2 = self.leakyrelu2(c2)
    c3 = self.conv3(c2)
    c3 = self.batch3(c3)
    c3 = self.leakyrelu3(c3)
    c4 = self.conv4(c3)
    c4 = self.batch4(c4)
    c4 = self.leakyrelu4(c4)
    c5 = self.conv5(c4)
    c5 = self.batch5(c5)
    c5 = self.leakyrelu5(c5)
    c6 = self.conv6(c5)
    c6 = self.batch6(c6)
    c6 = self.leakyrelu6(c6)

    # middle
    m = self.conv7(c6)
    m = self.batch7(m)
    m = self.leakyrelu7(m)

    # expansive path
    u1 = self.deconv1(m)
    u1 = tf.concat([u1, c5], axis=-1)
    u1 = self.debatch1(u1)
    u1 = self.drop1(u1)
    u2 = self.deconv2(u1)
    u2 = tf.concat([u2, c4], axis=-1)
    u2 = self.debatch2(u2)
    u2 = self.drop2(u2)
    u3 = self.deconv3(u2)
    u3 = tf.concat([u3, c3], axis=-1)
    u3 = self.debatch3(u3)
    u3 = self.drop3(u3)
    u4 = self.deconv4(u3)
    u4 = tf.concat([u4, c2], axis=-1)
    u4 = self.debatch4(u4)
    u5 = self.deconv5(u4)
    u5 = tf.concat([u5, c1], axis=-1)
    u5 = self.debatch5(u5)
    output = self.conv6(u5)
    output = tf.nn.softmax(output, axis=-1)
    
    return output
