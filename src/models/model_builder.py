import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.utils.logger import logger
from src.utils.constants import IMAGE_SIZE, LEARNING_RATE

class ModelBuilder:
    """constrói a arquitetura do modelo de aprendizado profundo, tipicamente usando transferência de aprendizado."""
    def __init__(self, image_size=IMAGE_SIZE, learning_rate=LEARNING_RATE):
        self.image_size = image_size
        self.learning_rate = learning_rate
        logger.info(f"modelbuilder inicializado com tamanho de imagem {self.image_size} e taxa de aprendizado {self.learning_rate}.")

    def build_transfer_learning_model(self):
        """constrói um modelo de transferência de aprendizado baseado no mobilenetv2 para classificação binária."""
        logger.info("construindo modelo de transferência de aprendizado usando mobilenetv2...")
        
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

        base_model = tf.keras.applications.MobileNetV2(input_shape=self.image_size + (3,),
                                                       include_top=False,
                                                       weights='imagenet')
        
        base_model.trainable = False
        logger.info("modelo base mobilenetv2 carregado e congelado.")

        # Cria o novo modelo em cima do modelo base
        inputs = keras.Input(shape=self.image_size + (3,))
        x = preprocess_input(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = keras.Model(inputs, outputs)

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])
        
        logger.info("modelo compilado com sucesso.")
        model.summary(print_fn=logger.info)
        
        return model

if __name__ == "__main__":
    logger.info("executando exemplo de modelbuilder...")
    builder = ModelBuilder()
    model = builder.build_transfer_learning_model()
    logger.info("modelo construído com sucesso para demonstração.")