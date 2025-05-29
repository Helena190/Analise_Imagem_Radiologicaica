import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.utils.logger import logger
from src.utils.constants import IMAGE_SIZE, LEARNING_RATE

class ModelBuilder:
    """
    Constrói a arquitetura do modelo de aprendizado profundo, tipicamente usando transferência de aprendizado.
    """
    def __init__(self, image_size=IMAGE_SIZE, learning_rate=LEARNING_RATE):
        self.image_size = image_size
        self.learning_rate = learning_rate
        logger.info(f"ModelBuilder initialized with image size {self.image_size} and learning rate {self.learning_rate}.")

    def build_transfer_learning_model(self):
        """
        Constrói um modelo de transferência de aprendizado baseado no MobileNetV2 para classificação binária.

        Retorna:
            tf.keras.Model: O modelo Keras compilado.
        """
        logger.info("Building transfer learning model using MobileNetV2...")
        
        # Função de pré-processamento para MobileNetV2
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

        # Carrega o modelo base (MobileNetV2) sem a camada de classificação superior
        # e com pesos pré-treinados no ImageNet.
        base_model = tf.keras.applications.MobileNetV2(input_shape=self.image_size + (3,),
                                                       include_top=False,
                                                       weights='imagenet')
        
        # Congela o modelo base para evitar que seus pesos sejam atualizados durante o treinamento
        base_model.trainable = False
        logger.info("MobileNetV2 base model loaded and frozen.")

        # Cria o novo modelo em cima do modelo base
        inputs = keras.Input(shape=self.image_size + (3,))
        x = preprocess_input(inputs) # Aplica o pré-processamento do MobileNetV2
        x = base_model(x, training=False) # Executa o modelo base em modo de inferência
        x = layers.GlobalAveragePooling2D()(x) # Pooling médio global para achatar as características
        x = layers.Dropout(0.2)(x) # Dropout para regularização
        outputs = layers.Dense(1, activation='sigmoid')(x) # Camada de saída para classificação binária

        model = keras.Model(inputs, outputs)

        # Compila o modelo
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])
        
        logger.info("Model compiled successfully.")
        model.summary(print_fn=logger.info) # Log do resumo do modelo
        
        return model

if __name__ == "__main__":
    logger.info("Running ModelBuilder example...")
    builder = ModelBuilder()
    model = builder.build_transfer_learning_model()
    logger.info("Model built successfully for demonstration.")