import os
import uuid
from flask import Flask, request, render_template, redirect, url_for, flash, session, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import numpy as np

from src.utils.constants import MODELS_DIR, MODEL_CHECKPOINT_FILENAME, IMAGE_SIZE, TARGET_LABELS
from src.utils.logger import logger

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_CHECKPOINT_FILENAME)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'secret' 
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = None

def load_keras_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            logger.info(f"Carregando modelo Keras em: {MODEL_PATH}")
            model = tf.keras.models.load_model(MODEL_PATH)
            logger.info("Modelo Keras carregado com sucesso.")
            
            dummy_input = np.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
            _ = model.predict(dummy_input)
            logger.info("Model warm-up prediction complete.")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo Keras: {e}")
            flash(f"Erro ao carregar o modelo: {e}", "error")
            model = None
    else:
        logger.error(f"Model file not found at {MODEL_PATH}")
        flash(f"Model file not found at {MODEL_PATH}. Please ensure it's trained and available.", "error")
        model = None


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size):
    try:
        img = Image.open(image_path)
        # Ensure image is RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        img_array = tf.keras.preprocessing.image.img_to_array(img) 

        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Erro ao processar imagem {image_path}: {e}")
        return None
    

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # file upload
        if 'file' not in request.files:
            flash('No file part', 'warning')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('Nenhum arquivo selecionado', 'warning')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # clear previous session data
            session.pop('filename', None)
            session.pop('prediction', None)
            session.pop('error', None)

            filename = secure_filename(file.filename)
            # unique filename to avoid conflicts and browser caching issues
            unique_id = uuid.uuid4().hex
            _, ext = os.path.splitext(filename)
            unique_filename = f"{unique_id}{ext}"
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            try:
                file.save(filepath)
                session['filename'] = unique_filename
                flash('Imagem carregada com sucesso!', 'success')
            except Exception as e:
                logger.error(f"Erro ao salvar arquivo: {e}")
                flash(f"Erro ao salvar arquivo: {e}", 'error')
                session['error'] = f"Erro ao salvar arquivo: {e}"
            return redirect(url_for('index'))

    filename = session.get('filename')
    prediction = session.get('prediction')
    error_message = session.get('error')

    return render_template('index.html', filename=filename, prediction=prediction, error_message=error_message)

@app.route('/classify', methods=['POST'])
def classify_image():
    global model
    if model is None:
        flash("Modelo não carregado. Não é possível classificar.", "error")
        session['error'] = "Modelo não carregado. Não é possível classificar."
        return redirect(url_for('index'))

    filename = session.get('filename')
    if not filename:
        flash('Nenhuma imagem enviada para classificação.', 'warning')
        session['error'] = 'Nenhuma imagem enviada para classificação.'
        return redirect(url_for('index'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        flash('Imagem carregada não encontrada.', 'error')
        session.pop('filename', None)
        session['error'] = 'Imagem carregada não encontrada.'
        return redirect(url_for('index'))

    try:
        processed_image = preprocess_image(filepath, IMAGE_SIZE)
        if processed_image is None:
            flash('Erro ao processar imagem.', 'error')
            session['error'] = 'Erro ao processar imagem.'
            return redirect(url_for('index'))

        raw_prediction = model.predict(processed_image)[0]
        
        # TARGET_LABELS = ['0', '1'] ('0' No Finding, '1' Effusion)
        predicted_class_index = 1 if raw_prediction[0] > 0.5 else 0
        
        if predicted_class_index == 1:
            result_text = "Presença de Efusão (derrame pleural) detectada."
        else:
            result_text = "Sem achado de Efusão (derrame pleural)."

        prediction_details = f"{result_text} (Confiança: {raw_prediction[0]:.4f})"
        session['prediction'] = prediction_details
        flash('Imagem classificada com sucesso!', 'success')

    except Exception as e:
        logger.error(f"Erro durante classificação: {e}")
        flash(f"Erro durante classificação: {e}", 'error')
        session['prediction'] = f"Error: {e}"
        session['error'] = f"Erro durante classificação: {e}"

    return redirect(url_for('index'))

@app.route('/remove', methods=['POST'])
def remove_image():
    filename = session.pop('filename', None)
    session.pop('prediction', None)
    session.pop('error', None)

    if filename:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                flash('Imagem removida com sucesso.', 'success')
            except Exception as e:
                logger.error(f"Error removing file {filepath}: {e}")
                flash(f"Error removing file: {e}", 'error')
        else:
            flash('Imagem não encontrada. Não foi possível remover.', 'warning')
    else:
        flash('Sem imagem para remoção.', 'info')
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    load_keras_model() 
    if model is None:
        print("WARNING: Não foi possível carregar o Modelo Keras. Aplicação irá rodar, mas sem capacidade de classificação.")
    app.run(debug=True) 