from flask import Flask, render_template, url_for, current_app
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import base64

import matplotlib.pyplot as plt

app = Flask(__name__)

# Charge le modèle IA (ajuste le chemin en fonction de ton modèle)
gan_model = tf.keras.models.load_model('goblin.h5')

def generate_image(generator):
    # Générer un vecteur de bruit aléatoire
    latent_dim = 100
    noise = np.random.randn(1, latent_dim)

    # Utiliser le générateur pour produire une image
    generated_img = generator.predict(noise)

    # Réajuster la normalisation [0, 1]
    generated_img = 0.5 * generated_img + 0.5

    # Convertir l'image NumPy en format image PIL
    generated_img = (generated_img[0] * 255).astype(np.uint8)
    generated_img = Image.fromarray(generated_img)

    return generated_img

def generate_and_show_image(generator, epochs, latent_dim):
    # Générer un vecteur de bruit aléatoire
    noise = np.random.randn(1, latent_dim)

    # Utiliser le générateur pour produire une image
    generated_img = generator.predict(noise)

    # Réajuster la normalisation [0, 1]
    generated_img = 0.5 * generated_img + 0.5  

    # Créer la figure Matplotlib pour pouvoir la passer au front-end
    plt.switch_backend('agg')
    plt.imshow(generated_img[0, :, :, 0], cmap='gray')
    plt.axis('off')
    plt.title(f'Generated Image')

    # Sauvegarder la figure dans un buffer
    fig_buffer = BytesIO()
    plt.savefig(fig_buffer, format='png')
    fig_str = "data:image/png;base64," + base64.b64encode(fig_buffer.getvalue()).decode()
    plt.close()

    return fig_str

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/generate')
def generate():
     # Générer une image avec le modèle GAN
    fig_content = generate_and_show_image(gan_model.layers[0], 10000, 100)

    # Sauvegarder l'image à la racine du projet
    #save_path = 'generated_image.jpg'
    #generated_img.save(save_path)

    # Enregistre l'image générée temporairement
    #img_buffer = BytesIO()
    #generated_img.save(img_buffer, format="JPEG")

    # Convertir l'image en base64 pour l'afficher dans la page HTML
    #img_str = "data:image/jpeg;base64," + base64.b64encode(img_buffer.getvalue()).decode()

    return render_template('generate.html', generated_image=fig_content)

if __name__ == '__main__':
    app.run(debug=True)