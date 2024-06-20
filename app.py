from flask import Flask, jsonify, request
import torch
from torchvision.utils import save_image
import io
import base64
from mnist_generator import MnistGenerator

app = Flask(__name__, static_folder='templates')

latent_dim = 100
generator = MnistGenerator()
generator.load_state_dict(torch.load('models/generator.pth', map_location=torch.device('cpu')))
generator.eval()


@app.route('/')
def home():
    return app.send_static_file('index.html')


@app.route('/generate_number', methods=['POST'])
def generate_number():
    try:
        data = request.json
        number = data.get('number', '')

        images = []
        for _ in range(int(number)):
            random_tensor = torch.randn(1, latent_dim)
            img = generator(random_tensor)
            img = img.squeeze().detach().cpu()

            buf = io.BytesIO()
            save_image(img, buf, format='PNG')
            buf.seek(0)

            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            images.append(img_base64)
        return jsonify({'images': images})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
