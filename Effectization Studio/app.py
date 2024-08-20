from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image
import io
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from stability_sdk import client
from werkzeug.utils import secure_filename

# Replace 'your-api-key-here' with your actual API key from Stability AI
api_key = 'your api key here'
stability_api = client.StabilityInference(
    key=api_key,
    verbose=True
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def load_image(image_path):
    return Image.open(image_path)

def resize_image(image, size=(800, 450)):
    return image.resize(size, Image.LANCZOS)

def generate_image(prompt, input_image_path, output_image_path):
    # Load the input image
    init_image = load_image(input_image_path)

    # Resize the image to 800x450 pixels
    init_image = resize_image(init_image)

    # Convert to RGB if needed
    if init_image.mode != 'RGB':
        init_image = init_image.convert('RGB')

    # Generate the image based on the prompt and initial image
    answers = stability_api.generate(
        prompt=prompt,
        init_image=init_image,
        steps = 100,
        cfg_scale = 9.0,
        width=1024,
        height=1024,
        start_schedule=0.6,
        seed=123463446,
        sampler=generation.SAMPLER_K_EULER_ANCESTRAL
    )

    # Save the generated image
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                print("Request filtered by API")
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary))
                img.save(output_image_path)
                print(f"Image saved at {output_image_path}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' in request.files and request.files['image'].filename != '':
            image = request.files['image']
            image_filename = secure_filename(image.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
            image.save(image_path)

            modified_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'modified_' + image_filename)
            prompt = "Cyberpunk version of the image"
            generate_image(prompt, image_path, modified_image_path)

            return redirect(url_for('result', input_filename=image_filename, filename=os.path.basename(modified_image_path)))

    return render_template('index.html')

@app.route('/result')
def result():
    input_filename = request.args.get('input_filename', None)
    filename = request.args.get('filename', None)
    return render_template('result.html', input_filename=input_filename, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
