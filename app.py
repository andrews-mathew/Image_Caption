from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
from deep_translator import GoogleTranslator
import gradio as gr
import io

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_caption(image):
    image = image.convert("RGB")
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

def translate_to_Malayalam(english_caption):
    translator = GoogleTranslator(source='en', target='ml')
    Malayalam_caption = translator.translate(english_caption)
    return Malayalam_caption

def predict(image):
    english_caption = generate_caption(image)
    Malayalam_caption = translate_to_Malayalam(english_caption)
    return Malayalam_caption

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Textbox(label="Caption in Malayalam"),
    title="Image Captioning in Malayalam",
    description="Upload an image to get a caption in Malayalam!"
)

interface.launch()
