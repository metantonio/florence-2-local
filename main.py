import requests

from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 


model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

prompt = "<OD>" # Object Detection
# prompt = "<CAPTION>"
# prompt = "<DETAILED_CAPTION>"
# prompt = "<MORE_DETAILED_CAPTION>"
# prompt = "<DENSE_REGION_CAPTION>"
# prompt = "<REGION_PROPOSAL>"
# prompt = "<OCR>"
# prompt = "<OCR_WITH_REGION>"
#task_prompt = '<OPEN_VOCABULARY_DETECTION>' # You write a input and it will try to find it in the image
#results = run_example(task_prompt, text_input="a green car")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=prompt, images=image, return_tensors="pt")

generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    num_beams=3,
    do_sample=False
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

parsed_answer = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))

print(parsed_answer)
