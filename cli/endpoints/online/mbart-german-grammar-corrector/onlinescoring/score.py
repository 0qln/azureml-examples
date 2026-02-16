import os
import logging
import json
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, GenerationConfig, BitsAndBytesConfig

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    global tokenizer
    global generation_config

    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "model"
    )

    # deserialize the model
    model = MBartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = MBart50TokenizerFast.from_pretrained(model_path, src_lang="de_DE", tgt_lang="de_DE")
    generation_config = GenerationConfig.from_model_config(model.config)
    generation_config.forced_bos_token_id = tokenizer.lang_code_to_id["de_DE"]

    logging.info("Init complete")

def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info("model 1: request received")

    data = json.loads(raw_data)["data"]
    model_input = tokenizer(data, return_tensors='pt', padding=True)

    output = model.generate(
        model_input["input_ids"],
        attention_mask=model_input["attention_mask"],
        generation_config=generation_config
    )
    result = tokenizer.batch_decode(output, skip_special_tokens=True)

    logging.info("Request processed")
    return result
