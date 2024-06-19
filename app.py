from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import ezkl
import os
import tempfile
import time

import torch
import asyncio
from torchvision import transforms

app = Flask(__name__)
CORS(app)

# Paths to the necessary files
# MODEL_PATH = os.path.join('network_lenet.onnx')
COMPILED_MODEL_PATH = os.path.join('network.compiled')
PK_PATH = os.path.join('key.pk')
VK_PATH = os.path.join('key.vk')  # Verification key path
SETTINGS_PATH = os.path.join('settings.json')
SRS_PATH = os.path.join("kzg.srs")
verifier_address_path = os.path.join("address.json")

# Save the block hash where the contract is deployed
with open(verifier_address_path, 'r') as file:
    verifier_block_addr = file.read().rstrip()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform = transforms.Compose([transforms.ToTensor()])

@app.route('/', methods=['GET'])
def index():
    return jsonify({'status': 'ok', 'res': "Welcome to ezkl proving server"})

async def compute_proof(json_data):
    proof = False
    verification_res = None
    stats = {}

    with tempfile.NamedTemporaryFile() as pffo:
        start_time = time.time()
        
        # Load image data and preprocess
        inp = json_data.get('input')
        
        # Dump input json
        input_json = tempfile.NamedTemporaryFile(mode="w+")
        json.dump(inp, input_json)
        input_json.flush()
        
        witness = tempfile.NamedTemporaryFile(mode="w+")
        
        # Generate witness
        wit_start_time = time.time()
        wit = await ezkl.gen_witness(input_json.name, COMPILED_MODEL_PATH, witness.name)
        wit_end_time = time.time()
        stats['witness_generation_time'] = wit_end_time - wit_start_time

        # Generate proof
        proof_start_time = time.time()
        res = ezkl.prove(witness.name, COMPILED_MODEL_PATH, PK_PATH, pffo.name, 'single')
        proof_end_time = time.time()
        stats['proof_generation_time'] = proof_end_time - proof_start_time

        # Check if proof is generated
        if res.get("proof"):
            proof = True

        # Verify the proof
        verify_start_time = time.time()
        verification_res = await ezkl.verify_evm(
            verifier_block_addr,
            pffo.name,
            "http://127.0.0.1:3030"
        )
        verify_end_time = time.time()
        stats['proof_verification_time'] = verify_end_time - verify_start_time

        # Get outputs and process
        outputs = wit["outputs"]
        with open(SETTINGS_PATH) as f:
            settings = json.load(f)
        ezkl_outputs = [ezkl.felt_to_float(outputs[0][i], settings["model_output_scales"][0]) for i in range(10)]
        
        # Convert to tensor and decode prediction
        ezkl_witnessed_prediction = torch.tensor(ezkl_outputs)
        predict_decoded = int(torch.argmax(ezkl_witnessed_prediction, dim=-1))

        print("Actual:", json_data.get("label"), "Predicted:", predict_decoded)

    end_time = time.time()
    stats['total_execution_time'] = end_time - start_time

    return {
        "proof": proof,
        "verification": verification_res,
        "Actual Label": json_data.get("label"),
        "Predicted Label": predict_decoded,
        "stats": stats
    }

@app.route('/prove', methods=['POST'])
def prove_task():
    try:
        # Get JSON body
        json_data = request.get_json()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        res = loop.run_until_complete(compute_proof(json_data))
        loop.close()

        return jsonify({'status': 'ok', 'res': res})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == "__main__":
    app.run(port=8000, debug=True)
