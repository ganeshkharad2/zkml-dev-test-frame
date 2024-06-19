import requests
from PIL import Image
import io
import random
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import json
import numpy as np
import os 
import csv
import time

API_URL = 'http://127.0.0.1:8000/prove'  # Update with your Flask server URL
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set up the MNIST test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(root='./test', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # Convolutional encoder
        self.conv1 = nn.Conv2d(1, 6, 5)  # 1 input channel, 6 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 input channels, 16 output channels, 5x5 kernel

        # Fully connected layers / Dense block
        self.fc1 = nn.Linear(16 * 4 * 4, 120) 
        self.fc2 = nn.Linear(120, 84)         # 120 inputs, 84 outputs
        self.fc3 = nn.Linear(84, 10)          # 84 inputs, 10 outputs (number of classes)

    def forward(self, x):
        # Convolutional block
        x = F.avg_pool2d(F.sigmoid(self.conv1(x)), (2, 2)) # Convolution -> Sigmoid -> Avg Pool
        x = F.avg_pool2d(F.sigmoid(self.conv2(x)), (2, 2)) # Convolution -> Sigmoid -> Avg Pool

        # Flattening
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)  # No activation function here, will use CrossEntropyLoss later
        return x


def save_results_to_csv(data):
        
    # Define the CSV file path
    csv_file_path = 'zkp_execution_results.csv'

    # Check if the file exists
    file_exists = os.path.isfile(csv_file_path)

    # Define the headers for the CSV file
    headers = [
        'actual_label',"torch_model_prediction", 'ZKML_preredicted_label', 'proof', 'verification', 
        'proof_generation_time', 'proof_verification_time', 
        'total_execution_time', 'witness_generation_time',
        "torch_model_predict_time"
    ]

    # Open the file in append mode
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        
        # Write the header only if the file does not already exist
        if not file_exists:
            writer.writeheader()
        
        # Extract the row data from the dictionary
        row = {
            'actual_label': data['res']['Actual Label'],
            "torch_model_prediction": data["torch_model_prediction"],            
            'ZKML_preredicted_label': data['res']['Predicted Label'],
            'proof': data['res']['proof'],
            'verification': data['res']['verification'],
            'proof_generation_time': data['res']['stats']['proof_generation_time'],
            'proof_verification_time': data['res']['stats']['proof_verification_time'],
            'total_execution_time': data['res']['stats']['total_execution_time'],
            'witness_generation_time': data['res']['stats']['witness_generation_time'],
            "torch_model_predict_time" : data["torch_model_predict_time"]
        }
        
        # Write the row to the CSV file
        writer.writerow(row)
        print("results saved to csv")



def get_original_model_pred(input_data):
    torch_model_predict_start_time = time.time()
    model = torch.load("models/mnist_0.979.pkl")
    model.eval()
    prediction = model(input_data.float())
    torch_model_prediction = torch.argmax(prediction, dim=-1)
    torch_model_predict_end_time = time.time()
    torch_model_predict_time= torch_model_predict_end_time - torch_model_predict_start_time
    
    return int(torch_model_prediction), torch_model_predict_time

def test_api():
    for idx, (test_x, test_label) in enumerate(test_loader):
        if idx >= 210:
            break
        test_label =int(test_label)
        print("==========", idx)
        # print("True Label is: ", test_label)
        
        test_x = test_x.to(device)
         # normalize the image to 0 or 1 to reflect the inputs from the drawing board
        test_x = test_x.round()

        torch_model_prediction, torch_model_predict_time = get_original_model_pred(test_x)
        
        train_data_point = test_x.unsqueeze(0) # Add a batch dimension
        train_data_point = train_data_point.to(device)
        
        # Convert the tensor to numpy array and reshape it for JSON serialization
        x = train_data_point.cpu().detach().numpy().reshape([-1]).tolist()
        
        data = {
            'input':{'input_data': [x]},
            'label': test_label
        }

        # Serialize the data to a JSON string
        data_json = json.dumps(data)
        
        response = requests.post(
            API_URL,
            data=data_json,
            headers={'Content-Type': 'application/json'}
        )    
        

        if response.status_code == 200:
            print(f"Request {idx + 1}: Success")
            response_data = response.json()
            response_data["torch_model_prediction"] = torch_model_prediction
            response_data["torch_model_predict_time"] = torch_model_predict_time
            print(f"Response: {response_data}")
            save_results_to_csv(response_data)

        else:
            print(f"Request {idx + 1}: Failed with status code {response.status_code}")
            print(f"Response: {response.text}")
    
    
if __name__ == "__main__":
    test_api()
