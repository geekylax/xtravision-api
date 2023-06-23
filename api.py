import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F

app = FastAPI()
class DNN(nn.Module):
    
    def __init__(self):
        super(DNN, self).__init__()
        # We will create a simple neural network with 2 fully connected layers for our use case
        self.fc1 = nn.Linear(99, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.drop1 = nn.Dropout(p=0.4)
        self.fc4 = nn.Linear(128, 28) # 8 classes

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = F.relu(self.fc3(x))
        x = self.drop1(x)
        x = self.fc4(x)
        return x


class InputData(BaseModel):
    data: list[float]  # Changed field name from 'input' to 'data'

# Define a helper class for lazy loading and scaling the model
class ModelHandler:
    def __init__(self):
        self.model = None
    
    def load_model(self):
        if self.model is None:
            # Load the trained model
            self.model=DNN()
            self.model = torch.load("untrained_model.pt")
            self.model.eval()
    
        # Reshape the input data to the desired format
    def preprocess_input(self, input_data):
        # Preprocess the input data
        print(len(input_data))
        if len(input_data) == 100:
            input_data=input_data[:-1]
            
        preprocessed_data = torch.tensor(input_data, dtype=torch.float32)
        preprocessed_data = torch.unsqueeze(preprocessed_data, 0)

        return preprocessed_data
    
    def predict(self, input_data):
        # Convert the input data to a tensor
        

        # Make predictions using the model
        with torch.no_grad():
            output = self.model(input_data)
            predictions = output.argmax(dim=1).tolist()
        
        return predictions

# Create an instance of the model handler
model_handler = ModelHandler()

# Create a POST endpoint for generating predictions
@app.post("/predict")
def predict(input_data: InputData):
    # lazy load the model 
    model_handler.load_model()

    
    # Preprocess the input data
    preprocessed_input = model_handler.preprocess_input(input_data.input)
    
    # Make predictions using the model handler
    predictions = model_handler.predict(preprocessed_input)
    
    return {"predictions": predictions}

# Run the API server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80,factory=True)
