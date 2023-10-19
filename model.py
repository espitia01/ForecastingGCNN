import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import torch
import torch.nn as nn
import torch.optim as optim
from e2cnn import gspaces
from e2cnn import nn as enn

from data import generate_synthetic_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class EquivariantForecastingNet(nn.Module):
    def __init__(self):
        super(EquivariantForecastingNet, self).__init__()
        
        r2_act = gspaces.Rot2dOnR2(N=4)
        self.in_type = enn.FieldType(r2_act, [r2_act.trivial_repr])
        self.out_type = enn.FieldType(r2_act, 4 * [r2_act.regular_repr])
        
        self.layer1 = enn.R2Conv(self.in_type, self.out_type, kernel_size=3, padding=1, bias=True)
        self.relu1 = enn.ReLU(self.out_type)
        
        self.layer2 = enn.R2Conv(self.out_type, self.out_type, kernel_size=3, padding=1, bias=True)
        self.relu2 = enn.ReLU(self.out_type)
        
        self.final_layer = enn.R2Conv(self.out_type, self.in_type, kernel_size=3, padding=1, bias=True)
        
    def forward(self, x):
        x_geom = enn.GeometricTensor(x, self.in_type)
        x_geom = self.layer1(x_geom)
        x_geom = self.relu1(x_geom)
        x_geom = self.layer2(x_geom)
        x_geom = self.relu2(x_geom)
        x_geom = self.final_layer(x_geom)
        return x_geom.tensor

    def iterative_forecasting(self, initial_input, steps):
        forecasts = []
        current_input = initial_input
        
        for _ in range(steps):
            with torch.no_grad():
                forecast = self(current_input)
                forecasts.append(forecast.cpu())
            
            current_input = forecast  # use forecast as next input
        
        return torch.stack(forecasts)
    
def main():
    #Obtain data
    data = generate_synthetic_data(N = 256, L = 100, dt = 0.1, steps = 1000)
    print(np.shape(data))
    num_samples = data.shape[0]
    train_end = int(0.5 * num_samples) 
    valid_end = train_end + int(0.3 * num_samples)

    train_data = data[:train_end]
    valid_data = data[train_end:valid_end]
    test_data = data[valid_end:]
    print(np.shape(train_data), np.shape(valid_data), np.shape(test_data))

    model = EquivariantForecastingNet().to(device)
    criterion = nn.MSELoss()  
    optimizer = optim.Adam(model.parameters())

    num_epochs = 100
    batch_size = 32  

    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for i in range(0, train_end - batch_size, batch_size):
            inputs = train_data[i:i+batch_size].to(device)
            targets = train_data[i+1:i+1+batch_size].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / (train_end // batch_size)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for i in range(0, valid_data.shape[0] - batch_size, batch_size):
                inputs = valid_data[i:i+batch_size].to(device)
                targets = valid_data[i+1:i+1+batch_size].to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_valid_loss += loss.item()
        
        avg_valid_loss = total_valid_loss / (valid_data.shape[0] // batch_size)
        valid_losses.append(avg_valid_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")

    initial_input = test_data[0:1]  # Starting with the first test data
    forecasted_sequence = model.iterative_forecasting(initial_input.to(device), len(test_data) - 1)

    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs. Validation Loss")
    plt.legend()
    plt.show()

    r2conv = model.layer1

    # Expand the kernel weights
    expanded_filter, _ = r2conv.expand_parameters()

    # Visualize the expanded filters
    for i, filter in enumerate(expanded_filter):
        plt.figure(figsize=(6,5))
        im = plt.imshow(filter[0].detach().cpu().numpy(), cmap='gray')  # Visualize the first input channel of each filter
        plt.colorbar(im, orientation='vertical')
        plt.title(f"Filter {i+1}")
        plt.tight_layout()
        plt.show()
    
    from matplotlib.animation import FuncAnimation

    prediction = [timestep.squeeze() for timestep in forecasted_sequence]
    actual_data = [data.squeeze() for data in test_data]

    n_steps = min(len(actual_data), len(prediction))

    # Update function for the animation
    def update(num, img1, img2, img3, forecasted_array, valid_array):
        img1.set_data(valid_array[num])
        img2.set_data(forecasted_array[num])
        difference = valid_array[num] - forecasted_array[num]
        img3.set_data(difference)
        return [img1, img2, img3]

    # Set up the figure and axis
    fig, axarr = plt.subplots(1, 3, figsize=(15,5))

    # Setting up initial display
    img1 = axarr[0].imshow(actual_data[0], animated=True, cmap='gray')
    axarr[0].set_title('Actual Data')
    img2 = axarr[1].imshow(prediction[0], animated=True, cmap='gray')
    axarr[1].set_title('Forecasted Data')
    img3 = axarr[2].imshow(actual_data[0] - prediction[0], animated=True, cmap='gray')
    axarr[2].set_title('Difference')

    # Colorbar setup
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(img3, cax=cbar_ax)

    # Create the animation
    ani = FuncAnimation(fig, update, frames=range(n_steps), fargs=(img1, img2, img3, prediction, actual_data), blit=True)

    # Save the animation
    ani.save('forecast_animation.mp4', writer='ffmpeg', fps=15)

    plt.show()

    pred_values_10_10 = []  
    actual_values_10_10 = []  

    for i in range(len(prediction)):

        pred_value = prediction[i][190][100]
        actual_value = actual_data[i][190][100]
        

        pred_values_10_10.append(pred_value)
        actual_values_10_10.append(actual_value)
    print(pred_values_10_10[0])

    timesteps = list(range(len(np.array(pred_values_10_10))//12)) 

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(timesteps, pred_values_10_10[:16], label='Predicted Values at [10][10]', marker='o')
    plt.plot(timesteps, actual_values_10_10[:16], label='Actual Values at [10][10]', marker='x')

    # Add labels and title
    plt.xlabel('Timestep')
    plt.ylabel('Value at [10][10]')
    plt.title('Comparison of Predicted and Actual Values at [10][10]')
    plt.legend()

    # Show the plot
    plt.grid()
    plt.show()
    
if __name__ == "__main__":
    main()