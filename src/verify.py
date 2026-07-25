import torch
from train import BNN, binarize
from torchvision import datasets, transforms
import os
import json

def verify():
    device = torch.device("cpu")
    model = BNN().to(device)

    if os.path.exists('models/bnn_mnist.pth'):
        model.load_state_dict(torch.load('models/bnn_mnist.pth', map_location=device))
        print("Loaded trained model for verification.")
    else:
        print("No trained model found. Using random initialized weights to generate testbench data.")

    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    print("Testing 10 images against PyTorch model to save expected results...")

    os.makedirs('tests/data', exist_ok=True)

    with open('tests/data/expected.json', 'w') as f:
        results = []
        for i in range(10):
            img, target = test_dataset[i]
            img = img.unsqueeze(0).to(device)
            out = model(img)
            pred = out.argmax(dim=1).item()

            bin_img = binarize(img).detach().numpy().flatten()
            bin_str = "".join(['1' if x > 0 else '0' for x in bin_img])

            with open(f'tests/data/img_{i}.mem', 'w') as img_f:
                for bit in bin_str:
                    img_f.write(bit + "\n")

            results.append({"index": i, "target": target, "pred": pred})

        json.dump(results, f, indent=4)

    print("Verification data generated successfully.")

if __name__ == "__main__":
    verify()
