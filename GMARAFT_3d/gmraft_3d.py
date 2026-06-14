__author__ = "Semih Tarik Uenal"

import torch
from network_3d.model_old import GMARAFT_Denoiser3D  # adjust import if needed

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model = GMARAFT_Denoiser3D().to(device)
#     model.eval()

#     B, C, D, H, W = 4, 1, 64, 128, 128
#     image1 = torch.randn(B, C, D, H, W).to(device)
#     image2 = torch.randn(B, C, D, H, W).to(device)
#     context_image = torch.randn(B, 3, D, H, W).to(device)

#     # Forward pass
#     with torch.no_grad():
#         flow_preds, context_out = model(image1, image2, context_image, test_mode=False)

#     # Output shapes
#     print(f"Number of flow predictions: {len(flow_preds)}")
#     print(f"Flow shape: {flow_preds[-1].shape}")
#     print(f"Context shape: {context_out.shape}")

# if __name__ == "__main__":
#     main()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GMARAFT_Denoiser3D().to(device)
    model.train()  # <--- Enable training mode

    B, C, D, H, W = 1, 1, 32, 128, 128  # reduce size if debugging GPU memory
    image1 = torch.randn(B, C, D, H, W).to(device)
    image1.requires_grad_()
    image2 = torch.randn(B, C, D, H, W).to(device)
    context_image = torch.randn(B, 3, D, H, W).to(device)

    # Forward pass
    flow_preds, context_out = model(image1, image2, context_image, test_mode=False)

    # Dummy supervision: minimize flow magnitude (just for test)
    final_flow = flow_preds[-1]  # [B, 3, D*4, H*4, W*4]
    loss = (final_flow ** 2).mean()
    print("Dummy loss:", loss.item())

    # Backprop
    loss.backward()
    print("Backpropagation successful — grads exist:", image1.grad is not None)

if __name__ == "__main__":
    main()
