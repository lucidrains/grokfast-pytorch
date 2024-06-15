
def test_grokfast_adam():
    # toy model

    import torch
    from torch import nn

    # device = torch.device("mps")

    model = nn.Linear(10, 1)

    # import GrokFastAdamW and instantiate with parameters

    from grokfast_pytorch import GrokFastAdamW

    opt = GrokFastAdamW(model.parameters(), lr = 1e-4)

    # forward and backwards

    loss = model(torch.randn(10))
    loss.backward()

    # optimizer step

    opt.step()
    opt.zero_grad()
