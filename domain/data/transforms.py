import torch


class SimpleFreqSpace(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, img):
        return torch.fft.rfft2(img)


class SimpleComplex2Vec(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        n, m = x.shape[-2], x.shape[-1]
        return torch.cat(
            [
                torch.stack(
                    [
                        torch.cat(
                            [
                                x[:, : n // 2 + 1, 0:1].real,
                                x[:, 1 : (n + 1) // 2, 0:1].imag,
                            ],
                            -2,
                        ),
                        torch.cat(
                            [
                                x[:, : n // 2 + 1, m - 1 : m].real,
                                x[:, 1 : (n + 1) // 2, m - 1 : m].imag,
                            ],
                            -2,
                        ),
                    ],
                    dim=3,
                ),
                torch.view_as_real(x[:, :, 1:-1]),
            ],
            dim=2,
        )
