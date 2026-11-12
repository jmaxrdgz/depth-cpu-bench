# LNCA depth reconstruction

Source: Nunn and Nikoubin, [Efficient Monocular Depth Estimation on Embedded
Systems with Neural Cellular Automata](https://link.springer.com/article/10.1007/s42979-026-05327-4),
Methodology: LNCA architecture; Fig. 4; Task generality.

No public implementation was found in the article or targeted code searches.
`lnca-reimplementation` is our reconstruction, not author code or a validated
reproduction. No trained weights are available here.

The model uses 32 latent channels, three stride-two convolutions, four shared
residual NCA updates, PixelShuffle(8) decoding, and exponentiated log-depth.

| Component | Parameters |
| --- | ---: |
| Encoder | 19,392 |
| Perception | 288 |
| Update network | 2,656 |
| Decoder | 18,496 |
| Total | 40,832 |

These counts match the paper. We infer encoder 3×3 kernels and bias choices from
the component counts. Padding is zero, stride-two convolutions use padding one,
and updates are deterministic. Author padding, initialization and exact inference
preprocessing remain unverified. The article also mentions a hidden dimension of
16, which is not reconciled with its stated update structure/count. We follow
that explicit structure/count. The joint segmentation variant is not implemented.

Timing includes exponentiation, without metric clipping. At 192×640, the latent
grid is 24×80. Results use our common PyTorch protocol, not the paper's 160×120
ONNX Runtime smartphone deployment. Parameter agreement does not establish
operator equivalence, accuracy or reproduction of reported latency.
