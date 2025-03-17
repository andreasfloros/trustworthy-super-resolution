# [Trustworthy Image Super-Resolution via Generative Pseudoinverse](https://openreview.net/forum?id=UsYosqkokz)
Andreas Floros, Seyed-Mohsen Moosavi-Dezfooli, Pier Luigi Dragotti

## Abstract
We consider the problem of trustworthy image restoration, taking the form of a constrained optimization over the prior density. To this end, we develop generative models for the task of image super-resolution that respect the degradation process and that can be made asymptotically consistent with the low-resolution measurements, outperforming existing methods by a large margin in that respect.

<div align="center">
  <table>
    <tr>
      <td align="center">
        <b>Low-resolution</b><br>
        <img src="assets/lr.png" />
      </td>
      <td align="center">
        <b>Super-resolved</b><br>
        <img src="assets/sr.png" />
      </td>
      <td align="center">
        <b>High-resolution</b><br>
        <img src="assets/hr.png" />
      </td>
    </tr>
  </table>
</div>

## Setup
### Requirements
- [PyTorch](https://pytorch.org/get-started/locally/)
- [FFHQ](https://www.kaggle.com/datasets/potatohd404/ffhq-128-70k) (dev) and [CelebA-HQ](https://www.kaggle.com/datasets/denislukovnikov/celebahq256-images-only) (eval) for 16x16->128x128 face super-resolution

### Training
- First run `python train_flow.py`. This will train the normalizing flow according to Algorithm 1 of the paper
- Once the flow is trained, run `python train_dpm.py` with a link to your flow checkpoint. This is Algorithm 2 in the paper

### Evaluation
- Run `python eval.py` with the final checkpoint. We provide our trained model [here](https://github.com/andreasfloros/trustworthy-super-resolution/releases/tag/ffhq128-8x)

## Citation
```BibTex
@inproceedings{
  floros2025trustworthy,
  title={{Trustworthy Image Super-Resolution via Generative Pseudoinverse}},
  author={Andreas Floros and Seyed-Mohsen Moosavi-Dezfooli and Pier Luigi Dragotti},
  booktitle={ICLR 2025 Workshop on Deep Generative Model in Machine Learning: Theory, Principle and Efficacy},
  year={2025},
  url={https://openreview.net/forum?id=UsYosqkokz}
}
```
