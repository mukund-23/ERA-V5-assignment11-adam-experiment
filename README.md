# Adam Optimizer Experiments

This repository contains a notebook exploring a few practical aspects of the Adam optimizer:

* manual implementation of Adam and comparison with PyTorch
* effect of Adam bias correction
* update-to-weight ratios during learning-rate warmup
* cosine decay vs WSD learning-rate schedules
* relationship between learning rate and model width

The main file is `adam-optimizer.ipynb`.

## Setup

The experiments use:

* Python 3
* PyTorch
* NumPy
* Matplotlib

The training experiments use a small GPT-style character-level language model trained on the Tiny Shakespeare dataset.

The notebook automatically downloads the dataset if `input.txt` is not already present.

The model uses:

* 2 transformer blocks
* 4 attention heads
* context length of 128
* batch size of 32

For the training experiments, the data is split 90/10 into training and validation sets.

The notebook was run on CUDA.

## Part 1: Adam by hand vs PyTorch

The first experiment implements the Adam update manually and compares it against PyTorch's `torch.optim.Adam`.

The parameters used are:

```text
learning rate = 1e-3
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
```

The gradient sequence is:

```text
[0.5, -0.3, 0.8, 0.1, -0.6]
```

For each step, the notebook compares:

* first moment `m`
* second moment `v`
* bias-corrected first moment `m_hat`
* bias-corrected second moment `v_hat`
* parameter update
* updated parameter value

The largest absolute difference between the manual implementation and PyTorch was:

```text
5.128276275856436e-17
```

This confirms that the manual implementation matches PyTorch to numerical precision.

## Part 2: Bias correction

The second experiment compares Adam with and without bias correction.

The notebook first generates 20 gradients using a fixed random seed and plots the update sizes for both cases.

It then looks at how quickly the effect of bias correction becomes negligible.

Ignoring epsilon, the ratio between the update without bias correction and the update with bias correction is:

$$
\frac{1-\beta_1^t}{\sqrt{1-\beta_2^t}}
$$

For the Adam parameters used above, the analytical calculation shows that the difference falls below 1% at:

```text
step 3925
```

A simulation using a longer gradient sequence gives the same result:

```text
step 3925
```

So although the bias-correction effect becomes smaller over time, it takes a few thousand steps for the difference to fall below the 1% threshold used here.

## Part 3: Update-to-weight ratio during warmup

The third experiment looks at the quantity

$$
\frac{\|\Delta W\|}{\|W\|}
$$

for each weight matrix in the model.

The model width is 256 and training runs for 300 steps.

The learning rate uses linear warmup for the first 100 steps and then remains constant.

The ratio is tracked separately for the weight matrices of the model. LayerNorm parameters are excluded because their biases start at zero, making the ratio undefined.

A settling criterion is also applied to the ratios. The criterion uses a smoothed ratio and checks whether it stays within 5% of the mean ratio over the final 100 training steps.

The resulting settling steps were:

| Layer                    | Settling step |
| ------------------------ | ------------: |
| `tok.weight`             |   Not settled |
| `pos.weight`             |   Not settled |
| `blocks.0.qkv.weight`    |   Not settled |
| `blocks.0.proj.weight`   |           196 |
| `blocks.0.fc.weight`     |           197 |
| `blocks.0.fc_out.weight` |           286 |
| `blocks.1.qkv.weight`    |           260 |
| `blocks.1.proj.weight`   |   Not settled |
| `blocks.1.fc.weight`     |   Not settled |
| `blocks.1.fc_out.weight` |           200 |
| `head.weight`            |           225 |

Not all layers met the settling criterion within the 300-step run. This is expected from the definition of the criterion and should not be interpreted as meaning that the corresponding layers never stabilize.

## Part 4: Cosine vs WSD

The fourth experiment compares two learning-rate schedules:

### Cosine

* 100-step linear warmup
* cosine decay after warmup
* final learning rate is 10% of the peak learning rate

### WSD

* 100-step linear warmup
* stable learning rate
* linear decay during the final 20% of the schedule
* final learning rate is 10% of the peak learning rate

The schedules are planned over 300 steps, while the comparison is made after 200 steps.

The peak learning rate is tuned independently for each schedule using the validation loss after 200 steps.

The best peak learning rates found were:

| Schedule |  Peak LR | Validation loss |
| -------- | -------: | --------------: |
| Cosine   | 5.88e-03 |          1.9853 |
| WSD      | 5.88e-03 |          2.0098 |

Using the independently selected learning rates, the final comparison at step 200 was:

| Schedule |  Peak LR | LR at step 200 | Train loss, last 10 avg | Validation loss |
| -------- | -------: | -------------: | ----------------------: | --------------: |
| Cosine   | 5.88e-03 |       3.28e-03 |                  1.9000 |          1.9771 |
| WSD      | 5.88e-03 |       5.88e-03 |                  1.9127 |          2.0095 |

Cosine had the lower validation loss at step 200:

```text
Cosine: 1.9771
WSD:    2.0095
```

and therefore performed better in this particular experiment.

## Part 5: Learning rate vs model width

The final experiment investigates how the best tested learning rate changes with model width.

The widths tested were:

```text
256
512
1024
```

For each width, six peak learning rates were tested using the cosine schedule:

```text
1.0e-04
3.1e-04
9.8e-04
3.1e-03
9.6e-03
3.0e-02
```

The validation losses from the sweep were:

| Width | 1.0e-04 | 3.1e-04 | 9.8e-04 | 3.1e-03 | 9.6e-03 | 3.0e-02 |
| ----: | ------: | ------: | ------: | ------: | ------: | ------: |
|   256 |  2.6336 |  2.4861 |  2.2113 |  1.9283 |  1.8526 |  2.0909 |
|   512 |  2.4987 |  2.2696 |  1.9771 |  1.8265 |  2.0921 |  2.5412 |
|  1024 |  2.3266 |  2.0119 |  1.8659 |  1.8647 |  2.4945 |  2.8116 |

The best learning rate among the tested values for each width was:

| Width | Best tested LR |
| ----: | -------------: |
|   256 |        9.6e-03 |
|   512 |        3.1e-03 |
|  1024 |        3.1e-03 |

A log-log fit of the best tested learning rate against model width gives a slope of:

```text
-0.82
```

Using this fitted relationship to extrapolate to width 4096 gives:

```text
8.1e-04
```

The `-0.82` value should be treated as an empirical estimate from this sweep rather than a precise scaling law. Only three model widths were tested, and the learning-rate grid is relatively coarse.

## Reproducibility

The training function resets the PyTorch random seed to `0` before each run so that experiments use the same model initialization and training batches.

The validation batches are also fixed using a separate random generator with seed `1`.

This makes comparisons between learning rates, schedules, and model widths more consistent.

## Outputs

The notebook saves the generated plots under the `plots/` directory:

```text
plots/
├── part2_bias_correction.png
├── part3_update_ratio.png
├── part4_lr_tuning.png
├── part4_lr_schedules.png
├── part4_training_loss.png
└── part5_lr_sweep.png
```

## Running the notebook

Open `adam-optimizer.ipynb` in Jupyter or another compatible notebook environment and run the cells from top to bottom.

The notebook creates the `plots/` directory automatically and downloads the Tiny Shakespeare dataset if required.
