# Artificial Causal Inference
General framework for Artificial Causal Inference on real world scientific experiments

## Datasets

<table align="center">
  <tr>
    <th>Field</th>
    <th>Research</th>
    <th>Subject</th>
    <th>Version</th>
    <th>T<sup>1</sup></th>
    <th>Sample Size<sup>2</sup></th>
    <th>Effect Modifiers<sup>2</sup></th>
    <th>Annotations</th>
    <th>Description</th>
    <th>Preview</th>
    <th>Source</th>
  </tr>
  <tr>
    <td rowspan="5">Experimental Ecology</td> 
    <td rowspan="5">Social Immunity</td> 
    <td><code>ants</code></td> 
    <td><code>v1</code></td> 
    <td>3</td> 
    <td>14, 15, 15 (x2)</td> 
    <td>None</td> 
    <td>Grooming Y2F, B2F (full)</td> 
    <td>Ants triplets interactions (30min/video)</td> 
    <td><img src="preview/ants/v1/demo/clip.gif" alt="GIF 1" width="100" height="100"></td>
    <td><a href="https://figshare.com/account/items/28319693/edit">figshare</a></td> 
  </tr>
  <tr>
    <td><code>ants</code></td> 
    <td><code>v2</code></td>
    <td>2</td> 
    <td>24, 20 (x2)</td> 
    <td>None</td> 
    <td>Grooming Y2F, B2F (full)</td> 
    <td>Ants triplets interactions (10min/video)</td> 
    <td><img src="preview/ants/v2/demo/clip.gif" alt="GIF 2" width="100" height="100"></td>
    <td><a href="https://figshare.com/articles/dataset/ISTAnt_zip/26484934">figshare</a></td> 
  </tr>
  <tr>
    <td><code>ants</code></td> 
    <td><code>v3</code></td>
    <td>6</td> 
    <td>38,33,34,34,35,38 (x2)</td> 
    <td>None</td> 
    <td>Grooming Y2F, B2F (full), FOL, YOL, BOL</td> 
    <td>Ants triplets interactions (10min/video)</td> 
    <td><img src="preview/ants/v3/demo/clip.gif" alt="GIF 2" width="100" height="100"></td>
    <td>To be released</td> 
  </tr>
  <tr>
    <td><code>ants</code></td> 
    <td><code>v4</code></td>
    <td>6</td> 
    <td>20, 20, 20, 19, 14, 20 (x2)</td> 
    <td>None</td> 
    <td>Grooming Y2F, B2F (full), FOL, YOL, BOL</td> 
    <td>Ants triplets interactions (10min/video)</td> 
    <td><img src="preview/ants/v4/demo/clip.gif" alt="GIF 2" width="100" height="100"></td>
    <td>To be released</td> 
  </tr>
  <tr>
    <td><code>ants</code></td> 
    <td><code>v5</code></td>
    <td></td> 
    <td></td> 
    <td>None</td> 
    <td>None</td> 
    <td>Ants triplets interactions (10min/video)</td> 
    <td>...</td>
    <td>...</td>
  </tr>
  <tr>
    <td rowspan="2">Neuroscience</td> 
    <td rowspan="2">Autism</td>
    <td><code>mice</code></td> 
    <td><code>v1</code></td>
    <td>3+3</td> 
    <td>12+12,12+12,12+12 (x6, x12)</td> 
    <td>Sex</td> 
    <td>Sniffing NN, NT x subjects (half) </td> 
    <td>Mice quadruplets interactions (15-30 min/video)</td> 
    <td><img src="preview/mice/v1/demo/clip.gif" alt="GIF 3" width="100" height="100"></td>
    <td>Cleaning</td>
  </tr>
  <tr>
    <td><code>mice</code></td> 
    <td><code>v2</code></td>
    <td>3</td> 
    <td>12, 12, 12</td> 
    <td>Sex</td> 
    <td>None</td> 
    <td>Mixed (1-3) mice quadruplets interactions (15-30 min/video)</td> 
    <td><img src="preview/mice/v1/demo/clip.gif" alt="GIF 3" width="100" height="100"></td>
    <td>Cleaning</td>
  </tr>
  <tr>
    <td>Biology</td> 
    <td>...</td>
    <td><code>frogs</code></td> 
    <td><code>v1</code></td> 
    <td>...</td> 
    <td>...</td> 
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
  </tr>
  <tr>
    <td>Medicine</td> 
    <td>Cancer</td>
    <td><code>brain</code></td>  
    <td><code>v1</code></td>
    <td>...</td> 
    <td>...</td> 
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td><img src="preview/brain/v1/demo/brain.png" alt="image 1" width="150" height="100"></td>
    <td>Ready but private</td> 
  </tr>
  <tr>
    <td>Chemistry</td> 
    <td>...</td>
    <td><code>cells</code></td> 
    <td><code>v1</code></td> 
    <td>...</td> 
    <td>...</td> 
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td> 
  </tr>
</table>

<sup>1</sup> number of treatments including the control<br>
<sup>2</sup> sample size per treatment (simmetry factor with multiple individuals per observation)<br>
<sup>2</sup> effect modifiers of interest, e.g., for CATE estimation


## Prediction Powered Causal Inference

...

#### Reference
```bibtex
@inproceedings{cadei2025prediction,
  title={Prediction-Powered Causal Inferences},
  author={Cadei, Riccardo and Demirel, Ilker and De Bartolomeis, Piersilvio and Lindorfer, Lukas and Cremer, Sylvia and Schmid, Cordelia and Locatello, Francesco},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
```

## Exploratory Causal Inference
...

#### Reference

```bibtex
@article{mencattini2025exploratory,
  title={Exploratory Causal Inference in SAEnce},
  author={Mencattini, Tommaso and Cadei, Riccardo and Locatello, Francesco},
  journal={arXiv preprint arXiv:2510.14073},
  year={2025}
}
```

