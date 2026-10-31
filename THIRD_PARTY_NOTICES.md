# Third-party attribution

The global-feature regression path and checkpoint key layout in
`src/newkcat/model.py` follow the official CataPro `KcatModel`:
<https://github.com/zchwang/CataPro/blob/master/inference/model.py>.
The dataset/cache importer targets the official CataPro file formats:
<https://github.com/zchwang/CataPro/blob/master/inference/generate_features.py>.
The upstream work is *Robust enzyme discovery and engineering with deep learning
using CataPro*, Nature Communications (2025),
<https://doi.org/10.1038/s41467-025-58038-4>.

Upstream CataPro MIT notice, retained here for the compatible implementation:

> MIT License
>
> Copyright (c) 2024 zchwang
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

PyTorch, RDKit, NumPy, and the optional pandas dependency are installed separately
and retain their respective licenses. This repository does not distribute their
code, CataPro data, pretrained language-model weights, GNN-SOM weights, or a
licensed reaction database. Users supply their own permitted data and weights.
