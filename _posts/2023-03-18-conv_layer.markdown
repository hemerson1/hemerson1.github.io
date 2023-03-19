---
layout: post
title:  "Convolutional Layer from Scratch"
date:   2023-03-18 13:24:43 +0000
categories: Machine Learning 
---

Convolutional layers have been a staple of image recognition since their inception in the 1990s. This post briefly summarises the mathematics underlying their functionality and provides implementation code in Python.

### Overview: 

A **convolutional layer** is a specialised architecture capable of more easily encoding image-specific features, such as spatial dependencies between pixels. To capture this information, each image tensor is convolved with a **kernel**, which performs an element-wise product for each overlapping region and sums the result. Typically, multiple kernels are utilised which creates an arbitrary number of features maps, each describing different characteristics of the image. 

![Convolution](/assets/images/convolution-2023-03-19.png "This is a caption"){:width="500" .centre-image}

<figcaption class="fig-caption"> <b>Figure:</b> A convolution of a single channel image tensor with a kernel and the resulting feature map. The image was taken from <a href="https://insightsimaging.springeropen.com/articles/10.1007/s13244-018-0639-9">Yamashita et al</a>.</figcaption>

To train the layer, the parameters of the kernel are updated via back-propagation, in the same way as the weights of a standard linear layer. In the current formulation, an image can only be convolved a finite number of times as with each transformation the resulting feature map shrinks. This can be addressed by **padding** the input to the layer with additional rows and columns. Furthermore, the step size or **stride** of the kernel can also be modified to reduce the overlap of successive features and hence reduce the size of the feature map. 

### Mathematics:

A mathematical description of the convolutional layer is described as follows. Suppose the input to the $$l$$th convolutional layer in a series of layers has the dimensionality $$(H^{l} \times W^{l} \times D^{l})$$, where $$H^{l}$$, $$W^{l}$$ and $$D^{l}$$ are the height, width and number of channels in the input, $$x^{l}$$. If the convolutional layer is composed of $$D$$ kernels, then the combination of all kernels, $$\textbf{f}$$ is a four-tensor in $$\mathbf{R}^{H \times W \times D^{l} \times D}$$. For the case of stride 1 and no padding, the output of the convolutional layer, $$\textbf{y}$$ is given by:
\\[y_{ i^{l+1}, j^{l+1}, d} = \sum_{i=0}^{H} \sum_{j=0}^{W} \sum_{d^{l}=0}^{D^{l}} f_{i, j, d^{l}, d} \times x^{l}_{i^{l+1}+i, j^{l+1}+j, d^{l},}\\] where the index variables  $$0 \le i < H$$, $$0 \le j < W$$, $$0 \le d^{l} < D^{l}$$ and $$0 \le d < D$$ denote specific elements within the kernel. A bias term is also typically included, however this has been omitted from the notation. For more information refer to [Wu](https://cs.nju.edu.cn/wujx/paper/CNN.pdf).

### Implementation:

A re-implementation of the standard PyTorch convolutional layer can be observed [here](https://github.com/hemerson1/Blog-Code/blob/main/Machine%20Learning/conv_net.ipynb). To avoid using a series of ```for``` loops, this work utilises ```torch.nn.functional.unfold``` to extract kernel-sized sliding windows of the input tensor as shown below: 

```python
def forward(self, input):

        # Input: (batch, input_size, height, width)
        H_in, W_in = input.shape[-2], input.shape[-1]
        H_out, W_out = self._calc_output(H_in, W_in)

        # break into patches ready for convolution
        patches = F.unfold(
            input, kernel_size=(self.kernel_size, self.kernel_size),
            stride=self.stride, padding=self.padding)
        patches = patches.transpose(1, 2)

        # transpose weights and perform multiplication
        w = self.weights.view(self.weights.size(0), -1).T
        output = torch.matmul(patches, w).transpose(1, 2)
        output = output.view(output.shape[0], self.weights.shape[0], H_out, W_out)
        output += self.biases.reshape(1, -1, 1, 1)

        return output
```
With the default hyperparameters, the algorithm achieves $$\sim 97\%$$ accuracy on the MNIST dataset which is comparable performance to the in-built Conv2d layer with the same settings.