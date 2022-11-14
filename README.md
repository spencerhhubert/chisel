# chisel

Diffusion is a generative AI process by which we can take a set of data and generate more stuff that belongs in that set.

It works by adding noise to the data in steps and learning to undo the noise at each step.

This has been almost exclusively applied to images.

Seems like it should work for 3D data as well. Like 3D meshes where we can define a shape as being a set of triangular-faces that all connect, and a face being a three points in space that all connects.

These meshes can be described as regular old graphs. I happened upon PyTorch Geometric, which is a treasure of useful functions for dealing with graphs and neural networks.

Hopefully we can generate good 3D cad assets like gears and stuff soon!
Really need a dataset that connects words -> assets lol

