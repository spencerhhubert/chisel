# chisel

## CAD Copilot - Generate meshes/STLs from textual prompts

Chisel uses a similar architecture to Stable Diffusion where a transformer UNet neural network is used to learn to undo gaussian noise added an image, except instead of RGB values in an image, we consider the 3D coordinate values that comprise a 3D mesh.

A mesh, often seen in the STL file format, are 3D objects defined by listed of vertices connected by faces, that are themselves each defined in terms of three vertices. This is equivalent to a hypergraph where each hyperedge is a face, and we use [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) for learning on these hypergraphs.

<img src="https://raw.githubusercontent.com/spencerhhubert/chisel/main/assets/undo_noise.gif" width=300 alt="Undoing the noise on a 3D mesh">

*this is not a product of the model, this is an manual algorithmically generated example of how it works

The initial training is being completed with the [ABC Dataset](https://deep-geometry.github.io/abc-dataset/).