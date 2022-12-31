#Mesh -> Data (mesh -> hypergraph)
def makeHyperIncidenceMatrix(mesh):
    faces = mesh.face.t()
    for i,face in enumerate(faces):
        idx = torch.tensor([i]*len(face))
        hyperedge = torch.stack((face,idx),dim=0)
        if not 'out' in locals():
            out = hyperedge
        else:
            out = torch.cat((out,hyperedge),dim=-1)
    return out

def makeFaceTensor(incidenceMat):
    #as is the edges never even change so this isn't necessary but if the model gets more complex to account for clipping edges then perhaps it'll modify edges and then we'll need this
    return None

def applyNoise(x,distribution):
    return x + (torch.randn(x.shape) * (distribution ** 0.5))


