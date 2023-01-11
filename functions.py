import torch

#Mesh -> Data (mesh -> hypergraph)
def makeHyperIncidenceMatrix(mesh):
    faces = mesh.face.t()
    for i,face in enumerate(faces):
        idx = torch.tensor([i]*len(face),device=mesh.pos.device)
        hyperedge = torch.stack((face,idx),dim=0)
        if not 'out' in locals():
            out = hyperedge
        else:
            out = torch.cat((out,hyperedge),dim=-1)
    return out

#get the point in space that is the average of the points that make up the face
def makeHyperEdgeFeatures(x, incidence_matrix):
    out = torch.zeros((incidence_matrix.shape[1],x.shape[1]),device=x.device)
    for i in range(incidence_matrix.shape[1]):
        out[i] = torch.mean(x[incidence_matrix[0,i]],dim=0)
    return out

def makeFaceTensor(incidenceMat):
    #as is the edges never even change so this isn't necessary but if the model gets more complex to account for clipping edges then perhaps it'll modify edges and then we'll need this
    return None

def applyNoise(x,distribution):
    return x + (torch.randn(x.shape,device=x.device) * (distribution ** 0.5))


