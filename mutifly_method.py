class ComputationGraph(object):
    
    def forward(self,inputs):
        # [pass inputs to input gates...]
        # forward the computational graph:
        for gate in self.graph.nodes_topologically_sorted():
            gate.forward()
#        return loss # the final gat
class MutiplyGate(object):
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z #local gradient
    def forward(self): #forward 실행때 local 메모리RAM에 채워놓고
        z = self.x*self.y
        return z
    def backward(self,dz): #메모리 소진
        dx = self.y*dz # [dz/dx X dL/dz] = dL/dx
        dy = self.x*dz # [dz/dy X dL/dz] = dL/dy
        return[dx,dy]