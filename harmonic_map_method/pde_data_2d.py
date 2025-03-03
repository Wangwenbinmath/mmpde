from fealpy.backend import backend_manager as bm
from sympy import *

class Poissondata2d():
    def __init__(self , u :str ,x : str ,y : str, D = [0,1,0,1]):
        u = sympify(u)
        self.u = lambdify([x,y], u ,'numpy')
        f_str = -diff(u,x,2) - diff(u,y,2)
        self.f = lambdify([x,y], f_str)
        self.grad_ux = lambdify([x,y], diff(u,x,1))
        self.grad_uy = lambdify([x,y], diff(u,y,1))
        self.domain = D

    def domain(self):
        return self.domain
    
    def solution(self, p):
        x = p[...,0]
        y = p[...,1]

        return self.u(x,y)
    
    def source(self,p, index ,):
        x = p[...,0][index]
        y = p[...,1][index]
        return  self.f(x,y)

    def gradient(self,p):
        x = p[...,0]
        y = p[...,1]
        val = bm.zeros_like(p)
        val[...,0] = self.grad_ux(x,y)
        val[...,1] = self.grad_uy(x,y)
        return val
    
    def dirichlet(self,p ):
        return self.solution(p)

class Convectdiffdata2d():
    def __init__(self , u :str ,x : str ,y : str, a , b , D = [0,1,0,1]):
        u = sympify(u)
        self.u = lambdify([x,y], u ,'numpy')
        f_str = b*(-diff(u,x,2) - diff(u,y,2)) + a[0]*diff(u,x,1) + a[1]*diff(u,y,1)
        self.f = lambdify([x,y], f_str)
        self.grad_ux = lambdify([x,y], diff(u,x,1))
        self.grad_uy = lambdify([x,y], diff(u,y,1))
        self.a = a
        self.b = b
        self.domain = D

    def domain(self):
        return self.domain
    
    def solution(self, p):
        x = p[...,0]
        y = p[...,1]

        return self.u(x,y)
    
    def source(self,p):
        x = p[...,0]
        y = p[...,1]
        return  self.f(x,y)

    def gradient(self,p):
        x = p[...,0]
        y = p[...,1]
        val = bm.zeros_like(p)
        val[...,0] = self.grad_ux(x,y)
        val[...,1] = self.grad_uy(x,y)
        return val
    
    def dirichlet(self,p ):
        return self.solution(p)