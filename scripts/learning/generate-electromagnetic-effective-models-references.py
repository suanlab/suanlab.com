"""Independent SciPy ODE/quadrature, matrix exponential/boundary solve and mpmath references.
Run from repository root. No product imports. Deterministic output; no timestamps.
Requires numpy, scipy and mpmath only in the authoring environment.
"""
from pathlib import Path
import json,math,itertools
import numpy as np
from scipy.integrate import solve_ivp,quad
from scipy.linalg import expm
import mpmath as mp
mp.mp.dps=240
lines=[]
for L,C,length,theta in itertools.product([.5,1.,2.],[.5,1.,2.],[.3,1.],[.02,.2,.8,1.4,2.,4.]):
 speed=1/math.sqrt(L*C);Z=math.sqrt(L/C);omega=theta*speed/length;voltage=1.
 sol=solve_ivp(lambda x,y:np.array([-1j*omega*L*y[1],-1j*omega*C*y[0]]),[length,0],np.array([1.+0j,0j]),rtol=3e-13,atol=2e-15,dense_output=True)
 Vin,Iin=sol.y[:,-1];Y=Iin/Vin;UE=quad(lambda x:.25*C*abs(sol.sol(x)[0]/Vin)**2,0,length,epsabs=2e-12)[0];UM=quad(lambda x:.25*L*abs(sol.sol(x)[1]/Vin)**2,0,length,epsabs=2e-12)[0]
 samples=[dict(fraction=u,voltage=float((sol.sol(u*length)[0]/Vin).real),currentImaginary=float((sol.sol(u*length)[1]/Vin).imag))for u in[0.,.2,.6,1.]]
 h=omega*1e-4
 B=lambda w:math.tan(w*length/speed)/Z
 derivative=(-B(omega+2*h)+8*B(omega+h)-8*B(omega-h)+B(omega-2*h))/(12*h)
 assert abs(Y.imag-math.tan(theta)/Z)<2e-10 and abs(UE+UM-.25*C*length/math.cos(theta)**2)<2e-10
 lines.append(dict(system=dict(inductancePerLength=L,capacitancePerLength=C,length=length,inputVoltage=voltage),theta=theta,inputSusceptance=float(Y.imag),electricCycleEnergy=UE,magneticCycleEnergy=UM,energyFromIndependentDerivative=.25*derivative,samples=samples))
bilayers=[]
for e1,e2,f,w in itertools.product([1.,4.,9.],[1.,4.,9.],[.2,.5,.8],[.02,.2,1.,1.5,2.,3.]):
 def matrix(e,width):return expm(1j*w*np.array([[0.,1.],[e,0.]])*width)
 M1=matrix(e1,f);M2=matrix(e2,1-f);M=M2@M1;stack=np.linalg.matrix_power(M,3);fw=np.array([1.,1.]);bw=np.array([1.,-1.]);r,t=np.linalg.solve(np.column_stack([stack@bw,-fw]),-stack@fw);R,T=abs(r)**2,abs(t)**2
 fields=[]
 for x in[0.,f*.5,f,1.,2.4,3.]:
  cell=min(2,int(x));z=x-cell
  P=(matrix(e1,z)if z<=f else matrix(e2,z-f)@M1)@np.linalg.matrix_power(M,cell)
  E,H=P@(fw+r*bw);fields.append(dict(position=x,electric=[float(E.real),float(E.imag)],normalizedMagnetic=[float(H.real),float(H.imag)],normalizedFlux=float((E*np.conj(H)).real)))
 D=float(np.trace(M).real/2);assert abs(R+T-1)<2e-12
 bilayers.append(dict(system=dict(epsilon1=e1,epsilon2=e2,fraction=f,period=.7),frequency=w,periods=3,matrix=dict(a=float(M[0,0].real),b=float(M[0,1].imag),c=float(M[1,0].imag),d=float(M[1,1].real)),halfTrace=D,reflectance=float(R),transmittance=float(T),reflection=[float(r.real),float(r.imag)],transmission=[float(t.real),float(t.imag)],fields=fields))
line_precision=[]
for theta in[1e-4,1e-8,1e-12,1e-20,1e-100,math.nextafter(math.pi/2,0),math.nextafter(math.pi/2,math.inf)]:
 x=mp.mpf(theta);ratio=mp.tan(x)/x;energy=1/mp.cos(x)**2
 line_precision.append(dict(theta=theta,correction=str(ratio-1),energyFitContrast=str(energy-ratio),magneticEnergyFraction=str((1-mp.sin(2*x)/(2*x))/2),susceptance=str(mp.tan(x))))
cell_precision=[]
for e1,e2,f,w in[(1.,9.,.5,w)for w in[1e-2,1e-4,1e-8,1e-12,1e-20,1e-50]]+[(1.,math.nextafter(1.,math.inf),.5,.7),(1.,9.,.5,1.),(.25,16.,.5,2.)]:
 E1,E2,F,W=map(mp.mpf,[e1,e2,f,w]);a=W*mp.sqrt(E1)*F;b=W*mp.sqrt(E2)*(1-F);D=mp.cos(a)*mp.cos(b)-(mp.sqrt(E1/E2)+mp.sqrt(E2/E1))*mp.sin(a)*mp.sin(b)/2
 cell_precision.append(dict(system=dict(epsilon1=e1,epsilon2=e2,fraction=f,period=1.),frequency=w,halfTraceMinusOne=str(D-1),halfTracePlusOne=str(D+1),classification='gap'if abs(D)>1 else'pass',principalPhase=str(mp.acos(D).real),attenuation=str(abs(mp.acos(D).imag))))
result=dict(method='Independent reverse ODE,spatial quadrature,matrix exponential,boundary linear solve and 240-digit exact-binary inputs. Selected evidence,not rigorous global bounds.',lineCases=lines,bilayerCases=bilayers,linePrecision=line_precision,bilayerPrecision=cell_precision)
p=Path('scripts/fixtures/electromagnetic-effective-models-independent.json');p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(lineCases=len(lines),bilayerCases=len(bilayers),linePrecision=len(line_precision),bilayerPrecision=len(cell_precision))))
