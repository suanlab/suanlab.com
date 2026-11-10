"""Independent symbolic complex matrices; no production module imports."""
from pathlib import Path
import sympy as s,json,argparse
I=s.I
base=dict(stateReal0=1,stateImag0=0,stateReal1=1,stateImag1=0,referenceReal0=1,referenceImag0=0,referenceReal1=-1,referenceImag1=0,basisReal0=1,basisImag0=0,basisReal1=1,basisImag1=0,phaseA=1,phaseB=1,scaleNumerator=1,scaleDenominator=1)
specs=[('equal Z statistics orthogonal rays',{}),('positive imaginary relative phase',dict(stateReal1=0,stateImag1=1)),('negative imaginary relative phase',dict(stateReal1=0,stateImag1=-1)),('zero second component',dict(stateReal1=0)),('zero first component',dict(stateReal0=0)),('same ray imaginary scaling',dict(referenceReal0=0,referenceImag0=1,referenceReal1=0,referenceImag1=1)),('same ray negative scaling',dict(referenceReal0=-1,referenceReal1=-1)),('identity global phase',dict(phaseA=1,phaseB=0)),('negative global phase',dict(phaseA=0,phaseB=1)),('negative raw scaling',dict(scaleNumerator=-3,scaleDenominator=7)),('rational global phase',dict(phaseA=2,phaseB=1)),('computational basis',dict(basisReal0=1,basisReal1=0)),('imaginary basis',dict(basisReal1=0,basisImag1=1)),('general complex basis',dict(basisReal0=2,basisImag0=3,basisReal1=-4,basisImag1=5)),('general complex states',dict(stateReal0=-3,stateImag0=2,stateReal1=5,stateImag1=-7,referenceReal0=4,referenceImag0=-2,referenceReal1=1,referenceImag1=3)),('small first component',dict(stateReal0=1,stateReal1=1000000)),('small second component',dict(stateReal0=1000000,stateReal1=1)),('maximum complex integers',dict(stateReal0=1000000,stateImag0=-1000000,stateReal1=999999,stateImag1=1000000,referenceReal0=-999999,referenceImag0=1000000,referenceReal1=1000000,referenceImag1=999999,basisReal0=-1000000,basisImag0=999999,basisReal1=1000000,basisImag1=-1000000)),('small global scale',dict(scaleNumerator=1,scaleDenominator=1000000)),('large global scale',dict(scaleNumerator=1000000)),('skew basis coefficient mass exceeds one',dict(stateReal0=0,stateReal1=1,referenceReal1=0)),('skew basis coefficient mass below one',dict(referenceReal1=1)),('phase pair rescaling',dict(phaseA=1000000,phaseB=1000000)),('complex reference overlap',dict(referenceReal0=0,referenceImag0=1,referenceReal1=1,referenceImag1=0))]
def scalar(x):
 x=s.simplify(x)
 return str(x) if x.is_Rational else 'decimal:'+str(s.N(x,170))
def encode(x):
 if isinstance(x,s.MatrixBase):return [[encode(x[i,j])for j in range(x.cols)]for i in range(x.rows)]
 if isinstance(x,dict):return{k:encode(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return[encode(v)for v in x]
 return scalar(x)
def complex_value(x):return dict(real=s.re(x).simplify(),imaginary=s.im(x).simplify())
def abs2(x):return s.expand(x*s.conjugate(x)).simplify()
cases=[]
for label,patch in specs:
 p={**base,**patch}
 def vector(prefix):return s.Matrix([p[f'{prefix}Real{i}']+I*p[f'{prefix}Imag{i}']for i in range(2)])
 z,w,b=map(vector,['state','reference','basis']);N=(z.H*z)[0];W=(w.H*w)[0];B=(b.H*b)[0];psi=z/s.sqrt(N);phi=w/s.sqrt(W);u,v=b
 U=s.Matrix([[u,-s.conjugate(v)],[v,s.conjugate(u)]])/s.sqrt(B);az=U.H*psi;aw=U.H*phi;g=s.Rational(p['scaleNumerator'],p['scaleDenominator'])*(p['phaseA']+I*p['phaseB'])**2/(p['phaseA']**2+p['phaseB']**2);gz=g*z;G=(gz.H*gz)[0];gp=gz/s.sqrt(G);ov=(psi.H*phi)[0];go=(psi.H*gp)[0];cross=s.conjugate(z[0])*z[1];non=s.Matrix([[1,-1],[0,1]])*psi;Gram=s.Matrix([[1,1],[1,2]])
 expected=dict(rawNormSquared=N,referenceRawNormSquared=W,basisRawNormSquared=B,globalRawNormSquared=G,stateAmplitudes=[complex_value(x)for x in psi],referenceAmplitudes=[complex_value(x)for x in phi],basisColumns=[[complex_value(U[i,j])for i in range(2)]for j in range(2)],basisCoordinates=[complex_value(x)for x in az],referenceBasisCoordinates=[complex_value(x)for x in aw],standardProbabilities=[abs2(x)for x in psi],referenceStandardProbabilities=[abs2(x)for x in phi],basisProbabilities=[abs2(x)for x in az],referenceBasisProbabilities=[abs2(x)for x in aw],overlap=complex_value(ov),reverseOverlap=complex_value(s.conjugate(ov)),fidelity=abs2(ov),fidelityDefect=1-abs2(ov),globalNormalizedOverlap=complex_value(go),globalAmplitudes=[complex_value(x)for x in gp],globalBasisProbabilities=[abs2(x)for x in U.H*gp],globalRepresentativeDistanceSquared=(2-2*s.re(go)).simplify(),bloch=[2*s.re(cross)/N,2*s.im(cross)/N,(abs2(z[0])-abs2(z[1]))/N],plusXProbability=(N+2*s.re(cross))/(2*N),plusYProbability=(N+2*s.im(cross))/(2*N),nonorthogonal=dict(coordinates=[complex_value(x)for x in non],naiveCoefficientSquaredSum=(non.H*non)[0],gramNormSquared=(non.H*Gram*non)[0],normResidual=(non.H*Gram*non)[0]-1))
 cases.append(dict(label=label,input=p,expected=encode(expected)))
result=dict(method='Independent SymPy normalized complex vectors, conjugate transpose and unitary matrix multiplication; exact algebraic results,170-digit decimal irrational coordinates.',cases=cases)
a=argparse.ArgumentParser();a.add_argument('--output',default='scripts/fixtures/quantum-state-reference.json');args=a.parse_args();Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(preparations=len(cases))))
