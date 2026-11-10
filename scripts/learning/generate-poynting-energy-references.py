"""Independent reference generator (Python, NumPy, SciPy, mpmath).
Run: python3 scripts/learning/generate-poynting-energy-references.py
No product imports. Finite differences and quadrature are observations, not bounds.
"""
from pathlib import Path
import json

def source_reference():
    from pathlib import Path
    import math,json,itertools
    import numpy as np
    from scipy.integrate import quad

    def d(f,x,h=1e-4):return(-f(x+2*h)+8*f(x+h)-8*f(x-h)+f(x-2*h))/(12*h)
    sources=[];pulses=[]
    for i,(b0,b1,b2,R)in enumerate(itertools.product([-1.,0.,1.],[-.5,0.,.5],[-.2,0.,.2],[.5,1.,2.])):
     eps=[.5,1.,2.][i%3];mu=[1.,.5,2.][i%3];H=1.2;t=.7;p=np.array([.2,-.3,.4])
     B=lambda time:np.array([0.,0.,b0+b1*time+.5*b2*time*time])
     E=lambda p,time:np.array([.5*(b1+b2*time)*p[1],-.5*(b1+b2*time)*p[0],0.])
     J=lambda p:np.array([-.5*eps*b2*p[1],.5*eps*b2*p[0],0.])
     energy=lambda p,time:.5*eps*float(E(p,time)@E(p,time))+.5*float(B(time)@B(time))/mu
     S=lambda p,time:np.cross(E(p,time),B(time))/mu
     divergence=0.
     for j in range(3):
      def component(x):
       q=p.copy();q[j]=x;return S(q,t)[j]
      divergence+=d(component,p[j])
     work=float(J(p)@E(p,t));local=d(lambda time:energy(p,time),t)+divergence+work
     U=lambda time:quad(lambda rad:energy(np.array([rad,0.,0.]),time)*2*math.pi*rad*H,0,R,epsabs=1e-11)[0]
     P=quad(lambda rad:float(J(np.array([rad,0.,0.]))@E(np.array([rad,0.,0.]),t))*2*math.pi*rad*H,0,R,epsabs=1e-11)[0]
     def surface(theta):
      n=np.array([math.cos(theta),math.sin(theta),0.]);return float(S(R*n,t)@n)*R*H
     outward=quad(surface,0,2*math.pi,epsabs=1e-11)[0]
     udot=d(U,t);global_residual=udot+outward+P
     assert abs(local)<1e-8 and abs(global_residual)<1e-7
     sources.append(dict(magneticOffset=b0,magneticRate=b1,magneticAcceleration=b2,permittivity=eps,permeability=mu,radius=R,height=H,time=t,energyDensity=energy(p,t),poynting=S(p,t).tolist(),fieldToMatterPowerDensity=work,independentLocalResidual=local,cylinderEnergy=U(t),independentEnergyRate=udot,outwardPoyntingPower=outward,fieldToMatterPower=P,independentCylinderResidual=global_residual))
    for eps,beta,width,A in itertools.product([.5,1.,2.],[0.,.5,1.],[.5,1.],[-1.,0.,1.]):
     mu=1.;c=1/math.sqrt(eps*mu);t=.3;z=.4;a,b=-1.,1.
     G=lambda x:A*math.exp(-x*x/(2*width*width))
     E=lambda z,time:beta*G(z-c*time)+(1-beta)*G(z+c*time)
     B=lambda z,time:(beta*G(z-c*time)-(1-beta)*G(z+c*time))/c
     u=lambda z,time:.5*eps*E(z,time)**2+.5*B(z,time)**2/mu
     S=lambda z,time:E(z,time)*B(z,time)/mu
     local=d(lambda time:u(z,time),t)+d(lambda x:S(x,t),z)
     U=lambda time:quad(lambda x:u(x,time),a,b,epsabs=1e-12)[0]
     outward=S(b,t)-S(a,t);udot=d(U,t);global_residual=udot+outward
     entire=quad(lambda x:u(x,t),-np.inf,np.inf,epsabs=1e-11)[0];expected=eps*A*A*width*math.sqrt(math.pi)*(beta*beta+(1-beta)**2)
     assert abs(local)<1e-8 and abs(global_residual)<1e-8 and abs(entire-expected)<1e-9
     density=u(z,t);velocity=None if density==0 else S(z,t)/density
     if velocity is not None:assert abs(velocity)<=c*(1+1e-14)
     pulses.append(dict(permittivity=eps,permeability=mu,amplitude=A,width=width,rightFraction=beta,z=z,time=t,energyDensity=density,poyntingZ=S(z,t),conditionalEnergyVelocity=velocity,speed=c,slabStart=a,slabEnd=b,slabEnergyPerArea=U(t),independentSlabEnergyRate=udot,outwardPowerPerArea=outward,independentSlabResidual=global_residual,independentLocalResidual=local,wholeLineEnergyPerArea=entire,analyticWholeLineEnergyPerArea=expected))
    # Passing energy balance is not sufficient: E=(0,k*x,0),B=J=0,static.
    negative=dict(electricField=[0.,1.,0.],magneticField=[0.,0.,0.],electricCurlZ=1.,magneticTimeDerivativeZ=0.,energyTimeDerivative=0.,poyntingDivergence=0.,fieldToMatterPower=0.,energyResidual=0.,faradayResidualZ=1.,fullMaxwellSolution=False)
    result=dict(provenance='Independent direct E/B/J energy and cross-product flux,spatial/time five-point differences,actual radial volume and oriented cylindrical surface quadrature,finite-slab and whole-line SciPy integration; no product imports.',sourceCases=sources,pulseCases=pulses,energyOnlyNegativeControl=negative,maxSourceLocalDifference=max(abs(r['independentLocalResidual'])for r in sources),maxSourceCylinderDifference=max(abs(r['independentCylinderResidual'])for r in sources),maxPulseSlabDifference=max(abs(r['independentSlabResidual'])for r in pulses),limitations=['Independent references for prescribed source and transverse pulse teaching families.','Constant positive coefficients only;not dispersive/hysteretic/general matter energy.','Prescribed bounded induction source and infinite-transverse plane pulses are not finite isolated devices.','Whole-line pulse integral is energy per transverse area,not finite total3D energy.','Poynting balance alone does not verify every Maxwell equation.','Observed errors are not certified bounds.'])

    return result

def cancellation_reference():
    from pathlib import Path
    import json,math,itertools
    import mpmath as mp
    from scipy.integrate import quad
    mp.mp.dps=180
    rows=[]
    for beta,width in itertools.product([0.,.5,1.],[1e-8,1e-12,1e-15]):
     a,b=1.,1.+width;t=.3
     def density(x):
      f=beta*math.exp(-(x-t)**2/2);g=(1-beta)*math.exp(-(x+t)**2/2);return f*f+g*g
     def primitive(x):return math.sqrt(math.pi)/2*(beta**2*math.erf(x-t)+(1-beta)**2*math.erf(x+t))
     stable=(b-a)*quad(lambda u:density(a+(b-a)*u),0,1,epsabs=1e-14)[0]
     A,B,T,Q=map(mp.mpf,[a,b,t,beta])
     ref=mp.sqrt(mp.pi)/2*(Q**2*(mp.erf(B-T)-mp.erf(A-T))+(1-Q)**2*(mp.erf(B+T)-mp.erf(A+T)))
     relative=abs(stable/float(ref)-1);assert relative<5e-15
     flux=lambda Z:Q**2*mp.exp(-(Z-T)**2)-(1-Q)**2*mp.exp(-(Z+T)**2)
     outward=flux(B)-flux(A)
     rows.append(dict(rightFraction=beta,slabStart=a,slabEnd=b,time=t,naivePrimitiveDifference=primitive(b)-primitive(a),stableQuadrature=stable,highPrecisionEnergyPerArea=float(ref),highPrecisionOutwardPowerPerArea=float(outward),relativeDifference=relative))
    # Near-balanced waves: evaluate energy flux from stable Gaussian difference.
    z,t=1.,1e-18;gp=math.exp(-(z-t)**2/2);gm=math.exp(-(z+t)**2/2);dg=gp*(-math.expm1(-2*z*t));stable=.25*dg*(gp+gm)
    Z,T=map(mp.mpf,[z,t]);F=mp.exp(-(Z-T)**2/2)/2;G=mp.exp(-(Z+T)**2/2)/2;reference=F*F-G*G
    assert abs(stable/float(reference)-1)<5e-15
    flux=dict(z=z,time=t,naivePoynting=(gp*gp-gm*gm)/4,stablePoynting=stable,highPrecisionPoynting=float(reference),highPrecisionEnergyDensity=float(F*F+G*G),conditionalEnergyVelocity=float(reference/(F*F+G*G)))
    # Nonzero absolute energy can be below double range although its normalized
    # analytic direction/velocity remains meaningful; zero amplitude is separate.
    underflow=[]
    for beta in[0.,.5,1.]:
     A=mp.mpf(1e-200);Q=mp.mpf(beta);density=A*A*(Q*Q+(1-Q)**2);S=A*A*(Q*Q-(1-Q)**2)
     underflow.append(dict(amplitude=1e-200,rightFraction=beta,mathematicallyNonzeroEnergy=True,logEnergyDensity=float(mp.log(density)),doubleEnergy=float(density),analyticConditionalVelocity=float(S/density),notAComputedDoubleQuotient=True))
    result=dict(provenance='180-digit mpmath references,normalized SciPy slab quadrature,and expm1 Gaussian difference; no product imports.',tinySlabs=rows,balancedFlux=flux,absoluteEnergyUnderflow=underflow,limitations=['Analytic conditional velocity is not an observed ratio when absolute energy is unavailable.','Exact zero amplitude has zero energy and undefined conditional energy velocity.','A tiny Gaussian tail is not a compactly supported pulse.'])

    return result

def scale_reference():
    from pathlib import Path
    import math,json
    import mpmath as mp
    mp.mp.dps=180
    mu=4*math.pi*1e-7;eps=1/(mu*299792458**2);b0,b1,b2,t,x,y=1e-6,.5e-6,.2e-6,.7,.2,-.3
    b=b0+b1*t+.5*b2*t*t;bp=b1+b2*t;r2=x*x+y*y
    magnetic=b*bp/mu;electric=eps*bp*b2*r2/4
    E,M,B0,B1,B2,T,X,Y=map(mp.mpf,[eps,mu,b0,b1,b2,t,x,y]);B=B0+B1*T+B2*T*T/2;BP=B1+B2*T;ref=E*BP*B2*(X*X+Y*Y)/4
    compatible=dict(magneticStorageRate=magnetic,poyntingDivergence=-magnetic,electricStorageRate=electric,fieldToMatterPower=-electric,naiveTotalResidual=(magnetic+electric)-magnetic-electric,stableGroupedResidual=(magnetic-magnetic)+(electric-electric))
    missing=dict(magneticStorageRate=magnetic,poyntingDivergence=-magnetic,electricStorageRate=electric,fieldToMatterPower=0.,naiveTotalResidual=(magnetic+electric)-magnetic,stableGroupedResidual=(magnetic-magnetic)+electric,highPrecisionResidual=float(ref),fullMaxwellSolution=False)
    assert missing['naiveTotalResidual']==0 and missing['stableGroupedResidual']>0
    assert abs(electric/float(ref)-1)<5e-15
    result=dict(provenance='Physical-scale SI coefficient example and180-digit exact-binary-input energy-component reference; no product imports.',compatible=compatible,missingConductionCurrent=missing,lesson='A dominant magnetic rate can hide a nonzero electric/source-work defect in a naive total. Show component powers and stable grouping; analytic grouping still needs independent verification.')

    return result

if __name__ == '__main__':
    result = dict(source=source_reference(), cancellation=cancellation_reference(), physicalScale=scale_reference())
    target = Path(__file__).resolve().parents[1] / 'fixtures/poynting-energy-independent.json'
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(sourceCases=len(result['source']['sourceCases']), pulseCases=len(result['source']['pulseCases']), tinySlabs=len(result['cancellation']['tinySlabs']), highPrecisionDigits=180)))
