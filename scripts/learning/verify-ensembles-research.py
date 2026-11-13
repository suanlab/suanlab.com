"""Preparation calculations; these do not certify new reader content or controls."""
from fractions import Fraction as F
from itertools import product
from math import comb, exp, log, pi, cos, sin
import json

def near(a,b,tol=1e-11): assert abs(a-b)<tol,(a,b)
reservoirs=0
for N in range(2,13):
    for K in range(1,N):
        states=[x for x in product([0,1],repeat=N) if sum(x)==K]
        joint={(a,b):F(sum(x[:2]==(a,b) for x in states),len(states)) for a,b in product([0,1],repeat=2)}
        p=F(K,N); covariance=joint[1,1]-p*p
        assert covariance==-p*(1-p)/(N-1)
        assert 1+p*(1-p)/(-covariance)==N
        assert joint[1,0]+joint[1,1]==p
        independent={(a,b):(p if a else 1-p)*(p if b else 1-p) for a,b in joint}
        tv=sum(abs(joint[k]-independent[k]) for k in joint)/2
        assert tv==2*p*(1-p)/(N-1)
        reservoirs+=1
# Resolve a physical relaxation model explicitly, beyond the static ensemble.
# Energy in kB*T0, temperature in T0, entropy in kB; n=3.
# The declared relaxation rate gamma=1/s supplies the physical clock.
n=3; gamma=1.; r0=1/(1+exp(1)); C=n*r0*(1-r0)
responses=[]
for omega in [.5,1.,2.]:
    target=C/(1+1j*omega/gamma)
    errors=[]
    for amplitude in [.02,.01]:
        period=2*pi/omega; dt=period/4096; r=r0
        energy_fourier=0j; entropy=0.; duration=5*period
        def equilibrium(t):return 1/(1+exp(1/(1+amplitude*cos(omega*t))))
        def rate(t,r):return gamma*(equilibrium(t)-r)
        for step in range(15*4096+1):
            t=step*dt
            if step>=10*4096:
                weight=.5 if step in [10*4096,15*4096] else 1.
                energy_fourier+=weight*n*r*complex(cos(omega*t),-sin(omega*t))*dt
                entropy+=weight*(-n*rate(t,r)/(1+amplitude*cos(omega*t)))*dt
            if step==15*4096:break
            a=rate(t,r); b=rate(t+dt/2,r+dt*a/2); c=rate(t+dt/2,r+dt*b/2); d=rate(t+dt,r+dt*c)
            r+=dt*(a+2*b+2*c+d)/6
            assert 0<r<1
        measured=2*energy_fourier/(duration*amplitude)
        error=abs(measured-target); errors.append(error)
        assert error<.001
        leading=C*amplitude**2*omega**2/(2*gamma*(1+(omega/gamma)**2))
        assert entropy/duration>0
        assert abs(entropy/duration/leading-1)<.002
        responses.append({'omega':omega,'amplitude':amplitude,'responseError':error,'meanEntropyProduction':entropy/duration,'linearOrderPrediction':leading})
    assert errors[1]<errors[0]/3.8
# Compute covariance and Fisher information directly from all four labeled states.
grand=0
for energies in [(1.,1.),(1.,2.),(0.,2.)]:
    for beta,alpha in product([.5,1.,2.],[-1.,0.,1.]):
        states=list(product([0,1],repeat=2)); E=[sum(e*x for e,x in zip(energies,s)) for s in states]; N=[sum(s) for s in states]
        weights=[exp(alpha*k-beta*e) for e,k in zip(E,N)]; Z=sum(weights); prob=[w/Z for w in weights]
        mean=lambda a:sum(p*x for p,x in zip(prob,a))
        cov=lambda a,b:mean([x*y for x,y in zip(a,b)])-mean(a)*mean(b)
        vee,vnn,ven=cov(E,E),cov(N,N),cov(E,N)
        occupancy=[mean([s[i] for s in states]) for i in range(2)]; variances=[r*(1-r) for r in occupancy]
        near(vee,sum(e*e*v for e,v in zip(energies,variances)))
        near(ven,sum(e*v for e,v in zip(energies,variances)))
        det=vee*vnn-ven*ven; expected=variances[0]*variances[1]*(energies[0]-energies[1])**2
        near(det,expected)
        if energies[0]!=energies[1]:
            assert det>0
            logits=[log(r/(1-r)) for r in occupancy]
            recovered_beta=(logits[0]-logits[1])/(energies[1]-energies[0])
            near(recovered_beta,beta); near(energies[0]+logits[0]/recovered_beta,alpha/beta)
        else:near(det,0)
        mu=alpha/beta
        score_beta=[mu*k-e for e,k in zip(E,N)]; score_mu=[beta*k for k in N]
        fisher_det=cov(score_beta,score_beta)*cov(score_mu,score_mu)-cov(score_beta,score_mu)**2
        near(fisher_det,beta*beta*expected)
        grand+=1
print(json.dumps({'status':'passed','preparationOnly':True,'finiteReservoirPairLaws':reservoirs,'drivenCanonicalResponses':responses,'grandCovarianceAndFisherPreparations':grand,'newReadersValidated':0},indent=2))
