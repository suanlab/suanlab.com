"""Independent isochron checks: exact rational identities and Cartesian ODE reference."""
from fractions import Fraction as F
from itertools import product
import cmath,json,math

def integrate(rhs, initial, times):
    # Independent Cartesian RK4. This comparison is not an interval certificate.
    state=list(initial); prior=0.; rows=[]
    for target in times:
        steps=max(1,math.ceil((target-prior)/.0005));h=(target-prior)/steps;t=prior
        for _ in range(steps):
            k1=rhs(t,state);k2=rhs(t+h/2,[state[i]+h*k1[i]/2 for i in range(2)]);k3=rhs(t+h/2,[state[i]+h*k2[i]/2 for i in range(2)]);k4=rhs(t+h,[state[i]+h*k3[i]for i in range(2)])
            state=[state[i]+h*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6 for i in range(2)];t+=h
        rows.append(state[:]);prior=target
    return rows


def verify():
    exact=0
    for mu,g,c,omega,x,y in product([F(1,4),F(1),F(2)],[F(1),F(2)],[F(-2),F(0),F(1)],[F(1),F(3)],[F(-1),F(1,2),F(2)],[F(0),F(1),F(2)]):
        r2=x*x+y*y;a=c/g;om=omega-c*mu/g
        if om==0:continue
        z=[(-y-a*x)/r2,(x-a*y)/r2]
        f=[(mu-g*r2)*x-(omega-c*r2)*y,(omega-c*r2)*x+(mu-g*r2)*y]
        assert sum(z[i]*f[i]for i in range(2))==om
        assert sum(t*t for t in z)==(1+a*a)/r2
        # Hessian of theta - a log(r/r*) from differentiating Cartesian components.
        H=[[(2*x*y+a*(x*x-y*y))/r2**2,(y*y-x*x+2*a*x*y)/r2**2],[(y*y-x*x+2*a*x*y)/r2**2,(-2*x*y+a*(y*y-x*x))/r2**2]]
        J=[[mu-g*(3*x*x+y*y)+2*c*x*y,-omega+c*(x*x+3*y*y)-2*g*x*y],[omega-c*(3*x*x+y*y)-2*g*x*y,mu-g*(x*x+3*y*y)-2*c*x*y]]
        assert all(sum(H[i][j]*f[j]+J[j][i]*z[j]for j in range(2))==0 for i in range(2))
        # Exact radial derivative of v = mu/(g r²)-1.
        v=mu/(g*r2)-1
        assert -2*mu*(x*f[0]+y*f[1])/(g*r2*r2)==-2*mu*v
        exact+=1
    flows=0;max_phase=0;max_v=0
    for mu,g,c,rscale,theta in product([.25,1.,2.],[1.,2.],[-1.,0.,1.],[.5,1.,1.5],[0.,.7]):
        omega=3.;a=c/g;rs=math.sqrt(mu/g);r0=rscale*rs;om=omega-c*mu/g
        def rhs(t,s):
            x,y=s;r2=x*x+y*y
            return [(mu-g*r2)*x-(omega-c*r2)*y,(omega-c*r2)*x+(mu-g*r2)*y]
        times=[0.,.1,.5,1.,2.]
        rows=integrate(rhs,[r0*math.cos(theta),r0*math.sin(theta)],times)
        for t,(x,y) in zip(times,rows):
            radius=math.hypot(x,y);phi=math.atan2(y,x)-a*math.log(radius/rs)
            target=theta-a*math.log(r0/rs)+om*t
            pe=abs(cmath.exp(1j*phi)-cmath.exp(1j*target));ve=abs(rs*rs/(radius*radius)-1-(1/rscale**2-1)*math.exp(-2*mu*t));max_phase=max(max_phase,pe);max_v=max(max_v,ve)
            assert pe<2e-10 and ve<2e-10
        flows+=1
    pulses=0;max_ratio=0
    for a,radius,theta,mag,direction in product([-2.,0.,1.,3.],[.5,1.,2.],[0.,.7,2.],[.001,.01,.1,.5,.9],[-2.,0.,1.,2.]):
        w=mag*cmath.exp(1j*direction);before=radius*cmath.exp(1j*theta);after=before+radius*w*cmath.exp(1j*theta)
        actual=cmath.exp(1j*(cmath.phase(after)-a*math.log(abs(after)/radius)-theta))
        delta=cmath.log(1+w);shift=delta.imag-a*delta.real
        assert abs(actual-cmath.exp(1j*shift))<2e-14
        linear=w.imag-a*w.real;error=abs(shift-linear);bound=math.sqrt(1+a*a)*mag*mag/(2*(1-mag));assert error<=bound+1e-14
        max_ratio=max(max_ratio,error/bound);pulses+=1
    # Finite pulse ODE relaxation: phase shift retained, polar angle approaches it.
    relax=0;max_shift=0
    for a,kappa,theta in product([-1.,0.,1.],[-.5,.1,.5],[0.,.7]):
        start=(1+kappa)*cmath.exp(1j*theta)
        def rhs(t,s):
            x,y=s;r2=x*x+y*y
            return [(1-r2)*x-(3-a*r2)*y,(3-a*r2)*x+(1-r2)*y]
        x,y=integrate(rhs,[start.real,start.imag],[8.])[0];r=math.hypot(x,y);actual=math.atan2(y,x)-a*math.log(r)-theta-(3-a)*8;expected=-a*math.log1p(kappa)
        error=abs(cmath.exp(1j*actual)-cmath.exp(1j*expected));assert error<2e-10;max_shift=max(max_shift,error);relax+=1
    return {'status':'passed','exactRationalPreparations':exact,'cartesianFlowPreparations':flows,'flowObservationTimes':5,'complexPulsePreparations':pulses,'pulseRelaxationPreparations':relax,'maxPhaseCircleError':max_phase,'maxRadialCoordinateError':max_v,'maxPulseRelaxationError':max_shift,'maxRemainderBoundRatio':max_ratio,'numericalReference':'Independent Cartesian RK4 with step at most 0.0005 s; floating-point comparison, not a directed enclosure'}
if __name__=='__main__':print(json.dumps(verify(),indent=2))
