"""Independent finite checks of the explicitly added quench and piston models.

Markov flux integration and entropy quadrature are independent of the displayed
closed solution. A Decimal reciprocal-series ODE integrator checks the piston's
RK4 and integrated viscous loss. No production calculator is imported.
"""
from math import exp,log,isclose
from decimal import Decimal as D,localcontext
import json

def close(a,b,tol=3e-11):assert isclose(a,b,rel_tol=tol,abs_tol=tol),(a,b)
def quadrature(fn,t):
    n=8192;h=t/n
    return h/3*(fn(0)+fn(t)+sum((4 if i%2 else 2)*fn(i*h) for i in range(1,n)))
def entropy(p):return -p*log(p)-(1-p)*log(1-p)
def kl(p,q):return p*log(p/q)+(1-p)*log((1-p)/(1-q))

quench_cases=0
for ea in [.2,log(3),3.]:
    for eb in [.1,log(9),4.]:
        wa,wb=exp(-ea),exp(-eb);pa,pb=wa/(1+wa),wb/(1+wb)
        work=(eb-ea)*pa;delta_f=-log((1+wb)/(1+wa));excess=work-delta_f
        close(excess,kl(pa,pb));assert excess>0
        for gamma in [.4,2.]:
            k01,k10=gamma*pb,gamma*(1-pb)
            close((1-pb)*k01,pb*k10)
            for t in [.2,1.,3.]:
                def flux(p):return (1-p)*k01-p*k10
                p=pa;h=t/2048
                for _ in range(2048):
                    a=flux(p);b=flux(p+h*a/2);c=flux(p+h*b/2);d=flux(p+h*c)
                    p+=h*(a+2*b+2*c+d)/6
                exact=lambda s:pb+(pa-pb)*exp(-gamma*s)
                close(p,exact(t));heat=eb*(p-pa)
                generation=entropy(p)-entropy(pa)-heat
                close(generation,kl(pa,pb)-kl(p,pb))
                close(excess,kl(p,pb)+generation)
                close(eb*p-ea*pa,work+heat)
                def rate(s):
                    q=exact(s)
                    return gamma*(q-pb)*log(q*(1-pb)/(pb*(1-q)))
                close(quadrature(rate,t),generation)
                assert generation>=0 and rate(t)>=0
                quench_cases+=1
# Identical gaps do not supply quench work or a target mismatch.
for ea in [.2,1.,3.]:
    p=1/(1+exp(ea));assert kl(p,p)==0

def rk4_step(x,v,d,h):
    rhs=lambda x,v:(v,1/x-1-d*v)
    a,b=rhs(x,v);c,f=rhs(x+h*a/2,v+h*b/2)
    g,j=rhs(x+h*c/2,v+h*f/2);k,l=rhs(x+h*g,v+h*j)
    return x+h*(a+2*c+2*g+k)/6,v+h*(b+2*f+2*j+l)/6

def taylor_step(x,v,d,dt,order=24):
    # x=sum a_n t^n; 1/x=sum b_n t^n. Recur directly from x''+d x'=1/x-1.
    a=[D(0)]*(order+1);a[0]=x;a[1]=v;b=[]
    for n in range(order-1):
        b.append(1/x if n==0 else -sum(a[k]*b[n-k] for k in range(1,n+1))/x)
        a[n+2]=(b[n]-D(int(n==0))-d*D(n+1)*a[n+1])/D((n+1)*(n+2))
    powers=[dt**n for n in range(2*order+1)]
    xn=sum(a[n]*powers[n] for n in range(order+1))
    vn=sum(D(n)*a[n]*powers[n-1] for n in range(1,order+1))
    w=[D(n+1)*a[n+1] for n in range(order)]
    loss=d*sum(w[i]*w[j]*powers[i+j+1]/D(i+j+1) for i in range(order) for j in range(order))
    return xn,vn,loss

piston_cases=0;budget_cases=0;refined_cases=0
with localcontext() as ctx:
    ctx.prec=50
    for sx,sv,sd in [('1','.8','.4'),('1','.8','0'),('.5','0','.4'),('2','0','2'),('1.05','0','4'),('1','0','.4')]:
        x0,v0,d=D(sx),D(sv),D(sd);x,v=x0,v0;loss=D(0)
        E0=v0*v0/2+x0-1-x0.ln();states={}
        for step in range(1,61):
            x,v,inc=taylor_step(x,v,d,D('.05'));loss+=inc
            assert x>0
            if step in [10,20,60]:
                energy=v*v/2+x-1-x.ln()
                assert abs(E0-energy-loss)<D('1e-22')
                net_heat=x-x0+(v*v-v0*v0)/2
                generation=(x/x0).ln()-net_heat
                assert abs(generation-loss)<D('1e-22')
                assert loss>=0
                states[step*50]=(float(x),float(v));budget_cases+=1
        for dt in [.001,.0005]:
            xf,vf=float(x0),float(v0)
            count=round(3/dt)
            for step in range(1,count+1):
                xf,vf=rk4_step(xf,vf,float(d),dt)
                assert xf>0
                key=round(step*dt*1000)
                if key in states and abs(step*dt-key/1000)<1e-10:
                    target=states[key]
                    close(xf,target[0]);close(vf,target[1]);refined_cases+=1
        if sd=='0':assert abs(loss)==0
        piston_cases+=1
    # A coordinate at the minimum with nonzero velocity is not a resting state.
    x,v,loss=taylor_step(D(1),D('.8'),D('.4'),D('.05'))
    assert x>1 and x-1-x.ln()>0 and v*v/2+x-1-x.ln()<D('.32')

print(json.dumps({'status':'passed','quenchRelaxationBudgets':quench_cases,
 'identicalGapPreparations':3,'pistonTaylorTrajectories':piston_cases,
 'pistonHeatEntropyBudgets':budget_cases,'rk4RefinementComparisons':refined_cases,
 'positiveVolumeAndOvershootExample':True}))
