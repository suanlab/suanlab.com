"""Independent finite-body, unrestored-device and joint-limit checks.

Integrate heat-driven body evolution, construct actual engine/leak/auxiliary
ports, and enumerate finite binomial counts separately from asymptotic formulas.
No product imports. Finite checks supplement, rather than replace, derivations.
"""
from math import exp, expm1, log, log1p, sqrt, comb, factorial, isclose
from fractions import Fraction as F
import json

def close(a,b,tol=2e-9):assert isclose(a,b,rel_tol=tol,abs_tol=tol),(a,b)

finite_cases=0;bounds=0
for Ch,Cc in [(1,1),(2,1),(1,3),(2.5,.7)]:
    for Th,Tc in [(600,300),(330,270),(500,499)]:
        lo,hi=Tc,Th
        for _ in range(100):
            mid=(lo+hi)/2
            entropy=Ch*log(mid/Th)+Cc*log(mid/Tc)
            if entropy>0:hi=mid
            else:lo=mid
        root=(lo+hi)/2
        Tf=Th*exp(Cc/(Ch+Cc)*log(Tc/Th));close(root,Tf)
        Ah=Ch*(Th-Tf);dt=Ah/4096;y=[Tc,0.]
        def field(q,state):
            hot=Th-q/Ch
            return [state[0]/(Cc*hot),1-state[0]/hot]
        for step in range(4096):
            q=step*dt;a=field(q,y)
            b=field(q+dt/2,[v+dt*w/2 for v,w in zip(y,a)])
            c=field(q+dt/2,[v+dt*w/2 for v,w in zip(y,b)])
            d=field(q+dt,[v+dt*w for v,w in zip(y,c)])
            y=[v+dt*(aa+2*bb+2*cc+dd)/6 for v,aa,bb,cc,dd in zip(y,a,b,c,d)]
        Wmax=Ch*Th+Cc*Tc-(Ch+Cc)*Tf
        close(y[0],Tf);close(y[1],Wmax)
        assert 0<Wmax/Ah<1-Tc/Th
        Ta=(Ch*Th+Cc*Tc)/(Ch+Cc)
        close(Ch*log(Ta/Th)+Cc*log(Ta/Tc),(Ch+Cc)*log(Ta/Tf))
        for end in [Tf,(Tf+Ta)/2,Ta,Ta+10]:
            generation=Ch*log(end/Th)+Cc*log(end/Tc)
            work=Ch*(Th-end)+Cc*(Tc-end)
            assert generation>=-1e-12 and work<=Wmax+1e-10
            close(Wmax-work,(Ch+Cc)*(end-Tf));bounds+=1
        if Ch==Cc:close(Tf,sqrt(Th*Tc))
        finite_cases+=1
for capacity in [1,2,5]:
    T=300.;Tf=T*exp(log(T/T)/2)
    assert 2*capacity*T-2*capacity*Tf==0
    assert capacity*(T-Tf)==0

# Exact rational composite realization: reversible engine + passive hot-to-cold
# leak + cold-isothermal auxiliary expansion, all ports assembled separately.
port_cases=0
Th,Tc,Qh=F(600),F(300),F(1000)
for delta_s in [F(0),F(1,2),F(1)]:
    for gen in [F(0),F(1,5),F(1,2)]:
        leak=gen/(1/Tc-1/Th);engine_heat=Qh-leak
        assert engine_heat>=0
        engine_work=engine_heat*(1-Tc/Th)
        cold_engine=-engine_heat*Tc/Th
        aux_heat=Tc*delta_s;work=engine_work+aux_heat
        cold=cold_engine-leak+aux_heat
        assert work==Qh+cold
        assert delta_s-Qh/Th-cold/Tc==gen
        apparent=-Qh/Th-cold/Tc
        assert apparent==gen-delta_s
        for reset_gen in [F(0),F(1,10)]:
            reset_input=Tc*(delta_s+reset_gen)
            reset_cold=-Tc*(delta_s+reset_gen)
            assert -delta_s-reset_cold/Tc==reset_gen
            assert work-reset_input==Qh*(1-Tc/Th)-Tc*(gen+reset_gen)
            assert work-reset_input<=Qh*(1-Tc/Th)
            port_cases+=1
# Algebra also permits nonzero stored energy, separate from the concrete gas.
for du,ds,g in [(F(100),F(1),F(1,5)),(F(-50),F(1,2),F(0))]:
    cold=Tc*(ds-Qh/Th-g);work=Qh+cold-du
    assert work==Qh*(1-Tc/Th)-(du-Tc*ds)-Tc*g

count_cases=0
for a in [-1.,0.,1.]:
    for N in [5,10,30,100]:
        x=log(N)+a;assert x>0
        q=1/(1+N*exp(a))
        law=[comb(N,m)*q**m*(1-q)**(N-m) for m in range(N+1)]
        close(sum(law),1)
        close(sum(m*p for m,p in enumerate(law)),N*q)
        close(sum((m-N*q)**2*p for m,p in enumerate(law)),N*q*(1-q))
        entropy=-sum(p*log(p) for p in [q,1-q])*N
        close(entropy,N*log1p(exp(-x))+x*N*q)
        close(law[0],exp(-N*log1p(exp(-x))))
        # Fixed-N energy response, independently differentiated in temperature.
        T=300/x;h=T*1e-5
        U=lambda t:300*N/(1+exp(300/t))
        close((U(T+h)-U(T-h))/(2*h),N*x*x*q*(1-q))
        count_cases+=1
limit_cases=0
for a in [-1.,0.,1.]:
    lam=exp(-a);N=10**8;x=log(N)+a;q=1/(1+N*exp(a))
    p0=exp(-N*log1p(exp(-x)))
    close(p0,exp(-lam),1e-7)
    S=N*log1p(exp(-x))+x*N*q
    close(S-lam*log(N),lam*(1+a),2e-6)
    C=N*x*x*q*(1-q);close(C/x**2,lam,1e-7)
    # Recurrence avoids large factorials and independently checks m=0..6.
    p=p0
    for m in range(7):
        close(p,exp(-lam)*lam**m/factorial(m),1e-7)
        p*=((N-m)/(m+1))*q/(1-q)
    limit_cases+=1
threshold_cases=0
for N in [1,10,1000,10**6]:
    for epsilon in [.01,.05,.25]:
        threshold=-log(expm1(-log1p(-epsilon)/N))
        all_ground=lambda x:exp(-N*log1p(exp(-x)))
        close(all_ground(threshold),1-epsilon)
        assert all_ground(threshold-.01)<1-epsilon<all_ground(threshold+.01)
        sufficient=log(N/epsilon)
        assert all_ground(sufficient)>=1-epsilon
        threshold_cases+=1

print(json.dumps({'status':'passed','finiteBodyIntegrations':finite_cases,
 'endpointWorkBounds':bounds,'equalTemperaturePreparations':3,
 'exactCompositeRunResetBudgets':port_cases,'nonzeroStoredEnergyBudgets':2,
 'finiteCountEnumerations':count_cases,'jointLimitChecks':limit_cases,
 'exactProbabilityThresholds':threshold_cases}))
